from typing import List, Optional, Dict
import os
import tiktoken
from collections import defaultdict
from datetime import datetime, timedelta

from pydantic import BaseModel
from langfuse import Langfuse
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        secret_key: str
        public_key: str
        host: str

    def __init__(self):
        self.type = "filter"
        self.name = "Langfuse Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "your-secret-key-here"),
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "your-public-key-here"),
                "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            }
        )
        self.langfuse = None
        self.chat_generations = {}
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.global_usage = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0})
        self.last_report_time = datetime.now()

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        self.set_langfuse()

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        if self.langfuse:
            self.langfuse.flush()

    async def on_valves_updated(self):
        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=False,
            )
            self.langfuse.auth_check()
        except UnauthorizedError:
            print("Langfuse credentials incorrect. Please re-enter your Langfuse credentials in the pipeline settings.")
        except Exception as e:
            print(f"Langfuse error: {e} Please re-enter your Langfuse credentials in the pipeline settings.")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        }
        
        if model not in pricing:
            print(f"Warning: Unknown model '{model}'. Cost set to 0.")
            return 0.0
        
        input_cost = (input_tokens / 1000) * pricing[model]["input"]
        output_cost = (output_tokens / 1000) * pricing[model]["output"]
        return input_cost + output_cost

    def update_global_usage(self, model: str, input_tokens: int, output_tokens: int, cost: float):
        self.global_usage[model]["input_tokens"] += input_tokens
        self.global_usage[model]["output_tokens"] += output_tokens
        self.global_usage[model]["cost"] += cost

    def report_global_usage(self):
        current_time = datetime.now()
        if current_time - self.last_report_time >= timedelta(hours=1):
            print("\nGlobal Usage Report:")
            for model, usage in self.global_usage.items():
                print(f"Model: {model}")
                print(f"  Total Input Tokens: {usage['input_tokens']}")
                print(f"  Total Output Tokens: {usage['output_tokens']}")
                print(f"  Total Cost: ${usage['cost']:.6f}")
            print("\n")
            self.last_report_time = current_time

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")

        trace = self.langfuse.trace(
            name=f"filter:{__name__}",
            input=body,
            user_id=user["id"] if user else None,
            metadata={"name": user["name"] if user else None},
            session_id=body.get("chat_id"),
        )

        try:
            input_tokens = sum(self.count_tokens(msg["content"]) for msg in body.get("messages", []))
            print(f"Input tokens: {input_tokens}")
        except Exception as e:
            print(f"Error counting input tokens: {e}")
            print(f"Message content: {body.get('messages', [])}")
            input_tokens = 0  # Set a default value

        generation = trace.generation_start(
            name=body.get("chat_id", "unknown"),
            model=body.get("model", "unknown"),
            model_parameters=body.get("model_parameters", {}),
            input=body.get("messages", []),
            metadata={"interface": "open-webui", "input_tokens": input_tokens},
        )

        self.chat_generations[body.get("chat_id", "unknown")] = {
            "generation": generation,
            "input_tokens": input_tokens,
        }
        print(f"Trace URL: {trace.get_trace_url()}")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        chat_id = body.get("chat_id", "unknown")
        if chat_id not in self.chat_generations:
            print(f"Warning: No generation data found for chat_id {chat_id}")
            return body

        generation_data = self.chat_generations[chat_id]
        generation = generation_data["generation"]
        input_tokens = generation_data["input_tokens"]

        # Extract all relevant data from the API response
        api_response = body.get("content", "")
        finish_reason = body.get("finish_reason", "unknown")
        model = body.get("model", "unknown")
        created = body.get("created", None)
        response_id = body.get("id", "unknown")

        # Use the provided token counts if available, otherwise estimate
        model_input_tokens = body.get("usage", {}).get("prompt_tokens", input_tokens)
        model_output_tokens = body.get("usage", {}).get("completion_tokens", self.count_tokens(api_response))
        total_tokens = body.get("usage", {}).get("total_tokens", model_input_tokens + model_output_tokens)

        print(f"API response content: {api_response}")
        print(f"Finish reason: {finish_reason}")
        print(f"Model: {model}")
        print(f"Created: {created}")
        print(f"Response ID: {response_id}")
        print(f"Token counts - Input: {model_input_tokens}, Output: {model_output_tokens}, Total: {total_tokens}")

        total_cost = self.calculate_cost(model_input_tokens, model_output_tokens, model)

        # Update global usage
        self.update_global_usage(model, model_input_tokens, model_output_tokens, total_cost)

        generation.end(
            output=api_response,
            prompt_tokens=model_input_tokens,
            completion_tokens=model_output_tokens,
            metadata={
                "interface": "open-webui",
                "model": model,
                "finish_reason": finish_reason,
                "created": created,
                "response_id": response_id,
                "total_cost": total_cost,
            },
        )

        # Log the token counts, cost, and model for verification
        print(f"Input tokens: {model_input_tokens}")
        print(f"Output tokens: {model_output_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Total cost: ${total_cost:.6f}")

        # Report global usage periodically
        self.report_global_usage()

        # Clean up the stored data for this chat
        del self.chat_generations[chat_id]

        return body
