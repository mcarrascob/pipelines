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
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")  # Use GPT-4 tokenizer for more accurate counts
        self.global_usage = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0})
        self.last_report_time = datetime.now()

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        self.set_langfuse()

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
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

    def count_tokens(self, messages: List[dict]) -> int:
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(self.tokenizer.encode(value))
                elif isinstance(value, (int, float)):
                    num_tokens += len(self.tokenizer.encode(str(value)))
                elif isinstance(value, bool):
                    num_tokens += 1  # 'true' or 'false'
                elif isinstance(value, (list, dict)):
                    # For complex types, we'll use a simple estimation
                    num_tokens += len(self.tokenizer.encode(str(value)))
                elif value is None:
                    num_tokens += 1  # 'null'
                else:
                    print(f"Warning: Unexpected type {type(value)} for key {key}")
                    num_tokens += 1  # Add a token as a precaution
                
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        pricing = {
            "gpt-4o": {"input": 0.03, "output": 0.06},
            "gpt-4o-mini": {"input": 0.001, "output": 0.002},
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
            user_id=user["id"],
            metadata={"name": user["name"]},
            session_id=body["chat_id"],
        )

        try:
            input_tokens = self.count_tokens(body["messages"])
        except Exception as e:
            print(f"Error counting tokens: {e}")
            print(f"Message content: {body['messages']}")
            input_tokens = 0  # Set a default value

        generation = trace.generation(
            name=body["chat_id"],
            model=body["model"],
            input=body["messages"],
            metadata={"interface": "open-webui", "input_tokens": input_tokens},
        )

        self.chat_generations[body["chat_id"]] = {
            "generation": generation,
            "input_tokens": input_tokens,
            "messages": body["messages"]  # Store the input messages
        }
        print(trace.get_trace_url())

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        if body["chat_id"] not in self.chat_generations:
            return body

        generation_data = self.chat_generations[body["chat_id"]]
        generation = generation_data["generation"]
        input_tokens = generation_data["input_tokens"]
        input_messages = generation_data["messages"]

        try:
            # Calculate tokens for the new generated message only
            output_tokens = self.count_tokens([body["messages"][-1]])
        except Exception as e:
            print(f"Error counting output tokens: {e}")
            print(f"Output message content: {body['messages'][-1]}")
            output_tokens = 0  # Set a default value

        total_cost = self.calculate_cost(input_tokens, output_tokens, body["model"])

        # Update global usage
        self.update_global_usage(body["model"], input_tokens, output_tokens, total_cost)

        generation.end(
            output=body["messages"][-1]["content"],  # Assuming the last message is the generated one
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "total_cost": total_cost,
            },
            metadata={
                "interface": "open-webui",
                "output_tokens": output_tokens,
                "model": body["model"],
            },
        )

        # Log the token counts, cost, and model for verification
        print(f"Model: {body['model']}")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Total tokens: {input_tokens + output_tokens}")
        print(f"Total cost: ${total_cost:.6f}")

        # Report global usage periodically
        self.report_global_usage()

        # Clean up the stored data for this chat
        del self.chat_generations[body["chat_id"]]

        return body
