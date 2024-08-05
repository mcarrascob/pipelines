from typing import List, Optional
from schemas import OpenAIChatMessage
import os
import uuid
import tiktoken

from utils.pipelines.main import get_last_user_message, get_last_assistant_message
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
            "gpt-4o": {"input": 0.03, "output": 0.06},  # Assuming gpt-4o pricing is same as gpt-4
        }
        
        if model not in pricing:
            print(f"Warning: Unknown model '{model}'. Cost set to 0.")
            return 0.0
        
        input_cost = (input_tokens / 1000) * pricing[model]["input"]
        output_cost = (output_tokens / 1000) * pricing[model]["output"]
        return input_cost + output_cost

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")
        print(f"Received body: {body}")
        print(f"User: {user}")

        if "chat_id" not in body:
            unique_id = f"SYSTEM MESSAGE {uuid.uuid4()}"
            body["chat_id"] = unique_id
            print(f"chat_id was missing, set to: {unique_id}")

        required_keys = ["model", "messages"]
        missing_keys = [key for key in required_keys if key not in body]
        
        if missing_keys:
            error_message = f"Error: Missing keys in the request body: {', '.join(missing_keys)}"
            print(error_message)
            raise ValueError(error_message)

        trace = self.langfuse.trace(
            name=f"filter:{__name__}",
            input=body,
            user_id=user["id"] if user else None,
            metadata={"name": user["name"] if user else None},
            session_id=body["chat_id"],
        )

        input_tokens = sum(self.count_tokens(msg["content"]) for msg in body["messages"])

        generation = trace.generation(
            name=body["chat_id"],
            model=body["model"],
            input=body["messages"],
            metadata={
                "interface": "open-webui",
                "input_tokens": input_tokens,
                "model_parameters": body.get("model_parameters", {}),
            },
        )

        self.chat_generations[body["chat_id"]] = {
            "generation": generation,
            "input_tokens": input_tokens,
        }
        print(f"Trace URL: {trace.get_trace_url()}")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        if body["chat_id"] not in self.chat_generations:
            return body

        generation_data = self.chat_generations[body["chat_id"]]
        generation = generation_data["generation"]
        input_tokens = generation_data["input_tokens"]

        user_message = get_last_user_message(body["messages"])
        generated_message = get_last_assistant_message(body["messages"])

        output_tokens = self.count_tokens(generated_message)
        total_tokens = input_tokens + output_tokens

        total_cost = self.calculate_cost(input_tokens, output_tokens, body["model"])

        generation.end(
            output=generated_message,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            metadata={
                "interface": "open-webui",
                "model": body["model"],
                "total_cost": total_cost,
            },
        )

        # Clean up the stored data for this chat
        del self.chat_generations[body["chat_id"]]

        return body
