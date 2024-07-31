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
        self.tokenizer = tiktoken.get_encoding("o200k_base")  # Updated to use cl100k_base
        self.global_usage = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0})
        self.last_report_time = datetime.now()
        self.token_correction_factor = 1.05  # 5% correction factor

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
        token_count = 0
        for message in messages:
            token_count += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                token_count += len(self.tokenizer.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    token_count -= 1  # role is always required and always 1 token
        token_count += 2  # Every reply is primed with <im_start>assistant
        return int(token_count * self.token_correction_factor)  # Apply correction factor

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }
        
        if model not in pricing:
            print(f"Warning: Unknown model '{model}'. Cost set to 0.")
            return 0.0
        
        input_cost = (input_tokens / 1000) * pricing[model]["input"]
        output_cost = (output_tokens / 1000) * pricing[model]["output"]
        return input_cost + output_cost

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")

        trace = self.langfuse.trace(
            name=f"filter:{__name__}",
            input=body,
            user_id=user["id"],
            metadata={"name": user["name"]},
            session_id=body["chat_id"],
        )

        input_tokens = self.count_tokens(body["messages"])

        generation = trace.generation(
            name=body["chat_id"],
            model=body["model"],
            input=body["messages"],
            metadata={"interface": "open-webui", "input_tokens": input_tokens},
        )

        self.chat_generations[body["chat_id"]] = {
            "generation": generation,
            "input_tokens": input_tokens
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

        # Count tokens for all messages, including the new generated message
        all_tokens = self.count_tokens(body["messages"])
        output_tokens = all_tokens - input_tokens

        total_cost = self.calculate_cost(input_tokens, output_tokens, body["model"])

        # Update global usage
        self.update_global_usage(body["model"], input_tokens, output_tokens, total_cost)

        usage_data = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": all_tokens,
        }

        # Only add cost to usage_data if it's not zero
        if total_cost > 0:
            usage_data["total_cost"] = total_cost

        generation.end(
            output=body["messages"][-1]["content"],  # Assuming the last message is the generated one
            usage=usage_data,
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
        print(f"Total tokens: {all_tokens}")
        if total_cost > 0:
            print(f"Total cost: ${total_cost:.6f}")
        else:
            print("Total cost: N/A (Unknown model)")

        # Report global usage periodically
        self.report_global_usage()

        return body
