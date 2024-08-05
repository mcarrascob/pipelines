from typing import List, Optional, Union
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
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Default to GPT-4 encoding

    # ... [previous methods remain unchanged] ...

    def count_tokens(self, messages: List[dict]) -> int:
        """Count tokens for a list of messages."""
        token_count = 0
        for message in messages:
            token_count += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                if isinstance(value, str):
                    token_count += len(self.tokenizer.encode(value))
                elif isinstance(value, (int, float)):
                    token_count += len(self.tokenizer.encode(str(value)))
                elif isinstance(value, bool):
                    token_count += 1  # 'true' or 'false'
                elif value is None:
                    continue  # Skip None values
                else:
                    # For complex types, we'll use their string representation
                    token_count += len(self.tokenizer.encode(repr(value)))
                
                if key == "name":  # If there's a name, the role is omitted
                    token_count -= 1  # Role is always required and always 1 token
        token_count += 2  # Every reply is primed with <im_start>assistant
        return token_count

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

        try:
            prompt_tokens = self.count_tokens(body["messages"])
        except Exception as e:
            print(f"Error counting tokens: {e}")
            prompt_tokens = 0  # Fallback value

        generation = trace.generation(
            name=body["chat_id"],
            model=body["model"],
            input=body["messages"],
            metadata={"interface": "open-webui", "prompt_tokens": prompt_tokens},
        )

        self.chat_generations[body["chat_id"]] = generation
        print(trace.get_trace_url())

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        if body["chat_id"] not in self.chat_generations:
            return body

        generation = self.chat_generations[body["chat_id"]]

        try:
            # Count tokens for the entire message history (prompt)
            prompt_tokens = self.count_tokens(body["messages"][:-1])  # Exclude the last message (assistant's response)

            # Count tokens for the generated message
            generated_message = get_last_assistant_message(body["messages"])
            completion_tokens = self.count_tokens([{"role": "assistant", "content": generated_message}])

            total_tokens = prompt_tokens + completion_tokens
        except Exception as e:
            print(f"Error counting tokens: {e}")
            prompt_tokens = completion_tokens = total_tokens = 0  # Fallback values

        generation.end(
            output=generated_message,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            metadata={"interface": "open-webui"},
        )

        return body
