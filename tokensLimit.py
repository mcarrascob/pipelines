import os
from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import tiktoken

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        max_tokens_per_user: Optional[int] = None

    def __init__(self):
        self.type = "filter"
        self.name = "Token Limit Filter"

        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("TOKEN_LIMIT_PIPELINES", "*").split(","),
                "max_tokens_per_user": int(
                    os.getenv("MAX_TOKENS_PER_USER", 1000000)  # Default to 1 million tokens
                ),
            }
        )

        self.user_tokens = {}  # Dictionary to track user tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Initialize tokenizer

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def count_tokens(self, messages: List[OpenAIChatMessage]) -> int:
        return sum(len(self.tokenizer.encode(str(message))) for message in messages)

    def token_limited(self, user_id: str, new_tokens: int) -> bool:
        current_tokens = self.user_tokens.get(user_id, 0)
        return current_tokens + new_tokens > self.valves.max_tokens_per_user

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")
        print(body)
        print(user)

        if user and user.get("role", "admin") == "user":
            user_id = user.get("id", "default_user")
            
            # Count tokens in the new message and context
            new_message_tokens = self.count_tokens([body.get("message", {})])
            context_tokens = self.count_tokens(body.get("context", []))
            total_new_tokens = new_message_tokens + context_tokens

            # Check token limit
            if self.token_limited(user_id, total_new_tokens):
                raise Exception("Token limit exceeded. Please try again later.")

            # Update token count
            self.user_tokens[user_id] = self.user_tokens.get(user_id, 0) + total_new_tokens

        return body
