import os
from typing import List, Optional, Dict
from pydantic import BaseModel
import tiktoken

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        max_tokens_per_user: int

    def __init__(self):
        self.type = "filter"
        self.name = "Simple Token Limit Filter"

        self.valves = self.Valves(
            pipelines=["*"],
            max_tokens_per_user=int(os.getenv("MAX_TOKENS_PER_USER", 1000000))  # Default to 1 million tokens
        )

        self.user_tokens: Dict[str, int] = {}  # Dictionary to track user tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Use a single tokenizer for simplicity

    def count_tokens(self, messages: List[dict]) -> int:
        """Count tokens for a list of messages."""
        token_count = 0
        for message in messages:
            token_count += len(self.tokenizer.encode(message.get('content', '')))
        return token_count

    def token_limited(self, user_id: str, new_tokens: int) -> bool:
        current_tokens = self.user_tokens.get(user_id, 0)
        return current_tokens + new_tokens > self.valves.max_tokens_per_user

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if user and user.get("role") == "user":
            user_id = user.get("id", "default_user")
            
            # Count tokens in the new message and all previous messages
            new_tokens = self.count_tokens(body.get("messages", []))
            
            # Check if adding these tokens would exceed the limit
            if self.token_limited(user_id, new_tokens):
                raise Exception("Token limit exceeded. Message blocked.")

            # If not exceeded, update the token count
            self.user_tokens[user_id] = self.user_tokens.get(user_id, 0) + new_tokens

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # We don't need to do anything in the outlet for this simple version
        return body
