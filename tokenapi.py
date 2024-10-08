import os
from typing import List, Optional, Dict
from pydantic import BaseModel
import tiktoken
import aiohttp
import json

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        max_tokens_per_user: int

    def __init__(self):
        self.type = "filter"
        self.name = "API-Based Token Limit Filter"
        self.valves = self.Valves(
            pipelines=["*"],
            max_tokens_per_user=int(os.getenv("MAX_TOKENS_PER_USER", 1000000))  # Default to 1 million tokens
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Use a single tokenizer for simplicity
        self.api_url = os.getenv("TOKEN_API_URL", "http://localhost:5000")  # Set your API URL here

    def count_tokens(self, messages: List[dict]) -> int:
        """Count tokens for a list of messages."""
        token_count = 0
        for message in messages:
            token_count += len(self.tokenizer.encode(message.get('content', '')))
        return token_count

    async def check_and_use_tokens(self, user_id: str, tokens: int) -> bool:
        """Check if user has enough tokens and use them if available."""
        async with aiohttp.ClientSession() as session:
            # First, check the user's token balance
            async with session.get(f"{self.api_url}/get_tokens?username={user_id}") as response:
                if response.status != 200:
                    raise Exception(f"Failed to get token balance: {await response.text()}")
                balance = (await response.json())['tokens']

            if balance < tokens:
                return False

            # If enough tokens, use them
            async with session.post(f"{self.api_url}/use_tokens", json={"username": user_id, "tokens": tokens}) as response:
                if response.status != 200:
                    raise Exception(f"Failed to use tokens: {await response.text()}")

        return True

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if user and user.get("role") == "user":
            user_id = user.get("id", "default_user")
            
            # Count tokens in the new message and all previous messages
            new_tokens = self.count_tokens(body.get("messages", []))
            
            # Check if user has enough tokens and use them
            if not await self.check_and_use_tokens(user_id, new_tokens):
                raise Exception("Insufficient tokens. Message blocked.")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # We don't need to do anything in the outlet for this version
        return body
