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

    def __init__(self):
        self.type = "filter"
        self.name = "API-Based Token Limit Filter"
        self.valves = self.Valves(pipelines=["*"])
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Use a single tokenizer for simplicity
        self.api_url = os.getenv("TOKEN_API_URL", "http://localhost:5000")  # Set your API URL here

    def count_tokens(self, messages: List[dict]) -> int:
        """Count tokens for a list of messages."""
        token_count = 0
        for message in messages:
            token_count += len(self.tokenizer.encode(message.get('content', '')))
        return token_count

    async def check_user_tokens(self, user_id: str) -> int:
        """Check user's token balance."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}/get_tokens?username={user_id}") as response:
                if response.status == 404:
                    return 0  # User not found in the database
                elif response.status != 200:
                    raise Exception(f"Failed to get token balance: {await response.text()}")
                return (await response.json())['tokens']

    async def use_tokens(self, user_id: str, tokens: int) -> bool:
        """Use tokens for a user."""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.api_url}/use_tokens", json={"username": user_id, "tokens": tokens}) as response:
                if response.status != 200:
                    raise Exception(f"Failed to use tokens: {await response.text()}")
                return True

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if user and user.get("role") == "user":
            user_id = user.get("id", "default_user")
            
            # Check user's token balance
            user_tokens = await self.check_user_tokens(user_id)
            
            if user_tokens <= 0:
                # User not in DB or has no tokens
                body['messages'] = []  # Clear messages to prevent LLM response
                body['stop'] = True  # Signal to stop processing
            else:
                # User has tokens, attach token count to body for later use
                body['user_tokens'] = user_tokens

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if user and user.get("role") == "user" and 'user_tokens' in body:
            user_id = user.get("id", "default_user")
            
            # Count tokens in the LLM response
            response_tokens = self.count_tokens([{'content': body.get('content', '')}])
            
            # Use (deduct) tokens
            await self.use_tokens(user_id, response_tokens)
            
            # Remove the user_tokens field from the body
            del body['user_tokens']

        return body
