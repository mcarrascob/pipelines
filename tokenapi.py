import os
from typing import List, Optional, Dict
from pydantic import BaseModel
import tiktoken
import aiohttp
import json
import logging

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0

    def __init__(self):
        self.type = "filter"
        self.name = "API-Based Token Limit Filter"
        self.valves = self.Valves(pipelines=["*"])
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Use a single tokenizer for simplicity
        self.api_url = os.getenv("TOKEN_API_URL", "http://host.docker.internal:8509")  # Set your API URL here
        self.logger = logging.getLogger(__name__)

    def count_tokens(self, messages: List[dict]) -> int:
        token_count = 0
        for message in messages:
            token_count += len(self.tokenizer.encode(message.get('content', '')))
        return token_count

    async def check_user_tokens(self, user_id: str) -> int:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/get_tokens?username={user_id}") as response:
                    if response.status == 404:
                        return 0
                    elif response.status != 200:
                        self.logger.error(f"Failed to get token balance: {await response.text()}")
                        return -1  # Indicate an error occurred
                    return (await response.json())['tokens']
        except aiohttp.ClientConnectorError:
            self.logger.error(f"Cannot connect to token API at {self.api_url}")
            return -1  # Indicate an error occurred

    async def use_tokens(self, user_id: str, tokens: int) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/use_tokens", json={"username": user_id, "tokens": tokens}) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to use tokens: {await response.text()}")
                        return False
                    return True
        except aiohttp.ClientConnectorError:
            self.logger.error(f"Cannot connect to token API at {self.api_url}")
            return False

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if user and user.get("role") == "user":
            user_id = user.get("id", "default_user")
            
            user_tokens = await self.check_user_tokens(user_id)
            
            if user_tokens == -1:
                self.logger.warning("Token API unavailable. Proceeding without token check.")
            elif user_tokens <= 0:
                # Instead of setting a 'stop' flag, we'll add an informative message
                body['messages'] = [{'role': 'system', 'content': 'You have no available tokens. Please recharge your account to continue using the service.'}]
            else:
                body['user_tokens'] = user_tokens

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if user and user.get("role") == "user" and 'user_tokens' in body:
            user_id = user.get("id", "default_user")
            response_tokens = self.count_tokens([{'content': body.get('content', '')}])
            
            if not await self.use_tokens(user_id, response_tokens):
                self.logger.warning(f"Failed to deduct {response_tokens} tokens for user {user_id}")
            
            del body['user_tokens']

        return body
