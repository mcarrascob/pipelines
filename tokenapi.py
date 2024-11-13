import os
from typing import List, Optional, Dict
from pydantic import BaseModel
import tiktoken
import aiohttp
import logging

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        target_user_roles: List[str] = ["user"]

    def __init__(self):
        self.type = "filter"
        self.name = "API-Based Token Limit Filter"
        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("TOKEN_LIMIT_PIPELINES", "*").split(","),
            }
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.api_url = os.getenv("TOKEN_API_URL", "http://host.docker.internal:8519")
        self.logger = logging.getLogger(__name__)

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def count_tokens(self, messages: List[dict]) -> int:
        token_count = 0
        for message in messages:
            token_count += len(self.tokenizer.encode(message.get('content', '')))
        return token_count

    async def get_user_info(self, username: str) -> Optional[int]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/get_tokens?username={username}") as response:
                    if response.status == 404:
                        raise Exception(f"User {username} not found in the system. Please make sure you're logged in correctly or contact support for assistance.")
                    elif response.status != 200:
                        raise Exception(f"Failed to get user info: {await response.text()}")
                    data = await response.json()
                    return data.get('tokens', 0)
        except aiohttp.ClientConnectorError:
            raise Exception(f"Cannot connect to API at {self.api_url}")

    async def use_tokens(self, username: str, tokens: int) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/use_tokens", json={"username": username, "tokens": tokens}) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to use tokens: {await response.text()}")
                    return True
        except aiohttp.ClientConnectorError:
            raise Exception(f"Cannot connect to API at {self.api_url}")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")
        print(body)
        print(user)

        if user and user.get("role") in self.valves.target_user_roles:
            username = user.get("name", "default_user")
            
            try:
                user_tokens = await self.get_user_info(username)
                
                if user_tokens <= 0:
                    raise Exception("Sorry, you don't have any tokens left. Please purchase more tokens to continue using our service.")
                
                incoming_tokens = self.count_tokens(body.get('messages', []))
                if incoming_tokens > user_tokens:
                    raise Exception(f"Your message requires {incoming_tokens} tokens, but you only have {user_tokens} available. Please shorten your message or purchase more tokens.")

                if not await self.use_tokens(username, incoming_tokens):
                    raise Exception("Failed to process your request. Please try again later or contact support.")

                self.logger.info(f"User {username} used {incoming_tokens} tokens. Remaining balance: {user_tokens - incoming_tokens}")
            
            except Exception as e:
                self.logger.error(f"Error processing request for user {username}: {str(e)}")
                raise

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if user and user.get("role") in self.valves.target_user_roles:
            username = user.get("name", "default_user")
            response_tokens = self.count_tokens([{'content': body.get('content', '')}])
            
            try:
                if not await self.use_tokens(username, response_tokens):
                    self.logger.warning(f"Failed to deduct {response_tokens} tokens for user {username}")
                else:
                    self.logger.info(f"Deducted {response_tokens} tokens for user {username}")
            except Exception as e:
                self.logger.error(f"Error deducting tokens for user {username}: {str(e)}")
                # We don't re-raise the exception here to avoid disrupting the response
                # But you might want to handle this differently based on your requirements

        return body
