import os
from typing import List, Optional, Dict, Tuple
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
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.api_url = os.getenv("TOKEN_API_URL", "http://host.docker.internal:8509")
        self.logger = logging.getLogger(__name__)
        
        # Custom error messages
        self.NO_TOKENS_ERROR = "Lo siento, no te quedan tokens. Por favor, contacta con los administradores para obtener más tokens."
        self.USER_NOT_FOUND_ERROR = "No pudimos encontrar tu cuenta en nuestro sistema. Por favor, asegúrate de haber iniciado sesión correctamente o contacta al soporte para obtener ayuda."

    def count_tokens(self, messages: List[dict]) -> int:
        token_count = 0
        for message in messages:
            token_count += len(self.tokenizer.encode(message.get('content', '')))
        return token_count

    async def get_user_info(self, user_id: str) -> Tuple[Optional[str], Optional[int]]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/get_user_info?user_id={user_id}") as response:
                    if response.status == 404:
                        self.logger.warning(f"User {user_id} not found in the API")
                        return None, None
                    elif response.status != 200:
                        self.logger.error(f"Failed to get user info: {await response.text()}")
                        return None, None
                    data = await response.json()
                    return data.get('username', user_id), data.get('tokens', 0)
        except aiohttp.ClientConnectorError:
            self.logger.error(f"Cannot connect to API at {self.api_url}")
            return None, None

    async def use_tokens(self, user_id: str, tokens: int) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/use_tokens", json={"username": user_id, "tokens": tokens}) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to use tokens: {await response.text()}")
                        return False
                    return True
        except aiohttp.ClientConnectorError:
            self.logger.error(f"Cannot connect to API at {self.api_url}")
            return False

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if user and user.get("role") == "user":
            user_id = user.get("id", "default_user")
            
            username, user_tokens = await self.get_user_info(user_id)
            
            if username is None:
                # User not found in API or API error
                body['messages'] = [{'role': 'system', 'content': self.USER_NOT_FOUND_ERROR}]
                return body
            
            if user_tokens <= 0:
                body['messages'] = [{'role': 'system', 'content': self.NO_TOKENS_ERROR}]
                return body
            
            # Calculate tokens for the incoming message
            incoming_tokens = self.count_tokens(body.get('messages', []))
            if incoming_tokens > user_tokens:
                body['messages'] = [{'role': 'system', 'content': f'Your message requires {incoming_tokens} tokens, but you only have {user_tokens} available. Please shorten your message or purchase more tokens.'}]
                return body

            # Deduct tokens for the incoming message
            if not await self.use_tokens(user_id, incoming_tokens):
                body['messages'] = [{'role': 'system', 'content': 'Failed to process your request. Please try again later or contact support.'}]
                return body

            self.logger.info(f"User {username} (ID: {user_id}) used {incoming_tokens} tokens. Remaining balance: {user_tokens - incoming_tokens}")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if user and user.get("role") == "user":
            user_id = user.get("id", "default_user")
            username, _ = await self.get_user_info(user_id)
            response_tokens = self.count_tokens([{'content': body.get('content', '')}])
            
            if not await self.use_tokens(user_id, response_tokens):
                self.logger.warning(f"Failed to deduct {response_tokens} tokens for user {username} (ID: {user_id})")
            else:
                self.logger.info(f"Deducted {response_tokens} tokens for user {username} (ID: {user_id})")

        return body
