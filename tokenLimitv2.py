import os
from typing import List, Optional, Dict
from pydantic import BaseModel
import tiktoken
import aiohttp
import logging
import json

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
        self.api_url = os.getenv("TOKEN_API_URL", "http://host.docker.internal:8509")
        self.logger = logging.getLogger(__name__)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.token_info = {}  # New attribute to store token information

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def count_tokens(self, messages: List[dict]) -> int:
        """Count tokens for a list of messages."""
        token_count = 0
        for message in messages:
            token_count += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                token_count += len(self.tokenizer.encode(str(value)))
            
            if "name" in message:  # If there's a name, the role is omitted
                token_count -= 1  # Role is always required and always 1 token

        token_count += 2  # Every reply is primed with <im_start>assistant
        
        # Adjust for the chain of messages
        if len(messages) > 1:
            token_count -= 2 * (len(messages) - 1)  # Subtract 2 for each message after the first

        return token_count

    def count_output_tokens(self, message: str) -> int:
        """Count tokens for the output message, accounting for potential JSON structure."""
        try:
            # Try to parse the message as JSON
            parsed_message = json.loads(message)
            # If successful, count tokens for the entire JSON structure
            return len(self.tokenizer.encode(json.dumps(parsed_message)))
        except json.JSONDecodeError:
            # If not JSON, count tokens for the raw string
            return len(self.tokenizer.encode(message))

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
        print(f"inlet:{__name__}")
        print(f"Received body: {body}")
        print(f"User: {user}")

        if user and user.get("role") in self.valves.target_user_roles:
            username = user.get("name", "default_user")
            
            try:
                user_tokens = await self.get_user_info(username)
                
                if user_tokens <= 0:
                    raise Exception("Sorry, you don't have any tokens left. Please purchase more tokens to continue using our service.")
                
                # Calculate tokens for the entire conversation context
                incoming_tokens = self.count_tokens(body.get('messages', []))
                
                if incoming_tokens > user_tokens:
                    raise Exception(f"Your conversation requires {incoming_tokens} tokens, but you only have {user_tokens} available. Please shorten your message or purchase more tokens.")

                self.logger.info(f"User {username} conversation requires {incoming_tokens} tokens. Current balance: {user_tokens}")
                
                # Store token information in the class attribute
                self.token_info[username] = {
                    'incoming_tokens': incoming_tokens,
                    'user_tokens': user_tokens
                }
            
            except Exception as e:
                self.logger.error(f"Error processing request for user {username}: {str(e)}")
                raise

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if user and user.get("role") in self.valves.target_user_roles:
            username = user.get("name", "default_user")

            try:
                token_info = self.token_info.pop(username, None)
                if token_info is None:
                    raise Exception("Token information not found. Make sure inlet() was called before outlet().")

                # Calculate tokens for the new response
                response_tokens = self.count_output_tokens(body.get('content', ''))

                # Total tokens to deduct
                tokens_to_deduct = token_info['incoming_tokens'] + response_tokens

                if await self.use_tokens(username, tokens_to_deduct):
                    self.logger.info(f"Deducted {tokens_to_deduct} tokens for user {username}")
                    self.logger.info(f"Conversation tokens: {token_info['incoming_tokens']}, Response tokens: {response_tokens}")
                else:
                    self.logger.warning(f"Failed to deduct {tokens_to_deduct} tokens for user {username}")
            except Exception as e:
                self.logger.error(f"Error deducting tokens for user {username}: {str(e)}")
                # We don't re-raise the exception here to avoid disrupting the response
                # But you might want to handle this differently based on your requirements

        return body
