import os
from typing import List, Optional, Dict
from pydantic import BaseModel
import tiktoken
import anthropic
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
        self.tokenizers = {}
        self.anthropic_tokenizer = anthropic.Tokenizer()

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def get_tokenizer(self, model: str):
        if model in self.tokenizers:
            return self.tokenizers[model]

        # Anthropic models
        if "claude" in model.lower():
            return self.anthropic_tokenizer

        # OpenAI models
        if model.startswith("gpt-") or model in ["text-davinci-002", "text-davinci-003"]:
            try:
                tokenizer = tiktoken.encoding_for_model(model)
            except KeyError:
                tokenizer = tiktoken.get_encoding("cl100k_base")

        # Microsoft models (assuming they use GPT tokenization)
        elif any(provider in model.lower() for provider in ["microsoft", "azure"]):
            tokenizer = tiktoken.get_encoding("cl100k_base")

        # Default fallback
        else:
            print(f"Warning: Unknown model {model}. Using default tokenizer.")
            tokenizer = tiktoken.get_encoding("cl100k_base")

        self.tokenizers[model] = tokenizer
        return tokenizer

    def count_tokens(self, messages: List[dict], model: str) -> int:
        """Count tokens for a list of messages."""
        tokenizer = self.get_tokenizer(model)
        
        if "claude" in model.lower():
            # Anthropic-specific token counting
            total_tokens = 0
            for message in messages:
                if message['role'] == 'system':
                    total_tokens += self.anthropic_tokenizer.count(anthropic.HUMAN_PROMPT + message['content'] + anthropic.AI_PROMPT)
                elif message['role'] == 'user':
                    total_tokens += self.anthropic_tokenizer.count(anthropic.HUMAN_PROMPT + message['content'])
                elif message['role'] == 'assistant':
                    total_tokens += self.anthropic_tokenizer.count(anthropic.AI_PROMPT + message['content'])
            return total_tokens
        else:
            # Token counting for other models
            token_count = 0
            for message in messages:
                token_count += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    token_count += len(tokenizer.encode(str(value)))
                
                if "name" in message:  # If there's a name, the role is omitted
                    token_count -= 1  # Role is always required and always 1 token

            token_count += 2  # Every reply is primed with <im_start>assistant
            
            # Adjust for the chain of messages
            if len(messages) > 1:
                token_count -= 2 * (len(messages) - 1)  # Subtract 2 for each message after the first

            return token_count

    def count_output_tokens(self, message: str, model: str) -> int:
        """Count tokens for the output message, accounting for potential JSON structure."""
        tokenizer = self.get_tokenizer(model)
        
        if "claude" in model.lower():
            return self.anthropic_tokenizer.count(message)
        
        try:
            # Try to parse the message as JSON
            parsed_message = json.loads(message)
            # If successful, count tokens for the entire JSON structure
            return len(tokenizer.encode(json.dumps(parsed_message)))
        except json.JSONDecodeError:
            # If not JSON, count tokens for the raw string
            return len(tokenizer.encode(message))

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

    async def get_model_info(self, model: str) -> Optional[float]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/get_model_info?model={model}") as response:
                    if response.status == 404:
                        raise Exception(f"Model {model} not found in the system.")
                    elif response.status != 200:
                        raise Exception(f"Failed to get model info: {await response.text()}")
                    data = await response.json()
                    return data.get('cost_multiplier', 1.0)
        except aiohttp.ClientConnectorError:
            raise Exception(f"Cannot connect to API at {self.api_url}")


    async def use_tokens(self, username: str, model: str, tokens: int) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/use_tokens", json={"username": username, "model": model, "tokens": tokens}) as response:
                    if response.status == 404:
                        raise Exception(f"User {username} or model {model} not found in the system.")
                    elif response.status == 400:
                        raise Exception(f"Insufficient tokens for user {username}.")
                    elif response.status != 200:
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
            model = body.get('model')
            
            if not model:
                raise ValueError("Model not specified in the request body")

            try:
                user_tokens = await self.get_user_info(username)
                cost_multiplier = await self.get_model_info(model)
                
                if user_tokens <= 0:
                    raise Exception("Sorry, you don't have any tokens left. Please purchase more tokens to continue using our service.")
                
                # Calculate tokens for the entire conversation context
                incoming_tokens = self.count_tokens(body.get('messages', []), model)
                actual_tokens_to_use = int(incoming_tokens * cost_multiplier)
                
                if actual_tokens_to_use > user_tokens:
                    raise Exception(f"Your conversation requires {actual_tokens_to_use} tokens, but you only have {user_tokens} available. Please shorten your message or purchase more tokens.")

                self.logger.info(f"User {username} conversation requires {actual_tokens_to_use} tokens. Current balance: {user_tokens}")
            
            except Exception as e:
                self.logger.error(f"Error processing request for user {username}: {str(e)}")
                raise

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if user and user.get("role") in self.valves.target_user_roles:
            username = user.get("name", "default_user")
            model = body.get('model')

            if not model:
                self.logger.error("Model not specified in the response body")
                return body

            try:
                # Calculate tokens for the entire conversation, including the new response
                total_tokens = self.count_tokens(body.get('messages', []), model)
                
                # Calculate tokens for the new response separately
                response_tokens = self.count_output_tokens(body.get('content', ''), model)

                # Total tokens to deduct
                tokens_to_deduct = total_tokens

                if await self.use_tokens(username, model, tokens_to_deduct):
                    self.logger.info(f"Deducted {tokens_to_deduct} tokens for user {username} on model {model}")
                    self.logger.info(f"Model: {model}, Conversation tokens: {total_tokens}, Response tokens: {response_tokens}")
                else:
                    self.logger.warning(f"Failed to deduct {tokens_to_deduct} tokens for user {username} on model {model}")
            except Exception as e:
                self.logger.error(f"Error deducting tokens for user {username}: {str(e)}")
                # We don't re-raise the exception here to avoid disrupting the response
                # But you might want to handle this differently based on your requirements

        return body
