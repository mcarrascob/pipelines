import os
from typing import List, Optional, Dict
from pydantic import BaseModel
import tiktoken
import aiohttp
import logging
import json
import uuid

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
        self.chat_generations = {}

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def get_tokenizer(self, model: str):
        if model not in self.tokenizers:
            try:
                self.tokenizers[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                print(f"Warning: Model {model} not found. Using default tokenizer.")
                self.tokenizers[model] = tiktoken.get_encoding("cl100k_base")
        return self.tokenizers[model]

    def count_tokens(self, messages: List[dict], model: str) -> int:
        """Count tokens for a list of messages."""
        tokenizer = self.get_tokenizer(model)
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

        if user and user.get("role") in self.valves.target_user_roles:
            username = user.get("name", "default_user")
            
            try:
                user_tokens = await self.get_user_info(username)
                
                if user_tokens <= 0:
                    raise Exception("Sorry, you don't have any tokens left. Please purchase more tokens to continue using our service.")
                
                # Calculate tokens for the entire conversation context
                input_tokens = self.count_tokens(body["messages"], body["model"])
                
                if input_tokens > user_tokens:
                    raise Exception(f"Your conversation requires {input_tokens} tokens, but you only have {user_tokens} available. Please shorten your message or purchase more tokens.")

                self.logger.info(f"User {username} conversation requires {input_tokens} tokens. Current balance: {user_tokens}")
                
                # Store token information
                self.chat_generations[body["chat_id"]] = {
                    "input_tokens": input_tokens,
                    "model": body["model"],
                    "username": username,
                    "user_tokens": user_tokens
                }
            
            except Exception as e:
                self.logger.error(f"Error processing request for user {username}: {str(e)}")
                raise

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        if body["chat_id"] not in self.chat_generations:
            return body

        generation_data = self.chat_generations[body["chat_id"]]
        fallback_input_tokens = generation_data["input_tokens"]
        model = generation_data["model"]
        username = generation_data["username"]

        # Get the last assistant message
        generated_message = body["messages"][-1]["content"] if body["messages"] else ""

        # Try to get token counts from API response
        api_usage = body.get("usage", {})
        prompt_tokens = api_usage.get("prompt_tokens")
        completion_tokens = api_usage.get("completion_tokens")
        total_tokens = api_usage.get("total_tokens")

        # If API doesn't provide token counts, use our own counting method
        if prompt_tokens is None:
            prompt_tokens = fallback_input_tokens
            print("Using fallback method for prompt tokens")
        if completion_tokens is None:
            completion_tokens = self.count_output_tokens(generated_message, model)
            print("Using fallback method for completion tokens")
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
            print("Using fallback method for total tokens")

        # Deduct tokens
        tokens_to_deduct = total_tokens
        try:
            if await self.use_tokens(username, model, tokens_to_deduct):
                self.logger.info(f"Deducted {tokens_to_deduct} tokens for user {username} on model {model}")
            else:
                self.logger.warning(f"Failed to deduct {tokens_to_deduct} tokens for user {username} on model {model}")
        except Exception as e:
            self.logger.error(f"Error deducting tokens for user {username}: {str(e)}")
            # We don't re-raise the exception here to avoid disrupting the response
            # But you might want to handle this differently based on your requirements

        # Add token usage information to the response
        body['token_usage'] = {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens
        }

        # Print token information for verification
        print(f"Model: {model}")
        print(f"API prompt tokens: {api_usage.get('prompt_tokens', 'Not provided')}")
        print(f"API completion tokens: {api_usage.get('completion_tokens', 'Not provided')}")
        print(f"API total tokens: {api_usage.get('total_tokens', 'Not provided')}")
        print(f"Reported prompt tokens: {prompt_tokens}")
        print(f"Reported completion tokens: {completion_tokens}")
        print(f"Reported total tokens: {total_tokens}")

        # Clean up
        del self.chat_generations[body["chat_id"]]

        return body
