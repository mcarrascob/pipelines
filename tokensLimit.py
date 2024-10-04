import os
from typing import List, Optional, Dict
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import tiktoken
import json
import uuid

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        max_tokens_per_user: Optional[int] = None

    def __init__(self):
        self.type = "filter"
        self.name = "Advanced Token Limit Filter"

        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("TOKEN_LIMIT_PIPELINES", "*").split(","),
                "max_tokens_per_user": int(
                    os.getenv("MAX_TOKENS_PER_USER", 1000000)  # Default to 1 million tokens
                ),
            }
        )

        self.user_tokens: Dict[str, int] = {}  # Dictionary to track user tokens
        self.user_conversations: Dict[str, List[Dict]] = {}  # Dictionary to track user conversations
        self.tokenizers: Dict[str, tiktoken.Encoding] = {}  # Dictionary to store tokenizers
        self.chat_generations: Dict[str, Dict] = {}  # Dictionary to track chat generations

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

    def token_limited(self, user_id: str, new_tokens: int) -> bool:
        current_tokens = self.user_tokens.get(user_id, 0)
        return current_tokens + new_tokens > self.valves.max_tokens_per_user

    def update_conversation(self, user_id: str, new_message: Dict, is_user: bool = True):
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = []
        
        self.user_conversations[user_id].append({
            'role': 'user' if is_user else 'assistant',
            'content': new_message.get('content', '')
        })

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

        if user and user.get("role", "admin") == "user":
            user_id = user.get("id", "default_user")
            
            # Calculate input tokens for all messages
            input_tokens = self.count_tokens(body["messages"], body["model"])
            
            # Check token limit
            if self.token_limited(user_id, input_tokens):
                raise Exception("Token limit exceeded. Please start a new conversation.")

            # Update conversation and token count
            self.update_conversation(user_id, body["messages"][-1])
            self.user_tokens[user_id] = self.user_tokens.get(user_id, 0) + input_tokens

            # Store information for outlet
            self.chat_generations[body["chat_id"]] = {
                "input_tokens": input_tokens,
                "model": body["model"],
                "user_id": user_id
            }

        print(f"Calculated input tokens: {input_tokens}")
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        if body["chat_id"] not in self.chat_generations:
            return body

        generation_data = self.chat_generations[body["chat_id"]]
        fallback_input_tokens = generation_data["input_tokens"]
        model = generation_data["model"]
        user_id = generation_data["user_id"]

        generated_message = body["messages"][-1] if body["messages"] else {}

        # Try to get token counts from API response
        api_usage = body.get("usage", {})
        prompt_tokens = api_usage.get("prompt_tokens", fallback_input_tokens)
        completion_tokens = api_usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = self.count_output_tokens(generated_message.get("content", ""), model)
        total_tokens = api_usage.get("total_tokens", prompt_tokens + completion_tokens)

        # Update user's token count
        self.user_tokens[user_id] = self.user_tokens.get(user_id, 0) + total_tokens

        # Update conversation with model's response
        self.update_conversation(user_id, generated_message, is_user=False)

        # Print token information for verification
        print(f"Model: {model}")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"User {user_id} total tokens: {self.user_tokens[user_id]}")

        return body
