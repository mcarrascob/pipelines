import os
from typing import List, Optional, Dict
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import tiktoken

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        max_tokens_per_user: Optional[int] = None

    def __init__(self):
        self.type = "filter"
        self.name = "Comprehensive Token Limit Filter"

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
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Initialize tokenizer

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def count_tokens(self, messages: List[Dict]) -> int:
        return sum(len(self.tokenizer.encode(msg.get('content', ''))) for msg in messages)

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

    def get_total_tokens(self, user_id: str, new_message: Dict) -> int:
        conversation = self.user_conversations.get(user_id, [])
        conversation_with_new = conversation + [{
            'role': 'user',
            'content': new_message.get('content', '')
        }]
        return self.count_tokens(conversation_with_new)

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")
        print(body)
        print(user)

        if user and user.get("role", "admin") == "user":
            user_id = user.get("id", "default_user")
            
            new_message = body.get("message", {})
            total_tokens = self.get_total_tokens(user_id, new_message)

            # Check token limit
            if self.token_limited(user_id, total_tokens):
                raise Exception("Token limit exceeded. Please start a new conversation.")

            # Update conversation and token count
            self.update_conversation(user_id, new_message)
            self.user_tokens[user_id] = total_tokens

        return body

    async def outlet(self, response: dict, user: Optional[dict] = None) -> dict:
        if user and user.get("role", "admin") == "user":
            user_id = user.get("id", "default_user")
            
            # Update conversation with model's response
            self.update_conversation(user_id, response, is_user=False)
            
            # Recalculate total tokens including model's response
            total_tokens = self.get_total_tokens(user_id, {})
            self.user_tokens[user_id] = total_tokens

        return response
