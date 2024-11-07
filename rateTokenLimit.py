import os
from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import time
import tiktoken

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        requests_per_minute: Optional[int] = None
        requests_per_hour: Optional[int] = None
        sliding_window_limit: Optional[int] = None
        sliding_window_minutes: Optional[int] = None
        max_input_tokens: Optional[int] = None

    class SpanishErrors:
        RATE_LIMIT_MINUTE = "Límite de solicitudes por minuto excedido. Por favor, espere un momento antes de intentar nuevamente."
        RATE_LIMIT_HOUR = "Límite de solicitudes por hora excedido. Por favor, inténtelo más tarde."
        RATE_LIMIT_WINDOW = "Límite de solicitudes en ventana de tiempo excedido. Por favor, espere unos minutos."
        TOKEN_LIMIT = "El mensaje excede el límite de tokens permitido ({} tokens). El límite máximo es de {} tokens."

    def __init__(self):
        self.type = "filter"
        self.name = "Rate & Token Limit Filter"
        
        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("RATE_LIMIT_PIPELINES", "*").split(","),
                "requests_per_minute": int(
                    os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", 10)
                ),
                "requests_per_hour": int(
                    os.getenv("RATE_LIMIT_REQUESTS_PER_HOUR", 1000)
                ),
                "sliding_window_limit": int(
                    os.getenv("RATE_LIMIT_SLIDING_WINDOW_LIMIT", 100)
                ),
                "sliding_window_minutes": int(
                    os.getenv("RATE_LIMIT_SLIDING_WINDOW_MINUTES", 15)
                ),
                "max_input_tokens": int(
                    os.getenv("MAX_INPUT_TOKENS", 10000)
                ),
            }
        )

        self.user_requests = {}
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Default OpenAI encoding

    def count_tokens(self, messages: List[OpenAIChatMessage]) -> int:
        """Count the number of tokens in the input messages."""
        total_tokens = 0
        for message in messages:
            # Count tokens in the content
            if message.content:
                total_tokens += len(self.tokenizer.encode(message.content))
            # Count tokens in the role (system, user, assistant)
            if message.role:
                total_tokens += len(self.tokenizer.encode(message.role))
        return total_tokens

    def check_token_limit(self, messages: List[OpenAIChatMessage]) -> tuple[bool, int]:
        """Check if the input messages exceed the token limit."""
        token_count = self.count_tokens(messages)
        return token_count > self.valves.max_input_tokens, token_count

    def get_rate_limit_error(self, limit_type: str) -> str:
        """Get the appropriate Spanish error message for the rate limit type."""
        if limit_type == "minute":
            return self.SpanishErrors.RATE_LIMIT_MINUTE
        elif limit_type == "hour":
            return self.SpanishErrors.RATE_LIMIT_HOUR
        else:
            return self.SpanishErrors.RATE_LIMIT_WINDOW

    def rate_limited(self, user_id: str) -> tuple[bool, Optional[str]]:
        """Check if a user is rate limited and return the type of limit if exceeded."""
        self.prune_requests(user_id)

        user_reqs = self.user_requests.get(user_id, [])

        if self.valves.requests_per_minute is not None:
            requests_last_minute = sum(1 for req in user_reqs if time.time() - req < 60)
            if requests_last_minute >= self.valves.requests_per_minute:
                return True, "minute"

        if self.valves.requests_per_hour is not None:
            requests_last_hour = sum(1 for req in user_reqs if time.time() - req < 3600)
            if requests_last_hour >= self.valves.requests_per_hour:
                return True, "hour"

        if self.valves.sliding_window_limit is not None:
            requests_in_window = len(user_reqs)
            if requests_in_window >= self.valves.sliding_window_limit:
                return True, "window"

        return False, None

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")
        print(body)
        print(user)

        if user.get("role", "admin") == "user":
            user_id = user["id"] if user and "id" in user else "default_user"
            
            # Check rate limits
            is_limited, limit_type = self.rate_limited(user_id)
            if is_limited:
                error_message = self.get_rate_limit_error(limit_type)
                raise Exception(error_message)

            # Check token limits
            if "messages" in body:
                exceeds_limit, token_count = self.check_token_limit(body["messages"])
                if exceeds_limit:
                    error_message = self.SpanishErrors.TOKEN_LIMIT.format(
                        token_count, 
                        self.valves.max_input_tokens
                    )
                    raise Exception(error_message)

            self.log_request(user_id)

        return body

    # ... (keeping the existing prune_requests and log_request methods unchanged)
    def prune_requests(self, user_id: str):
        """Prune old requests that are outside of the sliding window period."""
        now = time.time()
        if user_id in self.user_requests:
            self.user_requests[user_id] = [
                req
                for req in self.user_requests[user_id]
                if (
                    (self.valves.requests_per_minute is not None and now - req < 60)
                    or (self.valves.requests_per_hour is not None and now - req < 3600)
                    or (
                        self.valves.sliding_window_limit is not None
                        and now - req < self.valves.sliding_window_minutes * 60
                    )
                )
            ]

    def log_request(self, user_id: str):
        """Log a new request for a user."""
        now = time.time()
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        self.user_requests[user_id].append(now)
