import os
from typing import List, Optional, Dict
from pydantic import BaseModel
import tiktoken
import time
import logging
import uuid

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        requests_per_minute: Optional[int] = None
        requests_per_hour: Optional[int] = None
        sliding_window_limit: Optional[int] = None
        sliding_window_minutes: Optional[int] = None
        max_input_tokens: Optional[int] = None
        target_user_roles: List[str] = ["user"]

    class SpanishErrors:
        RATE_LIMIT_MINUTE = "Se ha excedido el límite de solicitudes por minuto. Por favor, espere un momento antes de intentar nuevamente."
        RATE_LIMIT_HOUR = "Se ha excedido el límite de solicitudes por hora. Por favor, inténtelo más tarde."
        RATE_LIMIT_WINDOW = "Se ha excedido el límite de solicitudes en ventana de tiempo. Por favor, espere unos minutos."
        TOKEN_LIMIT = "La conversación alcanzaría aproximadamente {} tokens con este mensaje, superando el límite de {} tokens.\nPor favor:\n- Inicie una nueva conversación\n- O elimine algunos mensajes anteriores\n- O divida su mensaje en partes más pequeñas"
        MISSING_KEYS = "Error: Faltan campos requeridos en la solicitud: {}"
        MODEL_ERROR = "Error: El modelo especificado no es válido o no está soportado."

    def __init__(self):
        self.type = "filter"
        self.name = "Rate & Token Limit Filter"
        
        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("RATE_LIMIT_PIPELINES", "*").split(","),
                "requests_per_minute": int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", 10)),
                "requests_per_hour": int(os.getenv("RATE_LIMIT_REQUESTS_PER_HOUR", 1000)),
                "sliding_window_limit": int(os.getenv("RATE_LIMIT_SLIDING_WINDOW_LIMIT", 100)),
                "sliding_window_minutes": int(os.getenv("RATE_LIMIT_SLIDING_WINDOW_MINUTES", 15)),
                "max_input_tokens": int(os.getenv("MAX_INPUT_TOKENS", 10000)),
            }
        )

        self.user_requests = {}
        self.tokenizers = {}
        self.logger = logging.getLogger(__name__)
        # Dictionary to store conversation histories and their token counts
        self.conversation_tokens = {}

    def get_tokenizer(self, model: str):
        """Get the appropriate tokenizer for the model."""
        if model not in self.tokenizers:
            try:
                self.tokenizers[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                self.logger.warning(f"Model {model} not found. Using default tokenizer.")
                self.tokenizers[model] = tiktoken.get_encoding("cl100k_base")
        return self.tokenizers[model]

    def estimate_response_tokens(self, model: str) -> int:
        """Estimate the number of tokens the model might use in its response."""
        # Conservative estimates based on common model behavior
        model_estimates = {
            "gpt-4": 1000,           # Estimate for GPT-4
            "gpt-3.5-turbo": 500,    # Estimate for GPT-3.5
            "default": 800           # Default estimate
        }
        return model_estimates.get(model, model_estimates["default"])

    def count_tokens(self, messages: List[dict], model: str) -> int:
        """Count tokens for a list of messages."""
        tokenizer = self.get_tokenizer(model)
        token_count = 0
        
        for message in messages:
            token_count += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                if value:  # Only count if value is not None or empty
                    token_count += len(tokenizer.encode(str(value)))
            
            if "name" in message:  # If there's a name, the role is omitted
                token_count -= 1  # Role is always required and always 1 token

        token_count += 2  # Every reply is primed with <im_start>assistant
        
        # Adjust for the chain of messages
        if len(messages) > 1:
            token_count -= 2 * (len(messages) - 1)  # Subtract 2 for each message after the first

        return token_count

    def get_conversation_token_count(self, messages: List[dict], model: str) -> tuple[int, int]:
        """
        Get the current token count and estimated total after response
        Returns: (current_tokens, estimated_total_tokens)
        """
        current_tokens = self.count_tokens(messages, model)
        estimated_response_tokens = self.estimate_response_tokens(model)
        estimated_total = current_tokens + estimated_response_tokens
        return current_tokens, estimated_total

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

    def get_rate_limit_error(self, limit_type: str) -> str:
        """Get the appropriate Spanish error message for the rate limit type."""
        if limit_type == "minute":
            return self.SpanishErrors.RATE_LIMIT_MINUTE
        elif limit_type == "hour":
            return self.SpanishErrors.RATE_LIMIT_HOUR
        else:
            return self.SpanishErrors.RATE_LIMIT_WINDOW

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")
        print(f"Received body: {body}")
        print(f"User: {user}")

        # Ensure chat_id exists
        if "chat_id" not in body:
            unique_id = f"SYSTEM MESSAGE {uuid.uuid4()}"
            body["chat_id"] = unique_id
            print(f"chat_id was missing, set to: {unique_id}")

        # Verify required fields
        required_keys = ["model", "messages"]
        missing_keys = [key for key in required_keys if key not in body]
        
        if missing_keys:
            error_message = self.SpanishErrors.MISSING_KEYS.format(", ".join(missing_keys))
            self.logger.error(error_message)
            raise ValueError(error_message)

        if user and user.get("role") in self.valves.target_user_roles:
            user_id = user["id"] if user and "id" in user else "default_user"
            
            # Check rate limits
            is_limited, limit_type = self.rate_limited(user_id)
            if is_limited:
                error_message = self.get_rate_limit_error(limit_type)
                raise Exception(error_message)

            try:
                # Calculate current tokens and estimate total after response
                current_tokens, estimated_total = self.get_conversation_token_count(
                    body["messages"], 
                    body["model"]
                )
                
                # Log token counts for debugging
                self.logger.info(f"Current tokens: {current_tokens}")
                self.logger.info(f"Estimated total after response: {estimated_total}")
                
                # Check if estimated total would exceed limit
                if estimated_total > self.valves.max_input_tokens:
                    error_message = self.SpanishErrors.TOKEN_LIMIT.format(
                        estimated_total,
                        self.valves.max_input_tokens
                    )
                    raise Exception(error_message)

                # Store conversation token count
                self.conversation_tokens[body["chat_id"]] = {
                    "current_tokens": current_tokens,
                    "model": body["model"]
                }

                self.logger.info(f"User {user_id} conversation current tokens: {current_tokens}")
                
            except Exception as e:
                self.logger.error(f"Error processing request for user {user_id}: {str(e)}")
                raise

            self.log_request(user_id)

        return body

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
