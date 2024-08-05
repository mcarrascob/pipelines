from typing import List, Optional
import os
import uuid
import json
import logging

from utils.pipelines.main import get_last_assistant_message
from pydantic import BaseModel
from langfuse import Langfuse
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        secret_key: str
        public_key: str
        host: str

    def __init__(self):
        self.type = "filter"
        self.name = "Langfuse Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "your-secret-key-here"),
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "your-public-key-here"),
                "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            }
        )
        self.langfuse = None
        self.chat_generations = {}

    async def on_startup(self):
        logger.info(f"on_startup:{__name__}")
        self.set_langfuse()

    async def on_shutdown(self):
        logger.info(f"on_shutdown:{__name__}")
        self.langfuse.flush()

    async def on_valves_updated(self):
        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=True,
            )
            self.langfuse.auth_check()
            logger.info("Langfuse initialized successfully")
        except UnauthorizedError:
            logger.error("Langfuse credentials incorrect. Please re-enter your Langfuse credentials in the pipeline settings.")
        except Exception as e:
            logger.error(f"Langfuse error: {e} Please re-enter your Langfuse credentials in the pipeline settings.")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        logger.debug(f"inlet:{__name__}")
        logger.debug(f"Received body: {body}")
        logger.debug(f"User: {user}")

        if "chat_id" not in body:
            unique_id = f"SYSTEM MESSAGE {uuid.uuid4()}"
            body["chat_id"] = unique_id
            logger.info(f"chat_id was missing, set to: {unique_id}")

        required_keys = ["model", "messages"]
        missing_keys = [key for key in required_keys if key not in body]
        
        if missing_keys:
            error_message = f"Error: Missing keys in the request body: {', '.join(missing_keys)}"
            logger.error(error_message)
            raise ValueError(error_message)

        trace = self.langfuse.trace(
            name=f"filter:{__name__}",
            input=body,
            user_id=user["id"] if user else None,
            metadata={"name": user["name"] if user else None},
            session_id=body["chat_id"],
        )

        generation = trace.generation(
            name=body["chat_id"],
            model=body["model"],
            input=body["messages"],
            metadata={"interface": "open-webui"},
        )

        self.chat_generations[body["chat_id"]] = generation
        logger.info(f"Langfuse trace URL: {trace.get_trace_url()}")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        logger.debug(f"outlet:{__name__}")
        if body["chat_id"] not in self.chat_generations:
            logger.warning(f"No generation found for chat_id: {body['chat_id']}")
            return body

        generation = self.chat_generations[body["chat_id"]]

        # Parse the response if it's a string
        if isinstance(body.get("response"), str):
            try:
                response = json.loads(body["response"])
                logger.debug(f"Parsed response: {response}")
            except json.JSONDecodeError:
                logger.error("Error: Unable to parse response as JSON")
                response = {}
        else:
            response = body.get("response", {})

        # Extract the generated message
        generated_message = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not generated_message:
            generated_message = get_last_assistant_message(body.get("messages", []))
        
        logger.debug(f"Generated message: {generated_message}")

        # Extract token usage from the API response
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        logger.info(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

        try:
            generation.end(
                output=generated_message,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                metadata={"interface": "open-webui"},
            )
            logger.info("Successfully logged generation to Langfuse")
        except Exception as e:
            logger.error(f"Error logging to Langfuse: {e}")

        # Print the model's response for visibility
        logger.info(f"Model response: {generated_message}")

        return body
