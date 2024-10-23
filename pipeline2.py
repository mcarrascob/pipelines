from typing import List, Optional
from schemas import OpenAIChatMessage
import os
import uuid
import tiktoken
import json

from utils.pipelines.main import get_last_user_message, get_last_assistant_message
from pydantic import BaseModel
from langfuse import Langfuse
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError


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
        self.tokenizers = {}

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        self.set_langfuse()

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        self.langfuse.flush()

    async def on_valves_updated(self):
        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=False,
            )
            self.langfuse.auth_check()
        except UnauthorizedError:
            print(
                "Langfuse credentials incorrect. Please re-enter your Langfuse credentials in the pipeline settings."
            )
        except Exception as e:
            print(f"Langfuse error: {e} Please re-enter your Langfuse credentials in the pipeline settings.")

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

        trace = self.langfuse.trace(
            name=f"filter:{__name__}",
            input=body,
            user_id=user["id"] if user else None,
            metadata={"name": user["name"] if user else None},
            session_id=body["chat_id"],
        )

        # Calculate input tokens for all messages
        input_tokens = self.count_tokens(body["messages"], body["model"])

        generation = trace.generation(
            name=body["chat_id"],
            model=body["model"],
            input=body["messages"],
            metadata={"interface": "open-webui", "input_tokens": input_tokens},
        )

        self.chat_generations[body["chat_id"]] = {"generation": generation, "input_tokens": input_tokens, "model": body["model"]}
        print(f"Calculated input tokens: {input_tokens}")
        print(trace.get_trace_url())

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        if body["chat_id"] not in self.chat_generations:
            return body

        generation_data = self.chat_generations[body["chat_id"]]
        generation = generation_data["generation"]
        fallback_input_tokens = generation_data["input_tokens"]
        model = generation_data["model"]

        generated_message = get_last_assistant_message(body["messages"])

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

        # Calculate costs for GPT-4o-mini specifically
        usage_data = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        if model.lower() in ["gpt-4o-mini", "gpt-4o-mini"]:
            # Calculate costs based on actual rates
            input_cost = (prompt_tokens * 0.00000015)  # $0.15 per 1M tokens
            output_cost = (completion_tokens * 0.0000006)  # $0.60 per 1M tokens
            total_cost = input_cost + output_cost
            
            # Add cost data to usage
            usage_data.update({
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost
            })

        generation.end(
            output=generated_message,
            usage=usage_data,
            metadata={"interface": "open-webui"},
        )

        # Print token information for verification
        print(f"Model: {model}")
        print(f"API prompt tokens: {api_usage.get('prompt_tokens', 'Not provided')}")
        print(f"API completion tokens: {api_usage.get('completion_tokens', 'Not provided')}")
        print(f"API total tokens: {api_usage.get('total_tokens', 'Not provided')}")
        print(f"Reported prompt tokens: {prompt_tokens}")
        print(f"Reported completion tokens: {completion_tokens}")
        print(f"Reported total tokens: {total_tokens}")
        
        if model.lower() in ["gpt-4o-mini", "gpt-4o-mini"]:
            print(f"Input cost: ${usage_data['input_cost']:.6f}")
            print(f"Output cost: ${usage_data['output_cost']:.6f}")
            print(f"Total cost: ${usage_data['total_cost']:.6f}")

        return body
