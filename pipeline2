from typing import List, Optional, Dict
from schemas import OpenAIChatMessage
import os
import uuid
import tiktoken
import json

from utils.pipelines.main import get_last_user_message, get_last_assistant_message
from pydantic import BaseModel
from langfuse import Langfuse
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError

class ModelPricing:
    def __init__(self):
        self.pricing = {
            "gpt-4o-mini": {
                "input": 0.00015,  # $0.15 per 1M tokens
                "output": 0.0006,  # $0.60 per 1M tokens
            }
        }

    def get_model_cost(self, model: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        model = model.lower()
        if model not in self.pricing:
            print(f"Warning: No pricing found for model {model}. Using default pricing.")
            return {
                "input_cost": 0,
                "output_cost": 0,
                "total_cost": 0
            }
        
        model_pricing = self.pricing[model]
        input_cost = (input_tokens * model_pricing["input"])
        output_cost = (output_tokens * model_pricing["output"])
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }

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
        self.pricing = ModelPricing()

    # ... [Previous methods remain the same until outlet] ...

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
        prompt_tokens = api_usage.get("prompt_tokens", fallback_input_tokens)
        completion_tokens = api_usage.get("completion_tokens", 
            self.count_output_tokens(generated_message, model))
        total_tokens = prompt_tokens + completion_tokens

        # Calculate costs
        costs = self.pricing.get_model_cost(model, prompt_tokens, completion_tokens)

        generation.end(
            output=generated_message,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "input_cost": costs["input_cost"],
                "output_cost": costs["output_cost"],
                "total_cost": costs["total_cost"]
            },
            metadata={
                "interface": "open-webui",
                "cost_details": costs
            },
        )

        # Print token and cost information for verification
        print(f"Model: {model}")
        print(f"Prompt tokens: {prompt_tokens} (Cost: ${costs['input_cost']:.6f})")
        print(f"Completion tokens: {completion_tokens} (Cost: ${costs['output_cost']:.6f})")
        print(f"Total tokens: {total_tokens} (Total Cost: ${costs['total_cost']:.6f})")

        return body
