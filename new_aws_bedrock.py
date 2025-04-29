"""
title: AWS Bedrock Claude Pipeline
author: Modified from G-mario's version
date: 2024-04-29
version: 1.1
license: MIT
description: A pipeline for generating text and processing images using the AWS Bedrock API (By Anthropic claude) with IAM role-based authentication.
requirements: requests, boto3
environment_variables: AWS_REGION_NAME
"""
import base64
import json
import logging
from io import BytesIO
from typing import List, Union, Generator, Iterator, Optional, Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from pydantic import BaseModel

import os
import requests

from utils.pipelines.main import pop_system_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REASONING_EFFORT_BUDGET_TOKEN_MAP = {
    "none": None,
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    "max": 32768,
}

# Maximum combined token limit for Claude 3.7
MAX_COMBINED_TOKENS = 64000


class Pipeline:
    class Valves(BaseModel):
        AWS_REGION_NAME: str = ""

    def __init__(self):
        self.type = "manifold"
        self.name = "Bedrock: "

        self.valves = self.Valves(
            AWS_REGION_NAME=os.getenv(
                "AWS_REGION_NAME", 
                os.getenv("AWS_REGION", 
                         os.getenv("AWS_DEFAULT_REGION", "us-east-1")
                )
            )
        )

        self.bedrock = None
        self.bedrock_runtime = None
        self.pipelines = []
        
        self.update_pipelines()

    async def on_startup(self):
        logger.info(f"on_startup:{__name__}")
        self.update_pipelines()

    async def on_shutdown(self):
        logger.info(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        logger.info(f"on_valves_updated:{__name__}")
        self.update_pipelines()

    def update_pipelines(self) -> None:
        try:
            session = boto3.Session(region_name=self.valves.AWS_REGION_NAME)
            self.bedrock = session.client('bedrock')
            self.bedrock_runtime = session.client('bedrock-runtime')
            self.pipelines = self.get_models()
            logger.info("Successfully initialized AWS clients using IAM role.")
        except NoCredentialsError:
            logger.error("No AWS credentials found. Please ensure the IAM role is correctly configured.")
            self.pipelines = [{"id": "error", "name": "No AWS credentials found. Please check IAM role configuration."}]
        except ClientError as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            self.pipelines = [{"id": "error", "name": f"Failed to initialize AWS clients: {e}"}]
        except Exception as e:
            logger.error(f"Error: {e}")
            self.pipelines = [
                {
                    "id": "error",
                    "name": "Could not fetch models from Bedrock, please set up AWS Instance/Task Role.",
                },
            ]

    def get_models(self):
        try:
            res = []
            response = self.bedrock.list_foundation_models(byProvider='Anthropic')
            for model in response['modelSummaries']:
                inference_types = model.get('inferenceTypesSupported', [])
                if "ON_DEMAND" in inference_types:
                    res.append({'id': model['modelId'], 'name': model['modelName']})
                elif "INFERENCE_PROFILE" in inference_types:
                    inferenceProfileId = self.getInferenceProfileId(model['modelArn'])
                    if inferenceProfileId:
                        res.append({'id': inferenceProfileId, 'name': model['modelName']})

            return res
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return [
                {
                    "id": "error",
                    "name": f"Could not fetch models from Bedrock: {e}",
                },
            ]

    def getInferenceProfileId(self, modelArn: str) -> str:
        response = self.bedrock.list_inference_profiles()
        for profile in response.get('inferenceProfileSummaries', []):
            for model in profile.get('models', []):
                if model.get('modelArn') == modelArn:
                    return profile['inferenceProfileId']
        return None

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logger.info(f"pipe:{__name__}")

        if not self.bedrock_runtime:
            return "Error: AWS Bedrock client not initialized. Please check IAM role configuration."

        system_message, messages = pop_system_message(messages)

        try:
            processed_messages = []
            image_count = 0
            for message in messages:
                processed_content = []
                if isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if item["type"] == "text":
                            processed_content.append({"text": item["text"]})
                        elif item["type"] == "image_url":
                            if image_count >= 20:
                                raise ValueError("Maximum of 20 images per API call exceeded")
                            processed_image = self.process_image(item["image_url"])
                            processed_content.append(processed_image)
                            image_count += 1
                else:
                    processed_content = [{"text": message.get("content", "")}]

                processed_messages.append({"role": message["role"], "content": processed_content})

            payload = {"modelId": model_id,
                       "messages": processed_messages,
                       "system": [{'text': system_message["content"] if system_message else 'You are an intelligent AI assistant'}],
                       "inferenceConfig": {
                           "temperature": body.get("temperature", 0.5),
                           "topP": body.get("top_p", 0.9),
                           "maxTokens": body.get("max_tokens", 4096),
                           "stopSequences": body.get("stop", []),
                        },
                        "additionalModelRequestFields": {"top_k": body.get("top_k", 200)}
                       }

            if body.get("stream", False):
                supports_thinking = "claude-3-7" in model_id
                reasoning_effort = body.get("reasoning_effort", "none")
                budget_tokens = REASONING_EFFORT_BUDGET_TOKEN_MAP.get(reasoning_effort)

                # Allow users to input an integer value representing budget tokens
                if (
                    not budget_tokens
                    and reasoning_effort not in REASONING_EFFORT_BUDGET_TOKEN_MAP.keys()
                ):
                    try:
                        budget_tokens = int(reasoning_effort)
                    except ValueError as e:
                        logger.error("Failed to convert reasoning effort to int", e)
                        budget_tokens = None

                if supports_thinking and budget_tokens:
                    # Check if the combined tokens (budget_tokens + max_tokens) exceeds the limit
                    max_tokens = payload["inferenceConfig"].get("maxTokens", 4096)
                    combined_tokens = budget_tokens + max_tokens

                    if combined_tokens > MAX_COMBINED_TOKENS:
                        error_message = f"Error: Combined tokens (budget_tokens {budget_tokens} + max_tokens {max_tokens} = {combined_tokens}) exceeds the maximum limit of {MAX_COMBINED_TOKENS}"
                        logger.error(error_message)
                        return error_message

                    payload["inferenceConfig"]["maxTokens"] = combined_tokens
                    payload["additionalModelRequestFields"]["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget_tokens,
                    }
                    # Thinking requires temperature 1.0 and does not support top_p, top_k
                    payload["inferenceConfig"]["temperature"] = 1.0
                    if "top_k" in payload["additionalModelRequestFields"]:
                        del payload["additionalModelRequestFields"]["top_k"]
                    if "topP" in payload["inferenceConfig"]:
                        del payload["inferenceConfig"]["topP"]
                return self.stream_response(model_id, payload)
            else:
                return self.get_completion(model_id, payload)
        except Exception as e:
            logger.error(f"Error in pipe: {e}")
            return f"Error: {e}"

    def process_image(self, image: str):
        img_stream = None
        content_type = None

        if image["url"].startswith("data:image"):
            mime_type, base64_string = image["url"].split(",", 1)
            content_type = mime_type.split(":")[1].split(";")[0]
            image_data = base64.b64decode(base64_string)
            img_stream = BytesIO(image_data)
        else:
            response = requests.get(image["url"])
            img_stream = BytesIO(response.content)
            content_type = response.headers.get('Content-Type', 'image/jpeg')

        media_type = content_type.split('/')[-1] if '/' in content_type else content_type
        return {
            "image": {
                "format": media_type,
                "source": {"bytes": img_stream.read()}
            }
        }

    def stream_response(self, model_id: str, payload: dict) -> Generator:
        try:
            streaming_response = self.bedrock_runtime.converse_stream(**payload)

            in_reasoning_context = False
            for chunk in streaming_response["stream"]:
                if in_reasoning_context and "contentBlockStop" in chunk:
                    in_reasoning_context = False
                    yield "\n </think> \n\n"
                elif "contentBlockDelta" in chunk and "delta" in chunk["contentBlockDelta"]:
                    if "reasoningContent" in chunk["contentBlockDelta"]["delta"]:
                        if not in_reasoning_context:
                            yield "<think>"

                        in_reasoning_context = True
                        if "text" in chunk["contentBlockDelta"]["delta"]["reasoningContent"]:
                            yield chunk["contentBlockDelta"]["delta"]["reasoningContent"]["text"]
                    elif "text" in chunk["contentBlockDelta"]["delta"]:
                        yield chunk["contentBlockDelta"]["delta"]["text"]
        except ClientError as e:
            logger.error(f"Error in stream_response: {e}")
            yield f"Error: {e}"

    def get_completion(self, model_id: str, payload: dict) -> str:
        try:
            response = self.bedrock_runtime.converse(**payload)
            return response['output']['message']['content'][0]['text']
        except ClientError as e:
            logger.error(f"Error in get_completion: {e}")
            return f"Error: {e}"
