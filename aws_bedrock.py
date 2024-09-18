import json
import logging
from typing import List, Union, Generator, Iterator

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from pydantic import BaseModel

import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        AWS_REGION_NAME: str = ""

    def __init__(self):
        self.type = "manifold"
        self.name = "AWS Bedrock: "

        self.valves = self.Valves(
            AWS_REGION_NAME=os.getenv("AWS_REGION_NAME", "us-east-1"),
        )

        self.bedrock = None
        self.bedrock_runtime = None
        self.pipelines = []

    async def on_startup(self):
        logger.info(f"on_startup:{__name__}")
        self.initialize_clients()

    async def on_shutdown(self):
        logger.info(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        logger.info(f"on_valves_updated:{__name__}")
        self.initialize_clients()

    def initialize_clients(self):
        try:
            # Create a boto3 session using the default credential provider chain
            session = boto3.Session(region_name=self.valves.AWS_REGION_NAME)

            self.bedrock = session.client('bedrock')
            self.bedrock_runtime = session.client('bedrock-runtime')
            
            # Test the connection by trying to list models
            self.pipelines = self.get_models()
            logger.info("Successfully initialized AWS clients using IAM role.")
        except NoCredentialsError:
            logger.error("No AWS credentials found. Please ensure the IAM role is correctly configured.")
            self.pipelines = [{"id": "error", "name": "No AWS credentials found. Please check IAM role configuration."}]
        except ClientError as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            self.pipelines = [{"id": "error", "name": f"Failed to initialize AWS clients: {e}"}]

    def get_models(self):
        if not self.bedrock:
            return [{"id": "error", "name": "AWS Bedrock client not initialized."}]
        try:
            response = self.bedrock.list_foundation_models(byProvider='Anthropic', byInferenceType='ON_DEMAND')
            return [
                {
                    "id": model["modelId"],
                    "name": model["modelName"],
                }
                for model in response["modelSummaries"]
            ]
        except ClientError as e:
            logger.error(f"Error fetching models: {e}")
            return [
                {
                    "id": "error",
                    "name": f"Could not fetch models from Bedrock: {e}",
                },
            ]

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logger.info(f"pipe:{__name__}")

        if not self.bedrock_runtime:
            return "Error: AWS Bedrock client not initialized. Please check IAM role configuration."

        try:
            processed_messages = self.process_messages(messages)
            
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": body.get("max_tokens", 4096),
                "temperature": body.get("temperature", 0.7),
                "top_k": body.get("top_k", 250),
                "top_p": body.get("top_p", 1),
                "stop_sequences": body.get("stop", []),
                "stream": body.get("stream", False),
                "messages": processed_messages,
            }

            if body.get("stream", False):
                return self.stream_response(model_id, payload)
            else:
                return self.get_completion(model_id, payload)
        except Exception as e:
            logger.error(f"Error in pipe: {e}")
            return f"Error: {e}"

    def process_messages(self, messages: List[dict]) -> List[dict]:
        processed_messages = []
        for message in messages:
            if message["role"] == "system":
                continue  # Skip system messages as they're handled separately in Claude v2
            processed_messages.append({
                "role": message["role"],
                "content": message.get("content", "")
            })
        return processed_messages

    def stream_response(self, model_id: str, payload: dict) -> Generator:
        try:
            response = self.bedrock_runtime.invoke_model_with_response_stream(
                modelId=model_id,
                body=json.dumps(payload)
            )
            for event in response['body']:
                chunk = json.loads(event['chunk']['bytes'].decode())
                if chunk['type'] == 'content_block_delta':
                    yield chunk['delta']['text']
                elif chunk['type'] == 'message_delta':
                    if 'stop_reason' in chunk['delta']:
                        break
        except ClientError as e:
            logger.error(f"Error in stream_response: {e}")
            yield f"Error: {e}"

    def get_completion(self, model_id: str, payload: dict) -> str:
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(payload)
            )
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        except ClientError as e:
            logger.error(f"Error in get_completion: {e}")
            return f"Error: {e}"
