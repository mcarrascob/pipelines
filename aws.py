import base64
import json
import logging
from io import BytesIO
from typing import List, Union, Generator, Iterator, Optional

import boto3
from botocore.exceptions import ClientError

from pydantic import BaseModel

import os
import requests

from utils.pipelines.main import pop_system_message


class Pipeline:
    class Valves(BaseModel):
        AWS_ACCESS_KEY: str = ""
        AWS_SECRET_KEY: str = ""
        AWS_REGION_NAME: str = ""

    def __init__(self):
        self.type = "manifold"
        self.name = "AWS Bedrock: "

        self.valves = self.Valves(
            **{
                "AWS_ACCESS_KEY": os.getenv("AWS_ACCESS_KEY", "your-aws-access-key-here"),
                "AWS_SECRET_KEY": os.getenv("AWS_SECRET_KEY", "your-aws-secret-key-here"),
                "AWS_REGION_NAME": os.getenv("AWS_REGION_NAME", "your-aws-region-name-here"),
            }
        )

        self.bedrock = None
        self.bedrock_runtime = None
        self.pipelines = []
        self.initialize_clients()

    def initialize_clients(self):
        try:
            self.bedrock = boto3.client(
                aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                service_name="bedrock",
                region_name=self.valves.AWS_REGION_NAME
            )
            self.bedrock_runtime = boto3.client(
                aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                service_name="bedrock-runtime",
                region_name=self.valves.AWS_REGION_NAME
            )
            self.pipelines = self.get_models()
        except ClientError as e:
            print(f"Failed to initialize AWS clients: {e}")
            self.pipelines = [{"id": "error", "name": "Failed to initialize AWS clients. Please check your credentials."}]

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        print(f"on_valves_updated:{__name__}")
        self.initialize_clients()

    def get_models(self):
        if not self.bedrock:
            return []
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
            print(f"Error fetching models: {e}")
            return [
                {
                    "id": "error",
                    "name": "Could not fetch models from Bedrock, please update the Access/Secret Key in the valves.",
                },
            ]

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        system_message, messages = pop_system_message(messages)

        logging.info(f"pop_system_message: {json.dumps(messages)}")

        try:
            processed_messages = self.process_messages(messages)
            
            payload = {
                "modelId": model_id,
                "messages": processed_messages,
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": body.get("max_tokens", 4096),
                "temperature": body.get("temperature", 0.7),
                "top_k": body.get("top_k", 250),
                "top_p": body.get("top_p", 1),
                "stop_sequences": body.get("stop", []),
                "stream": body.get("stream", False),
            }

            if system_message:
                payload["system"] = system_message

            if body.get("stream", False):
                return self.stream_response(model_id, payload)
            else:
                return self.get_completion(model_id, payload)
        except Exception as e:
            return f"Error: {e}"

    def process_messages(self, messages: List[dict]) -> List[dict]:
        processed_messages = []
        image_count = 0
        for message in messages:
            processed_content = []
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_content.append(item["text"])
                    elif item["type"] == "image_url":
                        if image_count >= 20:
                            raise ValueError("Maximum of 20 images per API call exceeded")
                        processed_image = self.process_image(item["image_url"])
                        processed_content.append(processed_image)
                        image_count += 1
            else:
                processed_content = message.get("content", "")

            processed_messages.append({"role": message["role"], "content": processed_content})
        return processed_messages

    def process_image(self, image: dict) -> str:
        img_stream = None
        if image["url"].startswith("data:image"):
            base64_string = image["url"].split(',')[1]
            image_data = base64.b64decode(base64_string)
            img_stream = BytesIO(image_data)
        else:
            img_stream = requests.get(image["url"]).content
        
        image_base64 = base64.b64encode(img_stream.read()).decode('utf-8')
        return f"<image>{image_base64}</image>"

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
            return f"Error: {e}"
