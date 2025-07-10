"""
title: AWS Bedrock Agent Pipe Function
author: Assistant
version: 1.1.0
description: A Pipe Function that integrates AWS Bedrock Agents with OpenWebUI, allowing you to interact with your AWS Bedrock Agents as custom models.
license: MIT
"""

import boto3
import json
import uuid
import asyncio
import re
from typing import List, Union, Generator, Iterator, Dict, Any, Optional
from pydantic import BaseModel, Field
from botocore.exceptions import ClientError


class Pipe:
    class Valves(BaseModel):
        """Configuration parameters for the AWS Bedrock Agent Pipe"""

        AWS_ACCESS_KEY_ID: str = Field(
            default="", description="AWS Access Key ID for authentication"
        )
        AWS_SECRET_ACCESS_KEY: str = Field(
            default="", description="AWS Secret Access Key for authentication"
        )
        AWS_SESSION_TOKEN: str = Field(
            default="",
            description="AWS Session Token (required for temporary credentials)",
        )
        AWS_REGION: str = Field(
            default="us-east-1",
            description="AWS region where your Bedrock Agent is deployed",
        )
        AGENT_ID: str = Field(default="", description="Your AWS Bedrock Agent ID")
        AGENT_ALIAS_ID: str = Field(
            default="TSTALIASID",
            description="Agent Alias ID (default: TSTALIASID for test alias)",
        )
        SESSION_TIMEOUT: int = Field(
            default=600, description="Session timeout in seconds"
        )
        ENABLE_TRACE: bool = Field(
            default=False, description="Enable detailed trace logging for debugging"
        )
        MAX_TOKENS: int = Field(default=2048, description="Maximum tokens for response")
        SHOW_DOCUMENT_CONTENT: bool = Field(
            default=False, description="Show document content excerpts in citations"
        )
        DEBUG_CITATIONS: bool = Field(
            default=False, description="Enable debug logging for citations"
        )
        ENABLE_FOLLOW_UPS: bool = Field(
            default=True, description="Enable follow-up questions in responses"
        )

    def __init__(self):
        """Initialize the AWS Bedrock Agent Pipe"""
        self.type = "manifold"
        self.id = "aws_bedrock_agent"
        self.name = "AWS Bedrock Agent"
        self.valves = self.Valves()
        self.session_cache = {}  # Cache for session management

    def pipes(self) -> List[dict]:
        """Define the available models/agents"""
        return [
            {
                "id": "bedrock_agent",
                "name": "AWS Bedrock Agent",
                "description": "AWS Bedrock Agent integration",
            }
        ]

    def _detect_language(self, text: str) -> str:
        """Detect if the text is in Spanish or English"""
        # Simple detection based on common Spanish words and patterns
        spanish_indicators = [
            # Common Spanish words
            "que",
            "cual",
            "cuales",
            "como",
            "donde",
            "cuando",
            "quien",
            "por que",
            "porque",
            "el",
            "la",
            "los",
            "las",
            "un",
            "una",
            "de",
            "del",
            "para",
            "con",
            "sin",
            "es",
            "son",
            "esta",
            "estan",
            "tiene",
            "tienen",
            "hacer",
            "ser",
            "estar",
            # Spanish question words
            "qué",
            "cuál",
            "cuáles",
            "cómo",
            "dónde",
            "cuándo",
            "quién",
            "por qué",
            # Spanish accented characters
            "á",
            "é",
            "í",
            "ó",
            "ú",
            "ñ",
            "ü",
        ]

        text_lower = text.lower()
        spanish_count = sum(
            1 for indicator in spanish_indicators if indicator in text_lower
        )

        # If we find Spanish indicators, consider it Spanish
        return "es" if spanish_count > 0 else "en"

    def _extract_json_from_text(self, text: str) -> tuple[str, dict]:
        """Extract JSON objects from text and return cleaned text and extracted JSON"""
        extracted_json = {}

        # Look for JSON patterns at the end of the text
        json_patterns = [
            r"```json\s*(\{.*?\})\s*```",  # JSON in code blocks
            r'\{[^{}]*"follow_ups"[^{}]*\}',  # JSON with follow_ups
            r'\{[^{}]*"tags"[^{}]*\}',  # JSON with tags
            r'\{.*?"follow_ups".*?\}',  # More flexible follow_ups pattern
        ]

        cleaned_text = text

        for pattern in json_patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
            for match in matches:
                try:
                    # Extract the JSON content
                    if "```json" in match.group(0):
                        json_str = match.group(1)
                    else:
                        json_str = match.group(0)

                    # Parse the JSON
                    json_data = json.loads(json_str)
                    extracted_json.update(json_data)

                    # Remove the JSON from the text
                    cleaned_text = cleaned_text.replace(match.group(0), "")

                except json.JSONDecodeError:
                    continue

        # Clean up any remaining artifacts
        cleaned_text = re.sub(
            r"\n\s*\n\s*\n", "\n\n", cleaned_text
        )  # Remove extra newlines
        cleaned_text = cleaned_text.strip()

        return cleaned_text, extracted_json

    def _format_follow_ups(self, follow_ups: List[str], language: str = "en") -> str:
        """Format follow-up questions in a nice way"""
        if not follow_ups:
            return ""

        header = (
            "\n\n💡 **Preguntas de seguimiento sugeridas:**\n"
            if language == "es"
            else "\n\n💡 **Suggested follow-up questions:**\n"
        )
        formatted = header

        for i, question in enumerate(follow_ups, 1):
            formatted += f"{i}. {question}\n"

        return formatted

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __task__: Optional[str] = None,
        __event_emitter__: Optional[callable] = None,
        __event_call__: Optional[callable] = None,
    ) -> Union[str, Generator, Iterator]:
        """Main pipe function that handles the agent interaction"""

        # Validate configuration
        if not self.valves.AGENT_ID:
            return "❌ Error: Agent ID not configured. Please set AGENT_ID in the function settings."

        if not self.valves.AWS_ACCESS_KEY_ID or not self.valves.AWS_SECRET_ACCESS_KEY:
            return "❌ Error: AWS credentials not configured. Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_SESSION_TOKEN (if using temporary credentials) in the function settings."

        try:
            # Extract user message from body
            messages = body.get("messages", [])
            if not messages:
                return "❌ Error: No messages found in request."

            # Get the last user message
            user_message = messages[-1].get("content", "")
            if not user_message:
                return "❌ Error: Empty message received."

            # Detect language from user message
            detected_language = self._detect_language(user_message)

            # Emit status if event emitter is available
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Connecting to AWS Bedrock Agent...",
                            "done": False,
                        },
                    }
                )

            # Initialize AWS session
            session_kwargs = {
                "aws_access_key_id": self.valves.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": self.valves.AWS_SECRET_ACCESS_KEY,
                "region_name": self.valves.AWS_REGION,
            }

            # Add session token if provided (for temporary credentials)
            if self.valves.AWS_SESSION_TOKEN:
                session_kwargs["aws_session_token"] = self.valves.AWS_SESSION_TOKEN

            session = boto3.Session(**session_kwargs)

            # Create Bedrock Agent Runtime client
            bedrock_agent_runtime = session.client("bedrock-agent-runtime")

            # Generate or get session ID for conversation continuity
            user_id = __user__.get("id", "anonymous") if __user__ else "anonymous"
            session_id = self._get_session_id(user_id)

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Invoking Bedrock Agent...",
                            "done": False,
                        },
                    }
                )

            # Prepare the invoke request
            invoke_params = {
                "agentId": self.valves.AGENT_ID,
                "agentAliasId": self.valves.AGENT_ALIAS_ID,
                "sessionId": session_id,
                "inputText": user_message,
            }

            # Add trace enablement if configured
            if self.valves.ENABLE_TRACE:
                invoke_params["enableTrace"] = True

            # Invoke the Bedrock Agent
            response = bedrock_agent_runtime.invoke_agent(**invoke_params)

            # Process the streaming response
            response_text = ""
            citations = []
            metadata = {}
            raw_chunks = []  # Store raw chunks for processing

            if "completion" in response:
                for event in response["completion"]:
                    if "chunk" in event:
                        chunk = event["chunk"]

                        if "bytes" in chunk:
                            chunk_text = chunk["bytes"].decode("utf-8")
                            raw_chunks.append(chunk_text)

                        # Handle attribution/citations
                        if "attribution" in chunk:
                            attribution = chunk["attribution"]
                            if "citations" in attribution:
                                citations.extend(attribution["citations"])

                    # Handle trace information
                    elif "trace" in event:
                        trace = event["trace"]
                        if self.valves.DEBUG_CITATIONS:
                            print(f"DEBUG - Trace event: {json.dumps(trace, indent=2)}")

                        # Look for citations in trace events
                        if "knowledgeBaseResponse" in trace:
                            kb_response = trace["knowledgeBaseResponse"]
                            if "retrievedReferences" in kb_response:
                                kb_citations = self._convert_kb_references_to_citations(
                                    kb_response["retrievedReferences"]
                                )
                                citations.extend(kb_citations)

                        if self.valves.ENABLE_TRACE:
                            print(f"Trace: {json.dumps(trace, indent=2)}")

            # Combine all chunks and process
            full_response = "".join(raw_chunks)

            # Extract JSON and clean the response
            cleaned_response, extracted_json = self._extract_json_from_text(
                full_response
            )

            # Use the cleaned response
            response_text = cleaned_response

            # Stream the cleaned response if event emitter is available
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "message", "data": {"content": response_text}}
                )

            # Add detailed citations to response if available
            if citations:
                # Use detected language for citations header
                if detected_language == "es":
                    response_text += "\n\n📚 **Fuentes y Referencias:**\n"
                else:
                    response_text += "\n\n📚 **Sources and References:**\n"
                response_text += self._format_citations(citations, detected_language)

            # Add follow-up questions if available and enabled
            if self.valves.ENABLE_FOLLOW_UPS and "follow_ups" in extracted_json:
                follow_ups = extracted_json["follow_ups"]
                if follow_ups and isinstance(follow_ups, list):
                    response_text += self._format_follow_ups(
                        follow_ups, detected_language
                    )

            # Add metadata tags if available
            if extracted_json.get("tags"):
                tags_label = "Etiquetas" if detected_language == "es" else "Tags"
                response_text += (
                    f"\n\n🏷️ **{tags_label}:** {', '.join(extracted_json['tags'])}"
                )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Response complete", "done": True},
                    }
                )

            return (
                response_text
                if response_text
                else "🤖 No response received from the agent."
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]

            if error_code == "ResourceNotFoundException":
                return f"❌ Error: Agent ID '{self.valves.AGENT_ID}' not found. Please check your Agent ID and region."
            elif error_code == "AccessDeniedException":
                return "❌ Error: Access denied. Please check your AWS credentials and IAM permissions."
            elif error_code == "ValidationException":
                return f"❌ Error: Invalid request - {error_message}"
            elif error_code == "ThrottlingException":
                return "❌ Error: Request throttled. Please try again later."
            else:
                return f"❌ AWS Error ({error_code}): {error_message}"

        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            print(error_msg)  # For debugging
            return error_msg

    def _convert_kb_references_to_citations(
        self, kb_references: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert Knowledge Base references to citation format"""
        citations = []
        for ref in kb_references:
            citation = {"retrievedReferences": [ref]}
            citations.append(citation)
        return citations

    def _format_citations(
        self, citations: List[Dict[str, Any]], language: str = "en"
    ) -> str:
        """Format citations with detailed document information"""
        formatted_citations = ""
        citation_number = 1
        unique_documents = set()  # To avoid duplicate documents

        # Language-specific labels
        relevance_label = "Relevancia" if language == "es" else "Relevance"
        excerpt_label = "Extracto" if language == "es" else "Excerpt"

        for citation in citations:
            if "retrievedReferences" in citation:
                for ref in citation["retrievedReferences"]:
                    # Extract document name
                    document_name = "Unknown Document"

                    # Extract location information
                    if "location" in ref:
                        location = ref["location"]

                        # Handle S3 location
                        if "s3Location" in location:
                            s3_location = location["s3Location"]
                            uri = s3_location.get("uri", "")

                            # Extract only the file name from URI
                            if uri:
                                document_name = uri.split("/")[-1]

                        # Handle other location types if needed
                        elif "webLocation" in location:
                            web_location = location["webLocation"]
                            url = web_location.get("url", "Unknown URL")
                            document_name = url.split("/")[-1] if "/" in url else url

                    # Skip duplicates
                    if document_name in unique_documents:
                        continue

                    unique_documents.add(document_name)
                    formatted_citations += f"\n**[{citation_number}]** {document_name}"

                    # Add confidence score if available
                    if "score" in ref:
                        score = ref["score"]
                        formatted_citations += f" ({relevance_label}: {score:.1f})"

                    # Add content excerpt if available and enabled
                    if "content" in ref and self.valves.SHOW_DOCUMENT_CONTENT:
                        content = ref["content"]
                        if "text" in content:
                            text = content["text"].strip()
                            # Truncate long content
                            if len(text) > 150:
                                text = text[:150] + "..."
                            formatted_citations += f"\n   *{excerpt_label}: {text}*"

                    formatted_citations += "\n"
                    citation_number += 1

        return formatted_citations

    def _get_session_id(self, user_id: str) -> str:
        """Generate or retrieve session ID for conversation continuity"""
        if user_id not in self.session_cache:
            self.session_cache[user_id] = str(uuid.uuid4())
        return self.session_cache[user_id]

    def _clear_session(self, user_id: str) -> None:
        """Clear session for a specific user (for new conversations)"""
        if user_id in self.session_cache:
            del self.session_cache[user_id]
