"""
title: Gemini with Google Search Grounding + File Support + Image Generation + Token Usage
id: gemini_grounding_file_image_pipe
description: Complete Gemini pipe with native Google Search grounding, comprehensive file/image support, image generation, and token usage tracking for OpenWebUI 0.6.15
author: Enhanced Implementation
version: 1.4.0
license: MIT
requirements: google-genai>=1.21.0
"""

import os
import re
import json
import base64
import asyncio
import logging
import mimetypes
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Literal
from pydantic import BaseModel, Field
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError, APIError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipe:
    class Valves(BaseModel):
        # Authentication
        GEMINI_API_KEY: str = Field(
            default="",
            description="Gemini API Key - Get from https://aistudio.google.com/app/apikey",
        )

        # Grounding Configuration
        ENABLE_AUTO_GROUNDING: bool = Field(
            default=True,
            description="Automatically enable grounding for web search requests",
        )
        GROUNDING_TEMPERATURE: float = Field(
            default=0.0,
            description="Temperature when grounding is active (0.0 recommended)",
        )
        DYNAMIC_RETRIEVAL_THRESHOLD: float = Field(
            default=0.3, description="Dynamic retrieval threshold (0.0-1.0)"
        )

        # Image Generation Configuration
        ENABLE_IMAGE_GENERATION: bool = Field(
            default=True,
            description="Enable image generation capabilities",
        )
        IMAGE_GENERATION_MODEL: str = Field(
            default="gemini-2.0-flash-preview-image-generation",
            description="Model for image generation (gemini-2.0-flash-preview-image-generation or imagen models)",
        )
        IMAGE_GENERATION_ONLY_MODEL: str = Field(
            default="imagen-4.0-generate-preview-06-06",
            description="Imagen model for image-only generation",
        )
        AUTO_DETECT_IMAGE_REQUESTS: bool = Field(
            default=True,
            description="Automatically detect when user wants to generate images",
        )
        IMAGE_ASPECT_RATIO: str = Field(
            default="1:1",
            description="Default aspect ratio for Imagen (1:1, 3:4, 4:3, 9:16, 16:9)",
        )
        IMAGE_PERSON_GENERATION: str = Field(
            default="allow_adult",
            description="Person generation policy (dont_allow, allow_adult, allow_all)",
        )
        NUMBER_OF_IMAGES: int = Field(
            default=1,
            description="Number of images to generate with Imagen (1-4, Imagen 4 Ultra: 1 only)",
        )

        # File Upload Configuration
        ENABLE_FILE_UPLOADS: bool = Field(
            default=True,
            description="Enable file upload processing",
        )
        MAX_FILE_SIZE_MB: int = Field(
            default=20,
            description="Maximum file size in MB",
        )
        SUPPORTED_IMAGE_FORMATS: str = Field(
            default="jpg,jpeg,png,gif,bmp,webp",
            description="Supported image formats (comma-separated)",
        )
        SUPPORTED_DOCUMENT_FORMATS: str = Field(
            default="pdf,txt,md,json,csv,xml,html",
            description="Supported document formats (comma-separated)",
        )
        AUTO_EXTRACT_TEXT: bool = Field(
            default=True,
            description="Automatically extract and include text from documents",
        )

        # Model Configuration
        THINKING_BUDGET: int = Field(
            default=8192, description="Thinking budget for Gemini 2.5 models (0-32768)"
        )
        ENABLE_ADAPTIVE_THINKING: bool = Field(
            default=True,
            description="Automatically adjust thinking budget based on query",
        )

        # Token Usage Configuration
        ENABLE_TOKEN_TRACKING: bool = Field(
            default=True,
            description="Enable detailed token usage tracking and logging",
        )
        LOG_TOKEN_USAGE: bool = Field(
            default=True,
            description="Log token usage to console for debugging",
        )
        EMIT_DETAILED_USAGE: bool = Field(
            default=True,
            description="Emit detailed usage breakdowns including thinking tokens",
        )

        # Performance
        MODEL_CACHE_TTL: int = Field(
            default=600, description="Model cache time in seconds"
        )
        RETRY_COUNT: int = Field(default=3, description="Number of retry attempts")

    def __init__(self):
        self.valves = self.Valves()
        self._model_cache = None
        self._model_cache_time = 0

    def _get_client(self) -> genai.Client:
        """Initialize and return Gemini client"""
        if not self.valves.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY required")
        return genai.Client(api_key=self.valves.GEMINI_API_KEY)

    def _strip_model_prefix(self, model_name: str) -> str:
        """Strip pipeline prefix from model name"""
        return re.sub(r"^.*?[./]", "", model_name)

    def _get_mime_type(self, filename: str, data: bytes) -> str:
        """Get MIME type from filename and data"""
        # Try to get MIME type from filename
        mime_type, _ = mimetypes.guess_type(filename)

        if mime_type:
            return mime_type

        # Fallback to detecting from file signature
        if data.startswith(b"\x89PNG"):
            return "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            return "image/gif"
        elif data.startswith(b"RIFF") and b"WEBP" in data[:12]:
            return "image/webp"
        elif data.startswith(b"%PDF"):
            return "application/pdf"
        elif filename.lower().endswith(".txt"):
            return "text/plain"
        elif filename.lower().endswith(".md"):
            return "text/markdown"
        elif filename.lower().endswith(".json"):
            return "application/json"
        elif filename.lower().endswith(".csv"):
            return "text/csv"
        elif filename.lower().endswith(".xml"):
            return "application/xml"
        elif filename.lower().endswith(".html"):
            return "text/html"
        else:
            return "application/octet-stream"

    def _is_supported_file(self, filename: str, mime_type: str) -> bool:
        """Check if file type is supported"""
        if not self.valves.ENABLE_FILE_UPLOADS:
            return False

        extension = Path(filename).suffix.lower().lstrip(".")

        supported_images = self.valves.SUPPORTED_IMAGE_FORMATS.lower().split(",")
        supported_docs = self.valves.SUPPORTED_DOCUMENT_FORMATS.lower().split(",")

        return extension in supported_images or extension in supported_docs

    def _extract_text_content(
        self, data: bytes, mime_type: str, filename: str
    ) -> Optional[str]:
        """Extract text content from supported document formats"""
        if not self.valves.AUTO_EXTRACT_TEXT:
            return None

        try:
            if mime_type.startswith("text/") or mime_type in [
                "application/json",
                "application/xml",
            ]:
                # Handle text-based files
                encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
                for encoding in encodings:
                    try:
                        return data.decode(encoding)
                    except UnicodeDecodeError:
                        continue

            # For other formats, return None to let Gemini handle natively
            return None

        except Exception as e:
            logger.warning(f"Failed to extract text from {filename}: {e}")
            return None

    def _detect_image_generation_request(self, messages: List[Dict]) -> bool:
        """Detect if the user is requesting image generation"""
        if (
            not self.valves.ENABLE_IMAGE_GENERATION
            or not self.valves.AUTO_DETECT_IMAGE_REQUESTS
        ):
            return False

        if not messages:
            return False

        latest_msg = messages[-1]
        content = latest_msg.get("content", "")

        if isinstance(content, list):
            text = " ".join(
                item.get("text", "") for item in content if item.get("type") == "text"
            )
        else:
            text = str(content)

        text_lower = text.lower()

        # Patterns that indicate image generation requests
        image_generation_patterns = [
            r"\b(generate|create|make|draw|produce|design)\s+(?:an?\s+)?image",
            r"\b(generate|create|make|draw|produce|design)\s+(?:a\s+)?(?:picture|photo|illustration|artwork)",
            r"\bimage\s+(?:of|showing|depicting|with)",
            r"\bpicture\s+(?:of|showing|depicting|with)",
            r"\b(?:can\s+you\s+)?(?:generate|create|make|draw|show\s+me)\s+",
            r"\bvisuali[sz]e",
            r"\bart(?:work)?\s+(?:of|showing|depicting)",
            r"\bsketch|drawing|painting|illustration",
            r"\brender|rendering",
        ]

        # Also check for explicit image generation commands
        explicit_patterns = [
            r"^/?(?:image|img|generate|create|make|draw)",
            r"\btext.to.image\b",
            r"\bimg2txt\b",
        ]

        all_patterns = image_generation_patterns + explicit_patterns

        for pattern in all_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"Detected image generation request: '{text[:100]}...'")
                return True

        return False

    def _is_image_generation_model(self, model_id: str) -> bool:
        """Check if the model is specifically for image generation"""
        image_models = [
            "gemini-2.0-flash-preview-image-generation",
            "imagen-3.0-generate-001",
            "imagen-3.0-fast-generate-001",
            "imagen-4.0-generate-preview-06-06",
            "imagen-4.0-ultra-generate-preview-06-06",
        ]
        return model_id in image_models

    async def _process_file(self, filename: str) -> Optional[types.Part]:
        """Process uploaded file and return appropriate Part"""
        try:
            # Read file using OpenWebUI's file system API
            try:
                # Try reading as binary first
                file_data = await self._read_file_async(filename)
            except Exception as e:
                logger.error(f"Failed to read file {filename}: {e}")
                return None

            # Check file size
            file_size_mb = len(file_data) / (1024 * 1024)
            if file_size_mb > self.valves.MAX_FILE_SIZE_MB:
                logger.warning(
                    f"File {filename} ({file_size_mb:.1f}MB) exceeds size limit"
                )
                return None

            # Get MIME type
            mime_type = self._get_mime_type(filename, file_data)

            # Check if file type is supported
            if not self._is_supported_file(filename, mime_type):
                logger.warning(f"Unsupported file type: {filename} ({mime_type})")
                return None

            logger.info(
                f"Processing file: {filename} ({mime_type}, {file_size_mb:.1f}MB)"
            )

            # For images, send directly to Gemini
            if mime_type.startswith("image/"):
                return types.Part.from_bytes(data=file_data, mime_type=mime_type)

            # For PDFs, send directly to Gemini (it can handle them natively)
            elif mime_type == "application/pdf":
                return types.Part.from_bytes(data=file_data, mime_type=mime_type)

            # For text-based documents, try to extract text
            else:
                text_content = self._extract_text_content(
                    file_data, mime_type, filename
                )
                if text_content:
                    # Include filename and content
                    formatted_content = f"File: {filename}\nContent:\n{text_content}"
                    return types.Part(text=formatted_content)
                else:
                    # Send binary data with MIME type
                    return types.Part.from_bytes(data=file_data, mime_type=mime_type)

        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return None

    async def _read_file_async(self, filename: str) -> bytes:
        """Async wrapper for reading files"""
        # This is a placeholder - in OpenWebUI, you would use their file API
        # For now, we'll simulate it with a regular file read
        # In actual OpenWebUI environment, this would be:
        # return await window.fs.readFile(filename)

        # For demonstration, we'll try to read from a uploads directory
        try:
            with open(filename, "rb") as f:
                return f.read()
        except:
            # In OpenWebUI, files are typically accessible via their API
            # This is where you'd integrate with OpenWebUI's file system
            raise FileNotFoundError(f"Could not access file: {filename}")

    def _detect_query_complexity(self, messages: List[Dict]) -> str:
        """Detect query complexity for adaptive thinking"""
        if not messages:
            return "medium"

        latest_msg = messages[-1]
        content = latest_msg.get("content", "")

        if isinstance(content, list):
            text = " ".join(
                item.get("text", "") for item in content if item.get("type") == "text"
            )
        else:
            text = str(content)

        text_lower = text.lower()

        # High complexity patterns
        high_patterns = [
            r"\b(code|algorithm|debug|programming|analyze|complex|calculate|research)\b",
            r"\b(step.by.step|detailed|comprehensive|explain.*how)\b",
            r"\b(analyze.*file|extract.*data|summarize.*document)\b",  # File analysis
            r"\b(generate|create|make|draw|design).*image\b",  # Image generation
        ]

        # Low complexity patterns
        low_patterns = [
            r"\b(what is|who is|define|simple|hello|thanks)\b",
            r"\b(yes.*no|true.*false)\b",
        ]

        for pattern in high_patterns:
            if re.search(pattern, text_lower):
                return "high"

        for pattern in low_patterns:
            if re.search(pattern, text_lower):
                return "low"

        return "medium"

    def _should_enable_grounding(self, messages: List[Dict]) -> bool:
        """Determine if grounding would benefit this query"""
        if not messages:
            return False

        latest_msg = messages[-1]
        content = latest_msg.get("content", "")

        if isinstance(content, list):
            text = " ".join(
                item.get("text", "") for item in content if item.get("type") == "text"
            )
        else:
            text = str(content)

        text_lower = text.lower()

        # Patterns that benefit from grounding
        grounding_patterns = [
            r"\b(current|latest|recent|today|now|2024|2025)\b",
            r"\b(news|weather|stock|price|market|result)\b",
            r"\b(who.*won|what.*happening|update|status)\b",
            r"\b(search|find.*online|look.*up|verify)\b",
        ]

        return any(re.search(pattern, text_lower) for pattern in grounding_patterns)

    def _emit_usage_data(self, response, __event_emitter__=None):
        """Emit token usage data to OpenWebUI"""
        if (
            not self.valves.ENABLE_TOKEN_TRACKING
            or not __event_emitter__
            or not hasattr(response, "usage_metadata")
        ):
            return

        try:
            usage_metadata = response.usage_metadata

            # Extract token counts with proper fallbacks
            prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
            completion_tokens = getattr(usage_metadata, "candidates_token_count", 0)
            total_tokens = getattr(usage_metadata, "total_token_count", 0)

            # Build usage data structure
            usage_data = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

            # Add thinking tokens if available (Gemini 2.5 models)
            if hasattr(usage_metadata, "thinking_token_count"):
                thinking_tokens = usage_metadata.thinking_token_count
                usage_data["thinking_tokens"] = thinking_tokens
                # Update total to include thinking tokens if not already included
                if total_tokens < (prompt_tokens + completion_tokens + thinking_tokens):
                    usage_data["total_tokens"] = (
                        prompt_tokens + completion_tokens + thinking_tokens
                    )

            # Emit to OpenWebUI
            __event_emitter__({"type": "usage", "data": usage_data})

            if self.valves.LOG_TOKEN_USAGE:
                logger.info(
                    f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {usage_data['total_tokens']}"
                )
                if "thinking_tokens" in usage_data:
                    logger.info(f"Thinking tokens: {usage_data['thinking_tokens']}")

        except Exception as e:
            logger.error(f"Error emitting usage data: {e}")

    def _get_supported_models(self) -> List[Dict[str, str]]:
        """Get list of supported Gemini models including image generation models"""
        try:
            client = self._get_client()
            models = client.models.list()

            gemini_models = []
            for model in models:
                if model.name and "gemini" in model.name.lower():
                    model_id = self._strip_model_prefix(model.name)
                    if "generateContent" in (model.supported_actions or []):
                        gemini_models.append(
                            {
                                "id": model_id,
                                "name": model.display_name or model_id,
                                "description": self._get_model_description(model_id),
                            }
                        )

            # Add image generation models if enabled
            if self.valves.ENABLE_IMAGE_GENERATION:
                image_models = [
                    {
                        "id": "gemini-2.0-flash-preview-image-generation",
                        "name": "Gemini 2.0 Flash Image Generation",
                        "description": "Conversational image generation | 🧠 Thinking | 🔍 Grounding | 🎨 Images",
                    },
                    {
                        "id": "imagen-4.0-generate-preview-06-06",
                        "name": "Imagen 4.0 Generate",
                        "description": "High-quality image generation | 🎨 Images | Multiple outputs",
                    },
                    {
                        "id": "imagen-4.0-ultra-generate-preview-06-06",
                        "name": "Imagen 4.0 Ultra",
                        "description": "Best quality image generation | 🎨 Images | Single output",
                    },
                    {
                        "id": "imagen-3.0-generate-001",
                        "name": "Imagen 3.0 Generate",
                        "description": "Standard image generation | 🎨 Images",
                    },
                    {
                        "id": "imagen-3.0-fast-generate-001",
                        "name": "Imagen 3.0 Fast",
                        "description": "Fast image generation | 🎨 Images",
                    },
                ]

                # Add image models that aren't already in the list
                existing_ids = {model["id"] for model in gemini_models}
                for img_model in image_models:
                    if img_model["id"] not in existing_ids:
                        gemini_models.append(img_model)

            return gemini_models
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            base_models = [
                {
                    "id": "gemini-2.5-flash",
                    "name": "Gemini 2.5 Flash",
                    "description": "Fast, efficient model with thinking, grounding, and file support",
                }
            ]
            if self.valves.ENABLE_IMAGE_GENERATION:
                base_models.extend(
                    [
                        {
                            "id": "gemini-2.0-flash-preview-image-generation",
                            "name": "Gemini 2.0 Flash Image Generation",
                            "description": "Conversational image generation | 🧠 Thinking | 🔍 Grounding | 🎨 Images",
                        },
                        {
                            "id": "imagen-4.0-generate-preview-06-06",
                            "name": "Imagen 4.0 Generate",
                            "description": "High-quality image generation | 🎨 Images",
                        },
                    ]
                )
            return base_models

    def _get_model_description(self, model_id: str) -> str:
        """Get enhanced description with capabilities"""
        capabilities = []

        if "2.5" in model_id:
            capabilities.append("🧠 Thinking")
        if any(v in model_id for v in ["2.5", "2.0", "1.5", "1.0"]):
            capabilities.append("🔍 Grounding")
        if "2." in model_id:
            capabilities.append("💻 Code Execution")
        if "image-generation" in model_id or "imagen" in model_id:
            capabilities.append("🎨 Images")

        # All models support files
        capabilities.append("📁 Files")

        base_desc = "Advanced AI model"
        if capabilities:
            return f"{base_desc} | {' | '.join(capabilities)}"
        return base_desc

    async def _prepare_content(
        self, messages: List[Dict]
    ) -> tuple[List[types.Content], Optional[str]]:
        """Prepare messages for Gemini API with enhanced file support"""
        system_message = None
        contents = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_message = content
                continue

            # Process content
            parts = []

            # Handle file references in OpenWebUI format
            files = msg.get("files", [])
            for file_info in files:
                if isinstance(file_info, dict):
                    filename = file_info.get("name") or file_info.get("filename")
                    if filename:
                        file_part = await self._process_file(filename)
                        if file_part:
                            parts.append(file_part)
                            logger.info(f"Added file to message: {filename}")
                elif isinstance(file_info, str):
                    # Direct filename
                    file_part = await self._process_file(file_info)
                    if file_part:
                        parts.append(file_part)
                        logger.info(f"Added file to message: {file_info}")

            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append(types.Part(text=item.get("text", "")))
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image"):
                            try:
                                header, encoded = image_url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                parts.append(
                                    types.Part.from_bytes(
                                        data=base64.b64decode(encoded),
                                        mime_type=mime_type,
                                    )
                                )
                                logger.info("Added inline image to message")
                            except Exception as e:
                                logger.warning(f"Failed to process image data: {e}")
                    elif item.get("type") == "file":
                        # Handle file references in content
                        filename = (
                            item.get("name") or item.get("filename") or item.get("url")
                        )
                        if filename:
                            file_part = await self._process_file(filename)
                            if file_part:
                                parts.append(file_part)
                                logger.info(
                                    f"Added referenced file to message: {filename}"
                                )
            else:
                parts.append(types.Part(text=str(content)))

            if parts:
                api_role = "model" if role == "assistant" else "user"
                contents.append(types.Content(role=api_role, parts=parts))

        return contents, system_message

    def _configure_generation(
        self,
        body: Dict,
        system_message: Optional[str],
        model_id: str,
        enable_grounding: bool,
        is_image_generation: bool = False,
    ) -> types.GenerateContentConfig:
        """Configure generation parameters"""

        # Base configuration
        config = types.GenerateContentConfig(
            system_instruction=system_message,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            top_k=body.get("top_k"),
            max_output_tokens=body.get("max_tokens"),
            stop_sequences=body.get("stop"),
        )

        # Configure response modalities for image generation
        if is_image_generation and "image-generation" in model_id:
            config.response_modalities = ["TEXT", "IMAGE"]
            logger.info("Enabled image generation response modalities")

        # Configure thinking for 2.5 models
        if "2.5" in model_id:
            thinking_budget = self.valves.THINKING_BUDGET

            if self.valves.ENABLE_ADAPTIVE_THINKING:
                complexity = self._detect_query_complexity(body.get("messages", []))

                # Check if files are present - increases complexity
                has_files = any(msg.get("files") for msg in body.get("messages", []))
                if has_files or is_image_generation:
                    complexity = "high"

                if complexity == "high":
                    thinking_budget = min(16384, thinking_budget * 2)
                elif complexity == "low":
                    thinking_budget = max(2048, thinking_budget // 2)
                    if "flash-lite" in model_id.lower():
                        thinking_budget = 0  # Disable for simple tasks

                # Reduce budget when grounding is active
                if enable_grounding:
                    thinking_budget = min(thinking_budget, 6144)

            # Apply model limits
            max_budget = 32768 if "pro" in model_id.lower() else 24576
            thinking_budget = min(thinking_budget, max_budget)

            config.thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget, include_thoughts=True
            )

            logger.info(f"Thinking budget: {thinking_budget} for {model_id}")

        # Configure grounding
        config.tools = []
        if (
            enable_grounding and not is_image_generation
        ):  # Don't enable grounding for pure image generation
            # Use Search as Tool for 2.0+ models, Search Retrieval for 1.x
            if any(v in model_id for v in ["2.5", "2.0"]):
                config.tools.append(types.Tool(google_search=types.GoogleSearch()))
                logger.info("Enabled Google Search as Tool")
            else:
                gs_retrieval = types.GoogleSearchRetrieval(
                    dynamic_retrieval_config=types.DynamicRetrievalConfig(
                        dynamic_threshold=self.valves.DYNAMIC_RETRIEVAL_THRESHOLD
                    )
                )
                config.tools.append(types.Tool(google_search_retrieval=gs_retrieval))
                logger.info("Enabled Google Search Retrieval")

            # Override temperature for grounding accuracy
            config.temperature = self.valves.GROUNDING_TEMPERATURE
            logger.info(
                f"Set temperature to {self.valves.GROUNDING_TEMPERATURE} for grounding"
            )

        return config

    async def _generate_with_imagen(
        self, prompt: str, model_id: str
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate images using Imagen models"""
        try:
            client = self._get_client()

            config = types.GenerateImagesConfig(
                number_of_images=self.valves.NUMBER_OF_IMAGES,
                aspect_ratio=self.valves.IMAGE_ASPECT_RATIO,
                person_generation=self.valves.IMAGE_PERSON_GENERATION,
            )

            # Imagen 4 Ultra can only generate 1 image
            if "ultra" in model_id.lower():
                config.number_of_images = 1

            logger.info(
                f"Generating {config.number_of_images} image(s) with {model_id}"
            )

            response = await self._retry_with_backoff(
                client.aio.models.generate_images,
                model=model_id,
                prompt=prompt,
                config=config,
            )

            if not response.generated_images:
                return "❌ No images were generated. Please try again with a different prompt."

            # Format response
            result = f"🎨 **Generated {len(response.generated_images)} image(s) with {model_id}**\n\n"
            result += f"**Prompt:** {prompt}\n\n"

            for i, generated_image in enumerate(response.generated_images, 1):
                try:
                    image_base64 = None

                    if hasattr(generated_image, "image") and generated_image.image:
                        # Method 1: Use image_bytes property (most reliable according to docs)
                        if hasattr(generated_image.image, "image_bytes"):
                            image_base64 = base64.b64encode(
                                generated_image.image.image_bytes
                            ).decode("utf-8")
                            logger.info(
                                f"Converted image {i} using image_bytes ({len(generated_image.image.image_bytes)} bytes)"
                            )

                        # Method 2: Try save without format parameter
                        elif hasattr(generated_image.image, "save"):
                            try:
                                from io import BytesIO

                                buffer = BytesIO()
                                generated_image.image.save(buffer)
                                image_base64 = base64.b64encode(
                                    buffer.getvalue()
                                ).decode("utf-8")
                                logger.info(f"Converted image {i} using save() method")
                            except Exception as e:
                                logger.warning(f"Save method failed for image {i}: {e}")

                    if image_base64:
                        result += f"**Image {i}:**\n"
                        result += f"![Generated Image {i}](data:image/png;base64,{image_base64})\n\n"
                    else:
                        # Debug logging
                        logger.error(
                            f"Could not extract image {i}. Generated image structure:"
                        )
                        logger.error(f"Generated image type: {type(generated_image)}")
                        logger.error(
                            f"Generated image attributes: {dir(generated_image)}"
                        )
                        if hasattr(generated_image, "image"):
                            logger.error(f"Image type: {type(generated_image.image)}")
                            logger.error(
                                f"Image attributes: {dir(generated_image.image)}"
                            )

                        result += f"**Image {i}:** ❌ Failed to process image data\n\n"

                except Exception as e:
                    logger.error(f"Error processing image {i}: {e}")
                    result += f"**Image {i}:** ❌ Error: {str(e)}\n\n"

            return result

        except Exception as e:
            logger.error(f"Imagen generation error: {e}")
            return f"❌ Image generation failed: {str(e)}"

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff"""
        for attempt in range(self.valves.RETRY_COUNT + 1):
            try:
                return await func(*args, **kwargs)
            except (ServerError, APIError) as e:
                if attempt == self.valves.RETRY_COUNT:
                    raise
                wait_time = min(2**attempt, 10)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s"
                )
                await asyncio.sleep(wait_time)
            except Exception:
                raise

    def _safe_text_yield(self, text: str) -> str:
        """Safely escape text for streaming to prevent JSON parsing errors"""
        if not text:
            return ""

        # Only remove truly problematic control characters, preserve important whitespace
        # Remove null bytes and other dangerous control chars but keep \n, \t, \r
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Normalize line endings but preserve them
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove any stray quotes or backslashes that might be at the end
        # that could interfere with JSON parsing
        text = text.rstrip("\\\"'")

        return text

    async def _handle_streaming_response(
        self, response_stream, __event_emitter__=None
    ) -> AsyncGenerator[str, None]:
        """Handle streaming response with enhanced processing and image support"""
        final_response = None
        thinking_content = ""
        in_thinking = False
        image_count = 0

        try:
            async for chunk in response_stream:
                final_response = chunk  # Keep reference for usage data

                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    continue

                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        text_content = str(part.text)

                        # Handle thinking content - convert to OpenWebUI format
                        if hasattr(part, "thought") and part.thought:
                            # This is thinking content from Gemini 2.5 models
                            if not in_thinking:
                                yield "<thinking>\n"
                                in_thinking = True

                            safe_text = self._safe_text_yield(text_content)
                            thinking_content += safe_text
                            yield safe_text
                        else:
                            # Regular content
                            if in_thinking:
                                # Close thinking section if we were in one
                                yield "\n</thinking>\n\n"
                                in_thinking = False

                            # Yield regular content with safety escaping
                            safe_text = self._safe_text_yield(text_content)
                            yield safe_text

                    elif hasattr(part, "inline_data") and part.inline_data:
                        # Handle generated images
                        if in_thinking:
                            yield "\n</thinking>\n\n"
                            in_thinking = False

                        image_count += 1
                        try:
                            # Convert binary image data to base64
                            image_base64 = base64.b64encode(
                                part.inline_data.data
                            ).decode("utf-8")

                            # Determine MIME type
                            mime_type = getattr(
                                part.inline_data, "mime_type", "image/png"
                            )

                            yield f"\n🎨 **Generated Image {image_count}:**\n"
                            yield f"![Generated Image {image_count}](data:{mime_type};base64,{image_base64})\n\n"

                            logger.info(f"Generated and yielded image {image_count}")
                        except Exception as e:
                            logger.error(f"Error processing generated image: {e}")
                            yield f"\n❌ Error displaying generated image {image_count}: {str(e)}\n\n"

                    elif hasattr(part, "executable_code") and part.executable_code:
                        # Code execution
                        if in_thinking:
                            yield "\n</thinking>\n\n"
                            in_thinking = False

                        code_text = f"```python\n{part.executable_code.code}\n```\n\n"
                        yield self._safe_text_yield(code_text)

                    elif (
                        hasattr(part, "code_execution_result")
                        and part.code_execution_result
                    ):
                        # Code results
                        if in_thinking:
                            yield "\n</thinking>\n\n"
                            in_thinking = False

                        result_text = f"**Output:**\n```\n{part.code_execution_result.output}\n```\n\n"
                        yield self._safe_text_yield(result_text)

            # Close thinking section if it was left open
            if in_thinking:
                yield "\n</thinking>\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            if in_thinking:
                yield "\n</thinking>\n\n"
            # Only log the error, don't yield it to avoid JSON issues
            logger.error(f"Streaming error details: {str(e)}")

        # Emit usage data at the end of streaming
        if final_response:
            self._emit_usage_data(final_response, __event_emitter__)

    def pipes(self) -> List[Dict[str, str]]:
        """Return available models"""
        try:
            return self._get_supported_models()
        except Exception as e:
            logger.error(f"Error in pipes: {e}")
            return [{"id": "error", "name": f"Error: {str(e)}"}]

    async def pipe(
        self, body: Dict[str, Any], __event_emitter__=None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Main pipe function with enhanced file support and image generation"""
        try:
            # Get model and messages
            model_id = self._strip_model_prefix(body.get("model", ""))
            messages = body.get("messages", [])
            stream = body.get("stream", False)

            logger.info(f"Processing request for model: {model_id}")

            # Check for files in messages
            has_files = any(msg.get("files") for msg in messages)
            if has_files:
                logger.info("File uploads detected in request")

            # Check if this is an image generation request
            is_image_generation = self._is_image_generation_model(
                model_id
            ) or self._detect_image_generation_request(messages)

            # For Imagen models or detected image requests, use appropriate model
            if is_image_generation:
                if "imagen" in model_id:
                    # Direct Imagen model request
                    if messages:
                        prompt = messages[-1].get("content", "")
                        if isinstance(prompt, list):
                            prompt = " ".join(
                                item.get("text", "")
                                for item in prompt
                                if item.get("type") == "text"
                            )
                        return await self._generate_with_imagen(prompt, model_id)
                    else:
                        return "❌ No prompt provided for image generation"

                elif not "image-generation" in model_id:
                    # Auto-detected image request but not using image generation model
                    # Switch to the default image generation model
                    if "imagen" in self.valves.IMAGE_GENERATION_MODEL:
                        prompt = messages[-1].get("content", "") if messages else ""
                        if isinstance(prompt, list):
                            prompt = " ".join(
                                item.get("text", "")
                                for item in prompt
                                if item.get("type") == "text"
                            )
                        return await self._generate_with_imagen(
                            prompt, self.valves.IMAGE_GENERATION_MODEL
                        )
                    else:
                        # Use Gemini image generation model
                        model_id = self.valves.IMAGE_GENERATION_MODEL

            # Check for web search feature
            web_search_enabled = body.get("features", {}).get("web_search", False)

            # Auto-enable grounding if beneficial (but not for pure image generation)
            enable_grounding = web_search_enabled or (
                self.valves.ENABLE_AUTO_GROUNDING
                and self._should_enable_grounding(messages)
                and not (is_image_generation and "imagen" in model_id)
            )

            if enable_grounding:
                logger.info("Grounding enabled for this request")

            # Prepare content (now with file support)
            contents, system_message = await self._prepare_content(messages)
            if not contents:
                return "Error: No valid message content found"

            # Configure generation
            config = self._configure_generation(
                body, system_message, model_id, enable_grounding, is_image_generation
            )

            # Make API call
            client = self._get_client()

            if stream:
                response_stream = await self._retry_with_backoff(
                    client.aio.models.generate_content_stream,
                    model=model_id,
                    contents=contents,
                    config=config,
                )
                return self._handle_streaming_response(
                    response_stream, __event_emitter__
                )
            else:
                response = await self._retry_with_backoff(
                    client.aio.models.generate_content,
                    model=model_id,
                    contents=contents,
                    config=config,
                )

                # Emit usage data for non-streaming response
                self._emit_usage_data(response, __event_emitter__)

                if response.text or (
                    response.candidates and response.candidates[0].content
                ):
                    result_text = ""

                    # Handle text content
                    if response.text:
                        result_text = response.text

                    # Handle generated images in non-streaming mode
                    if response.candidates:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            image_count = 0
                            for part in candidate.content.parts:
                                if hasattr(part, "inline_data") and part.inline_data:
                                    image_count += 1
                                    try:
                                        # Convert binary image data to base64
                                        image_base64 = base64.b64encode(
                                            part.inline_data.data
                                        ).decode("utf-8")

                                        # Determine MIME type
                                        mime_type = getattr(
                                            part.inline_data, "mime_type", "image/png"
                                        )

                                        result_text += (
                                            f"\n🎨 **Generated Image {image_count}:**\n"
                                        )
                                        result_text += f"![Generated Image {image_count}](data:{mime_type};base64,{image_base64})\n\n"

                                        logger.info(
                                            f"Generated and added image {image_count} to response"
                                        )
                                    except Exception as e:
                                        logger.error(
                                            f"Error processing generated image: {e}"
                                        )
                                        result_text += f"\n❌ Error displaying generated image {image_count}: {str(e)}\n\n"

                    result_text = self._safe_text_yield(result_text)
                    return result_text if result_text else "No content generated"
                else:
                    return "No content generated"

        except ClientError as e:
            error_msg = f"Client error: {e}"
            logger.error(error_msg)
            return self._safe_text_yield(error_msg)
        except (ServerError, APIError) as e:
            error_msg = f"API error: {e}"
            logger.error(error_msg)
            return self._safe_text_yield(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(error_msg)
            return self._safe_text_yield(error_msg)
