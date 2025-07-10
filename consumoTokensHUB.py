"""
title: Enhanced Token Management Filter (Sync/Async Compatible) - FIXED
author: openwebui-community
date: 2025-07-09
version: 3.2.1
license: MIT
description: Advanced filter that manages token usage with comprehensive detection and platform integration

Features:
- Synchronous operation (compatible with all OpenWebUI versions)
- Multi-turn conversation support (counts ALL conversation history)
- Thinking/reasoning token detection (internal model reasoning)
- Complete token breakdown (prompt + completion + thinking)
- Overdraft handling (graceful balance management)
- Enhanced detection strategies (multiple fallback methods)
- Platform integration (real-time usage recording)

Token Types Detected:
- Prompt tokens: Input conversation history (all messages)
- Completion tokens: Generated response content
- Thinking tokens: Internal model reasoning (o1, Claude thinking, etc.)
- Total tokens: Complete usage including all overhead

requirements: tiktoken, aiohttp, requests
"""

import os
import asyncio
import time
import re
import json
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import tiktoken

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        api_url: str = Field(
            default="http://localhost:5000",
            description="Base URL for your token management platform (where your Flask app runs)",
        )
        api_timeout: int = Field(
            default=10, description="Timeout in seconds for API requests"
        )
        tokenizer_model: str = Field(
            default="cl100k_base",
            description="Tokenizer model to use for counting tokens",
        )
        estimate_response_tokens: int = Field(
            default=150,
            description="Estimated tokens for model response (used for pre-validation)",
        )
        block_message: str = Field(
            default="❌ Insufficient tokens. Please top up your account to continue.",
            description="Message shown when user has insufficient tokens",
        )
        enable_debug: bool = Field(default=False, description="Enable debug logging")
        username_extraction_method: str = Field(
            default="email",
            description="How to extract username from user info: 'user_id', 'email', or 'username'",
        )
        enable_log_interceptor: bool = Field(
            default=True,
            description="Enable log interceptor to capture usage from other functions",
        )
        allow_overdraft: bool = Field(
            default=True,
            description="Allow token usage even when it exceeds user balance (sets balance to 0)",
        )
        include_conversation_history: bool = Field(
            default=True,
            description="Include full conversation history in token counting (recommended for accuracy)",
        )
        include_thinking_tokens: bool = Field(
            default=True,
            description="Include thinking/reasoning tokens in usage calculations",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.tokenizer = None
        self._initialize_tokenizer()

        # Enhanced persistent storage for token info
        self.active_requests = {}  # Store token info by request ID or user ID
        self.user_sessions = {}  # Store per-user session data

        # NEW: Enhanced usage capture systems
        self._captured_usage = {}  # Store usage captured from logs
        self._shared_usage_data = {}  # Store usage data shared between functions

        # Setup log interceptor if enabled
        if self.valves.enable_log_interceptor:
            self._setup_log_interceptor()

    def _get_request_id(self, body: dict, user: dict) -> str:
        """Generate a unique request ID for tracking token info"""
        # Try multiple methods to get a unique identifier
        request_id = (
            body.get("id") or body.get("request_id") or body.get("conversation_id")
        )

        if not request_id and user:
            # Fallback: use user ID + timestamp
            user_id = self.extract_username(user) or "unknown"
            request_id = f"{user_id}_{int(time.time() * 1000)}"

        return request_id or f"req_{int(time.time() * 1000)}"

    def _initialize_tokenizer(self):
        """Initialize the tokenizer"""
        try:
            self.tokenizer = tiktoken.get_encoding(self.valves.tokenizer_model)
            if self.valves.enable_debug:
                logger.info(
                    f"Tokenizer initialized with model: {self.valves.tokenizer_model}"
                )
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            # Fallback to a default tokenizer
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.warning("Using fallback tokenizer: cl100k_base")
            except Exception as fallback_error:
                logger.error(f"Fallback tokenizer also failed: {fallback_error}")
                self.tokenizer = None

    def _setup_log_interceptor(self):
        """Set up log interceptor to capture Google function usage data"""
        try:
            # Create a custom log handler to capture usage data
            class UsageLogHandler(logging.Handler):
                def __init__(self, filter_instance):
                    super().__init__()
                    self.filter_instance = filter_instance

                def emit(self, record):
                    # Look for Google function usage logs
                    if (
                        hasattr(record, "name")
                        and "function_google" in record.name
                        and "Token usage" in record.getMessage()
                    ):
                        try:
                            message = record.getMessage()
                            # Parse enhanced format: "Token usage - Prompt: 2, Completion: 9, Thinking: 5, Total: 59"
                            # Also handle old format: "Token usage - Prompt: 2, Completion: 9, Total: 59"

                            # Try enhanced format first (with thinking tokens)
                            match = re.search(
                                r"Prompt:\s*(\d+),\s*Completion:\s*(\d+),\s*Thinking:\s*(\d+),\s*Total:\s*(\d+)",
                                message,
                            )
                            if match:
                                prompt_tokens = int(match.group(1))
                                completion_tokens = int(match.group(2))
                                thinking_tokens = int(match.group(3))
                                total_tokens = int(match.group(4))

                                usage_data = {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "thinking_tokens": thinking_tokens,
                                    "total_tokens": total_tokens,
                                    "timestamp": time.time(),
                                    "source": "log_interceptor",
                                }
                            else:
                                # Fallback to old format (without thinking tokens)
                                match = re.search(
                                    r"Prompt:\s*(\d+),\s*Completion:\s*(\d+),\s*Total:\s*(\d+)",
                                    message,
                                )
                                if match:
                                    prompt_tokens = int(match.group(1))
                                    completion_tokens = int(match.group(2))
                                    total_tokens = int(match.group(3))

                                    usage_data = {
                                        "prompt_tokens": prompt_tokens,
                                        "completion_tokens": completion_tokens,
                                        "total_tokens": total_tokens,
                                        "timestamp": time.time(),
                                        "source": "log_interceptor",
                                    }
                                else:
                                    return  # Could not parse the log message

                            # Store with timestamp as key (recent usage)
                            timestamp_key = str(int(time.time() * 1000))
                            self.filter_instance._captured_usage[timestamp_key] = (
                                usage_data
                            )

                            if self.filter_instance.valves.enable_debug:
                                logger.info(f"Captured Google usage data: {usage_data}")

                        except Exception as e:
                            logger.error(f"Error parsing usage log: {e}")

            # Add the handler to the root logger
            usage_handler = UsageLogHandler(self)
            logging.getLogger().addHandler(usage_handler)

            if self.valves.enable_debug:
                logger.info("Log interceptor for usage data capture enabled")

        except Exception as e:
            logger.error(f"Failed to setup log interceptor: {e}")

    def _get_recent_captured_usage(self, max_age_seconds=30):
        """Get recently captured usage data from logs"""
        current_time = time.time()
        recent_usage = None

        # Find the most recent usage within max_age_seconds
        for timestamp_str, usage_data in list(self._captured_usage.items()):
            usage_time = usage_data.get("timestamp", 0)
            age = current_time - usage_time

            if age <= max_age_seconds:
                if recent_usage is None or usage_time > recent_usage.get(
                    "timestamp", 0
                ):
                    recent_usage = usage_data
            else:
                # Clean up old usage data
                del self._captured_usage[timestamp_str]

        return recent_usage

    def store_shared_usage(self, request_id: str, usage_data: dict):
        """Store usage data for sharing between functions"""
        self._shared_usage_data[request_id] = {**usage_data, "timestamp": time.time()}
        if self.valves.enable_debug:
            logger.info(f"Stored shared usage data for {request_id}: {usage_data}")

    def get_shared_usage(self, request_id: str) -> dict:
        """Get stored usage data"""
        return self._shared_usage_data.get(request_id)

    def count_tokens(self, messages: List[dict]) -> int:
        """Count tokens for a list of messages (includes full conversation history)."""
        if not self.tokenizer:
            logger.warning("Tokenizer not available, using character count estimation")
            total_chars = sum(
                len(str(message.get("content", ""))) for message in messages
            )
            return total_chars // 4  # Rough estimation: 1 token ≈ 4 characters

        token_count = 0
        try:
            for i, message in enumerate(messages):
                content = str(message.get("content", ""))
                role = message.get("role", "unknown")

                if content:
                    message_tokens = len(self.tokenizer.encode(content))
                    token_count += message_tokens

                    if self.valves.enable_debug:
                        logger.info(
                            f"Token count - Message {i+1}/{len(messages)}, Role: {role}, Content: '{content[:50]}...', Tokens: {message_tokens}"
                        )

            # Add system message overhead (OpenAI adds ~3 tokens per message for role formatting)
            if messages:
                formatting_overhead = len(messages) * 3
                token_count += formatting_overhead

                if self.valves.enable_debug:
                    logger.info(
                        f"Added {formatting_overhead} tokens for message formatting overhead"
                    )

        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback to character count
            total_chars = sum(
                len(str(message.get("content", ""))) for message in messages
            )
            return total_chars // 4

        if self.valves.enable_debug:
            logger.info(
                f"Total token count: {token_count} for {len(messages)} messages (full conversation history)"
            )

        return token_count

    def extract_username(self, user_info: dict) -> str:
        """Extract username from user info based on configuration."""
        if not user_info:
            return None

        method = self.valves.username_extraction_method

        if method == "user_id":
            return user_info.get("id")
        elif method == "email":
            return user_info.get("email")
        elif method == "username":
            return user_info.get("username") or user_info.get("name")
        else:
            # Fallback: try id first, then email, then username
            return (
                user_info.get("id")
                or user_info.get("email")
                or user_info.get("username")
                or user_info.get("name")
            )

    def check_user_tokens_sync(self, username: str) -> int:
        """Synchronous version of check_user_tokens"""
        try:
            import requests

            url = f"{self.valves.api_url}/pipeline/get_tokens"
            params = {"username": username}

            if self.valves.enable_debug:
                logger.info(f"Checking tokens for user {username} at {url} (sync)")

            response = requests.get(url, params=params, timeout=self.valves.api_timeout)

            if response.status_code == 200:
                result = response.json()
                tokens = int(result.get("tokens", 0))
                is_active = result.get("is_active", True)
                lookup_method = result.get("lookup_method", "unknown")

                if not is_active:
                    logger.warning(f"User {username} account is inactive")
                    return 0

                if self.valves.enable_debug:
                    logger.info(
                        f"SUCCESS: Found user via {lookup_method}, username={result.get('username')}, tokens={tokens}, active={is_active}"
                    )

                return tokens
            elif response.status_code == 404:
                logger.warning(f"User {username} not found in database")
                return 0
            else:
                logger.error(
                    f"API error for user {username}: {response.status_code} - {response.text}"
                )
                return 0

        except Exception as e:
            logger.error(f"Error checking user tokens (sync): {e}")
            return 0

    def record_platform_usage_sync(
        self, username: str, tokens: int, source: str = "openwebui"
    ) -> bool:
        """Synchronous version of record_platform_usage"""
        try:
            import requests

            url = f"{self.valves.api_url}/pipeline/use_tokens"
            payload = {
                "username": username,
                "tokens": tokens,
                "source": source,
                "allow_overdraft": self.valves.allow_overdraft,
                "description": f"OpenWebUI chat usage - {tokens} tokens (source: {source})",
            }

            if self.valves.enable_debug:
                overdraft_status = (
                    "overdraft allowed"
                    if self.valves.allow_overdraft
                    else "overdraft disabled"
                )
                logger.info(
                    f"Recording platform usage: {username} - {tokens} tokens ({overdraft_status}) (sync)"
                )

            response = requests.post(url, json=payload, timeout=self.valves.api_timeout)

            if response.status_code == 200:
                result = response.json()
                success = result.get("success", False)
                remaining = result.get("remaining_tokens", 0)
                lookup_method = result.get("lookup_method", "unknown")
                was_overdraft = result.get("was_overdraft", False)

                if self.valves.enable_debug:
                    overdraft_msg = " (OVERDRAFT)" if was_overdraft else ""
                    logger.info(
                        f"Platform usage recorded successfully for {username} via {lookup_method}: remaining={remaining} tokens{overdraft_msg}"
                    )

                return success

            elif response.status_code == 400:
                error_data = response.json()
                error_msg = error_data.get("error", "")
                if "Insufficient tokens" in error_msg:
                    if self.valves.allow_overdraft:
                        logger.warning(
                            f"Insufficient tokens for user {username} when recording usage (even with overdraft enabled): {error_msg}"
                        )
                    else:
                        logger.warning(
                            f"Insufficient tokens for user {username} when recording usage (overdraft disabled): {error_msg}"
                        )
                    return False

            logger.error(
                f"Failed to record platform usage for {username}: {response.status_code} - {response.text}"
            )
            return False

        except Exception as e:
            logger.error(f"Error recording platform usage for {username} (sync): {e}")
            return False

    def validate_conversation_handling(self, messages: List[dict]) -> Dict[str, Any]:
        """Validate that conversation history is being handled correctly"""
        validation_info = {
            "total_messages": len(messages),
            "conversation_turns": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "system_messages": 0,
            "total_content_length": 0,
            "estimated_tokens": 0,
        }

        for message in messages:
            role = message.get("role", "unknown")
            content = str(message.get("content", ""))

            if role == "user":
                validation_info["user_messages"] += 1
                validation_info["conversation_turns"] += 1
            elif role == "assistant":
                validation_info["assistant_messages"] += 1
            elif role == "system":
                validation_info["system_messages"] += 1

            validation_info["total_content_length"] += len(content)

        validation_info["estimated_tokens"] = self.count_tokens(messages)

        if self.valves.enable_debug:
            logger.info(f"Conversation validation: {validation_info}")

        return validation_info

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Filter to check token balance before sending request to LLM. (FIXED: Now synchronous)"""
        if self.valves.enable_debug:
            logger.info(f"Inlet filter called - Body keys: {list(body.keys())}")

        # Extract user information
        if not __user__:
            logger.warning("No user information provided, allowing request")
            if self.valves.enable_debug:
                body["__debug_info"] = (
                    "[DEBUG] ⚠️ No user information provided - token checking skipped"
                )
            return body

        username = self.extract_username(__user__)
        if not username:
            logger.warning(
                "Could not extract username from user info, allowing request"
            )
            if self.valves.enable_debug:
                debug_msg = f"[DEBUG] ⚠️ Username extraction failed. User info keys: {list(__user__.keys()) if __user__ else 'None'}, Method: {self.valves.username_extraction_method}"
                body["__debug_info"] = debug_msg
            return body

        # Count tokens in current request (includes full conversation history)
        current_messages = body.get("messages", [])

        if self.valves.enable_debug:
            # Validate conversation handling
            conversation_info = self.validate_conversation_handling(current_messages)
            logger.info(
                f"Processing conversation: {conversation_info['conversation_turns']} turns, {conversation_info['total_messages']} total messages"
            )

            # Log each message for transparency
            for i, msg in enumerate(current_messages):
                role = msg.get("role", "unknown")
                content_length = len(str(msg.get("content", "")))
                logger.info(
                    f"Message {i+1}/{len(current_messages)}: role={role}, length={content_length} chars"
                )

        request_tokens = self.count_tokens(current_messages)

        # Get current user token balance (FIXED: Using sync version)
        user_tokens = self.check_user_tokens_sync(username)

        # Check if user has any tokens
        if user_tokens <= 0:
            if self.valves.enable_debug:
                logger.info(f"User {username} has no tokens, blocking request")

            error_msg = self.valves.block_message
            if self.valves.enable_debug:
                error_msg += f"\n\n[DEBUG] User: {username}, Tokens: {user_tokens}, Request tokens: {request_tokens}, Status: BLOCKED - No tokens"

            raise Exception(error_msg)

        # Estimate total tokens needed (request + estimated response)
        estimated_total = request_tokens + self.valves.estimate_response_tokens

        if user_tokens < estimated_total:
            if self.valves.enable_debug:
                logger.info(
                    f"User {username} has insufficient tokens: {user_tokens} < {estimated_total}"
                )

            error_msg = f"{self.valves.block_message} (Need ~{estimated_total} tokens, have {user_tokens})"
            if self.valves.enable_debug:
                error_msg += f"\n\n[DEBUG] User: {username}, Available: {user_tokens}, Needed: {estimated_total}, Request: {request_tokens}, Estimated Response: {self.valves.estimate_response_tokens}, Status: BLOCKED - Insufficient tokens"

            raise Exception(error_msg)

        # Store token info for outlet using multiple persistence strategies
        request_id = self._get_request_id(body, __user__)

        token_data = {
            "username": username,
            "request_tokens": request_tokens,
            "user_tokens": user_tokens,
            "timestamp": time.time(),
            "estimated_total": estimated_total,
        }

        # Strategy 1: Store in class-level persistent storage
        self.active_requests[request_id] = token_data
        self.user_sessions[username] = token_data

        # Strategy 2: Store in body with multiple keys
        body["__token_data"] = token_data.copy()
        body["__user_tokens"] = token_data.copy()
        body["__request_id"] = request_id
        body["__username"] = username

        # Strategy 3: Store in metadata
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["__token_data"] = token_data.copy()
        body["metadata"]["__user_tokens"] = token_data.copy()
        body["metadata"]["__request_id"] = request_id

        # Strategy 4: Store in model-specific fields
        if "model" in body:
            body["model_metadata"] = body.get("model_metadata", {})
            body["model_metadata"]["__token_data"] = token_data.copy()

        if self.valves.enable_debug:
            logger.info(
                f"User {username} request approved: {request_tokens} tokens, {user_tokens} available"
            )
            logger.info(
                f"Stored token info with request_id: {request_id} using multiple strategies"
            )
            debug_info = f"[DEBUG] Token check passed - User: {username}, Available: {user_tokens}, Request: {request_tokens}, Estimated total: {estimated_total}"
            body["__debug_info"] = debug_info

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Filter to record token usage after receiving response from LLM."""
        if self.valves.enable_debug:
            logger.info(f"Outlet filter called - Body keys: {list(body.keys())}")
            logger.info(
                f"Full body structure for token detection: {json.dumps(body, indent=2, default=str)[:1000]}..."
            )

        # Try multiple strategies to recover token info
        token_info = None
        username = None
        recovery_method = None

        # Strategy 1: Direct from body
        if "__token_data" in body:
            token_info = body["__token_data"]
            recovery_method = "body.__token_data"
        elif "__user_tokens" in body:
            token_info = body["__user_tokens"]
            recovery_method = "body.__user_tokens"

        # Strategy 2: From metadata
        elif "metadata" in body and "__token_data" in body["metadata"]:
            token_info = body["metadata"]["__token_data"]
            recovery_method = "metadata.__token_data"
        elif "metadata" in body and "__user_tokens" in body["metadata"]:
            token_info = body["metadata"]["__user_tokens"]
            recovery_method = "metadata.__user_tokens"

        # Strategy 3: From model metadata
        elif "model_metadata" in body and "__token_data" in body["model_metadata"]:
            token_info = body["model_metadata"]["__token_data"]
            recovery_method = "model_metadata.__token_data"

        # Strategy 4: From persistent storage using request_id
        elif "__request_id" in body and body["__request_id"] in self.active_requests:
            token_info = self.active_requests[body["__request_id"]]
            recovery_method = "active_requests[request_id]"

        # Strategy 5: From persistent storage using username
        elif "__username" in body and body["__username"] in self.user_sessions:
            token_info = self.user_sessions[body["__username"]]
            recovery_method = "user_sessions[username]"

        # Strategy 6: Try to extract username and lookup in user_sessions
        elif __user__:
            username = self.extract_username(__user__)
            if username and username in self.user_sessions:
                token_info = self.user_sessions[username]
                recovery_method = "user_sessions[extracted_username]"

        if not token_info:
            if self.valves.enable_debug:
                logger.warning(
                    f"No token info found using any strategy, skipping token deduction"
                )
                available_keys = list(body.keys())
                metadata_keys = (
                    list(body.get("metadata", {}).keys()) if "metadata" in body else []
                )
                debug_msg = f"[DEBUG] ℹ️ No token deduction - Available body keys: {available_keys}, metadata keys: {metadata_keys}"

                if "content" in body:
                    if "__debug_info" in body:
                        body["content"] += f"\n\n{body['__debug_info']} | {debug_msg}"
                        del body["__debug_info"]
                    else:
                        body["content"] += f"\n\n{debug_msg}"
            return body

        if self.valves.enable_debug:
            logger.info(f"Token info recovered using: {recovery_method}")

        username = token_info["username"]
        request_tokens = token_info["request_tokens"]

        # ENHANCED TOKEN DETECTION STRATEGIES
        actual_total_tokens = None
        actual_prompt_tokens = None
        actual_completion_tokens = None
        actual_thinking_tokens = None  # Add thinking tokens variable
        token_source = "estimated"

        # Strategy 1: Look for usage data in HTML comments
        response_content = body.get("content", "")
        if isinstance(response_content, str) and "<!--USAGE:" in response_content:
            try:
                match = re.search(r"<!--USAGE:(\{[^}]+\})-->", response_content)
                if match:
                    usage_json = match.group(1)
                    usage_data = json.loads(usage_json)

                    actual_total_tokens = usage_data.get("total_tokens")
                    actual_prompt_tokens = usage_data.get("prompt_tokens")
                    actual_completion_tokens = usage_data.get("completion_tokens")
                    actual_thinking_tokens = usage_data.get("thinking_tokens")
                    token_source = "pipeline_comment"

                    body["content"] = re.sub(
                        r"<!--USAGE:\{[^}]+\}-->", "", response_content
                    )

                    if self.valves.enable_debug:
                        logger.info(
                            f"Extracted token usage from pipeline comment: {usage_data}"
                        )
            except Exception as e:
                logger.warning(f"Failed to parse usage from pipeline comment: {e}")

        # Strategy 2: Enhanced search in standard locations
        if actual_total_tokens is None:
            token_usage_locations = [
                # Standard locations
                body.get("usage"),
                body.get("token_usage"),
                body.get("model_usage"),
                body.get("tokens"),
                # Metadata locations
                body.get("metadata", {}).get("usage"),
                body.get("metadata", {}).get("token_usage"),
                body.get("metadata", {}).get("model_usage"),
                # Info locations
                body.get("info", {}).get("usage") if body.get("info") else None,
                body.get("info", {}).get("token_usage") if body.get("info") else None,
                # Response-specific locations
                body.get("response", {}).get("usage") if body.get("response") else None,
                (
                    body.get("response", {}).get("token_usage")
                    if body.get("response")
                    else None
                ),
                # Model response locations
                (
                    body.get("model_response", {}).get("usage")
                    if body.get("model_response")
                    else None
                ),
                (
                    body.get("completion", {}).get("usage")
                    if body.get("completion")
                    else None
                ),
                # Google-specific locations
                body.get("google_usage"),
                (
                    body.get("generation_info", {}).get("usage")
                    if body.get("generation_info")
                    else None
                ),
            ]

            for usage_data in token_usage_locations:
                if usage_data and isinstance(usage_data, dict):
                    total = (
                        usage_data.get("total_tokens")
                        or usage_data.get("total")
                        or usage_data.get("totalTokens")
                        or usage_data.get("total_token_count")
                    )

                    prompt = (
                        usage_data.get("prompt_tokens")
                        or usage_data.get("input_tokens")
                        or usage_data.get("promptTokens")
                        or usage_data.get("input")
                        or usage_data.get("prompt_token_count")
                    )

                    completion = (
                        usage_data.get("completion_tokens")
                        or usage_data.get("output_tokens")
                        or usage_data.get("completionTokens")
                        or usage_data.get("output")
                        or usage_data.get("completion_token_count")
                        or usage_data.get("generated_tokens")
                    )

                    thinking = (
                        usage_data.get("thinking_tokens")
                        or usage_data.get("reasoning_tokens")
                        or usage_data.get("cached_tokens")
                    )

                    if total is not None:
                        actual_total_tokens = int(total)
                        token_source = "model_reported_total"
                        if prompt is not None:
                            actual_prompt_tokens = int(prompt)
                        if completion is not None:
                            actual_completion_tokens = int(completion)
                        if thinking is not None:
                            actual_thinking_tokens = int(thinking)
                        break
                    elif prompt is not None and completion is not None:
                        actual_prompt_tokens = int(prompt)
                        actual_completion_tokens = int(completion)
                        if thinking is not None:
                            actual_thinking_tokens = int(thinking)
                            actual_total_tokens = (
                                actual_prompt_tokens
                                + actual_completion_tokens
                                + actual_thinking_tokens
                            )
                        else:
                            actual_total_tokens = (
                                actual_prompt_tokens + actual_completion_tokens
                            )
                        token_source = "model_reported_split"
                        break

        # Strategy 3: Broad search through all body keys
        if actual_total_tokens is None:
            if self.valves.enable_debug:
                logger.info("Attempting broad search for token usage in all body keys")

            def search_nested_dict(obj, path=""):
                """Recursively search for token usage data"""
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key

                        # Look for keys that might contain usage data
                        if any(
                            keyword in key.lower()
                            for keyword in ["usage", "token", "count"]
                        ):
                            if isinstance(value, dict):
                                # Check if this looks like usage data
                                if any(
                                    field in value
                                    for field in [
                                        "total_tokens",
                                        "prompt_tokens",
                                        "completion_tokens",
                                        "total",
                                        "input",
                                        "output",
                                    ]
                                ):
                                    if self.valves.enable_debug:
                                        logger.info(
                                            f"Found potential usage data at {current_path}: {value}"
                                        )
                                    return value
                            elif (
                                isinstance(value, (int, float))
                                and "total" in key.lower()
                            ):
                                if self.valves.enable_debug:
                                    logger.info(
                                        f"Found potential total token count at {current_path}: {value}"
                                    )
                                return {"total_tokens": value}

                        # Recurse into nested objects
                        if isinstance(value, dict):
                            result = search_nested_dict(value, current_path)
                            if result:
                                return result
                return None

            usage_data = search_nested_dict(body)
            if usage_data:
                total = (
                    usage_data.get("total_tokens")
                    or usage_data.get("total")
                    or usage_data.get("totalTokens")
                )
                prompt = (
                    usage_data.get("prompt_tokens")
                    or usage_data.get("input_tokens")
                    or usage_data.get("input")
                )
                completion = (
                    usage_data.get("completion_tokens")
                    or usage_data.get("output_tokens")
                    or usage_data.get("output")
                )
                thinking = usage_data.get("thinking_tokens") or usage_data.get(
                    "reasoning_tokens"
                )

                if total is not None:
                    actual_total_tokens = int(total)
                    token_source = "broad_search_total"
                    if prompt is not None:
                        actual_prompt_tokens = int(prompt)
                    if completion is not None:
                        actual_completion_tokens = int(completion)
                    if thinking is not None:
                        actual_thinking_tokens = int(thinking)
                elif prompt is not None and completion is not None:
                    actual_prompt_tokens = int(prompt)
                    actual_completion_tokens = int(completion)
                    if thinking is not None:
                        actual_thinking_tokens = int(thinking)
                        actual_total_tokens = (
                            actual_prompt_tokens
                            + actual_completion_tokens
                            + actual_thinking_tokens
                        )
                    else:
                        actual_total_tokens = (
                            actual_prompt_tokens + actual_completion_tokens
                        )
                    token_source = "broad_search_split"

        # Strategy 4: Check shared usage storage
        if actual_total_tokens is None:
            request_id = body.get("__request_id") or self._get_request_id(
                body, __user__
            )

            # Try different request ID variations
            possible_ids = [
                request_id,
                body.get("id"),
                body.get("chat_id"),
                body.get("session_id"),
                username,
            ]

            for rid in possible_ids:
                if rid and rid in self._shared_usage_data:
                    shared_data = self._shared_usage_data[rid]
                    actual_total_tokens = shared_data.get("total_tokens")
                    actual_prompt_tokens = shared_data.get("prompt_tokens")
                    actual_completion_tokens = shared_data.get("completion_tokens")
                    actual_thinking_tokens = shared_data.get("thinking_tokens")
                    token_source = "shared_storage"

                    if self.valves.enable_debug:
                        logger.info(
                            f"Found usage data in shared storage via {rid}: {shared_data}"
                        )

                    # Clean up old data
                    del self._shared_usage_data[rid]
                    break

        # Strategy 5: Check captured usage from logs
        if actual_total_tokens is None:
            captured_usage = self._get_recent_captured_usage(max_age_seconds=30)
            if captured_usage:
                actual_total_tokens = captured_usage.get("total_tokens")
                actual_prompt_tokens = captured_usage.get("prompt_tokens")
                actual_completion_tokens = captured_usage.get("completion_tokens")
                actual_thinking_tokens = captured_usage.get("thinking_tokens")
                token_source = "log_interceptor"

                if self.valves.enable_debug:
                    logger.info(f"Using captured usage data: {captured_usage}")

        # Strategy 6: Look for token info in response headers
        if actual_total_tokens is None:
            headers = body.get("headers", {})
            if isinstance(headers, dict):
                for key, value in headers.items():
                    if "token" in key.lower() and isinstance(value, (int, str)):
                        try:
                            if "total" in key.lower():
                                actual_total_tokens = int(value)
                                token_source = f"header_{key}"
                                break
                        except (ValueError, TypeError):
                            continue

        # Fallback: Count tokens manually if no model data available
        if actual_total_tokens is None:
            response_content = body.get("content", "")
            response_tokens = self.count_tokens([{"content": response_content}])
            actual_total_tokens = request_tokens + response_tokens
            actual_prompt_tokens = request_tokens
            actual_completion_tokens = response_tokens
            token_source = "manual_count"

            if self.valves.enable_debug:
                logger.info(
                    f"No model token usage found, using manual count: prompt={actual_prompt_tokens}, response={actual_completion_tokens}, total={actual_total_tokens}"
                )
        else:
            if self.valves.enable_debug:
                # Create detailed token breakdown for logging
                breakdown_parts = []
                if actual_prompt_tokens is not None:
                    breakdown_parts.append(f"prompt={actual_prompt_tokens}")
                if actual_completion_tokens is not None:
                    breakdown_parts.append(f"completion={actual_completion_tokens}")
                if actual_thinking_tokens is not None:
                    breakdown_parts.append(f"thinking={actual_thinking_tokens}")

                breakdown_str = (
                    ", ".join(breakdown_parts)
                    if breakdown_parts
                    else "breakdown unavailable"
                )

                logger.info(
                    f"Using model-reported token usage via {token_source}: total={actual_total_tokens} ({breakdown_str})"
                )

        # Use the actual total tokens for recording usage
        total_tokens = actual_total_tokens

        # Record usage in platform (use sync version)
        success = self.record_platform_usage_sync(username, total_tokens, token_source)

        # Clean up persistent storage
        if "__request_id" in body and body["__request_id"] in self.active_requests:
            del self.active_requests[body["__request_id"]]

        # Add debug info to response
        if self.valves.enable_debug:
            debug_parts = []

            # Add inlet debug info if available
            if "__debug_info" in body:
                debug_parts.append(body["__debug_info"])
                del body["__debug_info"]

            if success:
                # Show detailed token breakdown based on source
                if token_source == "manual_count":
                    debug_parts.append(
                        f"✅ Token usage: User: {username}, Request: {actual_prompt_tokens}, Response: {actual_completion_tokens}, Total recorded: {total_tokens} (manual count, via {recovery_method})"
                    )
                else:
                    # Create detailed breakdown including thinking tokens
                    breakdown_parts = []
                    if actual_prompt_tokens is not None:
                        breakdown_parts.append(f"prompt={actual_prompt_tokens}")
                    if actual_completion_tokens is not None:
                        breakdown_parts.append(f"completion={actual_completion_tokens}")
                    if actual_thinking_tokens is not None:
                        breakdown_parts.append(f"thinking={actual_thinking_tokens}")

                    breakdown_str = (
                        f" ({', '.join(breakdown_parts)})" if breakdown_parts else ""
                    )

                    debug_parts.append(
                        f"✅ Token usage: User: {username}, Total: {total_tokens}{breakdown_str}, Source: {token_source} (via {recovery_method})"
                    )
                logger.info(
                    f"Successfully recorded {total_tokens} tokens for user {username} using {token_source}"
                )
            else:
                # More detailed failure reporting
                overdraft_info = ""
                if self.valves.allow_overdraft:
                    overdraft_info = " (overdraft enabled)"
                else:
                    overdraft_info = " (overdraft disabled)"

                debug_parts.append(
                    f"❌ Token recording FAILED{overdraft_info}: User: {username}, Attempted: {total_tokens}, Source: {token_source} (via {recovery_method})"
                )
                logger.error(
                    f"Failed to record {total_tokens} tokens for user {username} using {token_source}{overdraft_info}"
                )

                # If overdraft is disabled and this might be the reason, suggest enabling it
                if not self.valves.allow_overdraft:
                    debug_parts.append(
                        f"💡 Tip: Enable 'allow_overdraft' if actual usage exceeds estimated tokens"
                    )

            if "content" in body:
                body["content"] += f"\n\n[DEBUG] {' | '.join(debug_parts)}"

        # Clean up all token info from body
        cleanup_keys = [
            "__token_data",
            "__user_tokens",
            "__debug_info",
            "__request_id",
            "__username",
        ]
        for key in cleanup_keys:
            if key in body:
                del body[key]

        return body
