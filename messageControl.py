"""
title: Personal Information Filter Pipeline
author: modified from open-webui
date: 2024-05-30
version: 1.3
license: MIT
description: Filter pipeline that prevents sending personal information to the OpenAI API with user notifications
requirements: requests, re
"""
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import re
import os

class PersonalInfoFilter:
    """Helper class to detect and handle personal information."""
    
    def __init__(self):
        # Regex patterns for different types of personal information
        self.patterns: Dict[str, str] = {
            'dni': r'[0-9]{8}[A-Z]',  # Spanish DNI format
            'nie': r'[XYZ][0-9]{7}[A-Z]',  # Spanish NIE format
            'passport': r'[A-Z]{2}[0-9]{6}',  # Generic passport format
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Generic phone format
            'credit_card': r'\b\d{4}[-. ]?\d{4}[-. ]?\d{4}[-. ]?\d{4}\b',
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',  # US Social Security Number
        }
        
        # Spanish error messages for each type
        self.error_messages: Dict[str, str] = {
            'dni': "Se ha detectado un DNI en su mensaje. Por razones de seguridad, no se permite enviar información personal.",
            'nie': "Se ha detectado un NIE en su mensaje. Por razones de seguridad, no se permite enviar información personal.",
            'passport': "Se ha detectado un número de pasaporte en su mensaje. Por razones de seguridad, no se permite enviar información personal.",
            'email': "Se ha detectado una dirección de correo electrónico en su mensaje. Por razones de seguridad, no se permite enviar información personal.",
            'phone': "Se ha detectado un número de teléfono en su mensaje. Por razones de seguridad, no se permite enviar información personal.",
            'credit_card': "Se ha detectado un número de tarjeta de crédito en su mensaje. Por razones de seguridad, no se permite enviar información personal.",
            'ssn': "Se ha detectado un número de seguridad social en su mensaje. Por razones de seguridad, no se permite enviar información personal."
        }

    def check_personal_info(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if text contains personal information and return type if found."""
        for info_type, pattern in self.patterns.items():
            if re.search(pattern, text):
                return True, info_type
        return False, None

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        enable_logging: bool = False
        strict_mode: bool = True

    class SpanishErrors:
        GENERAL_ERROR = "Se ha detectado información personal sensible en su mensaje. Por favor, revise y elimine cualquier dato personal antes de enviarlo."

    def __init__(self):
        self.type = "filter"
        self.name = "Personal Information Filter"
        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("PERSONAL_INFO_PIPELINES", "*").split(","),
                "enable_logging": bool(os.getenv("PERSONAL_INFO_LOGGING", False)),
                "strict_mode": bool(os.getenv("PERSONAL_INFO_STRICT_MODE", True))
            }
        )
        self.personal_info_filter = PersonalInfoFilter()

    async def on_startup(self):
        print(f"Starting Personal Information Filter Pipeline")
        pass

    async def on_shutdown(self):
        print(f"Shutting down Personal Information Filter Pipeline")
        pass

    def _check_messages(self, messages: List[OpenAIChatMessage]) -> Tuple[bool, Optional[str]]:
        """Check messages for personal information."""
        for message in messages:
            if message.content:
                has_personal_info, info_type = self.personal_info_filter.check_personal_info(message.content)
                if has_personal_info:
                    return True, info_type
        return False, None

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Filter personal information from the request body."""
        print(f"Processing request through Personal Information Filter")

        if user and user.get("role", "admin") == "user":
            # Check messages for personal information
            if "messages" in body:
                has_personal_info, info_type = self._check_messages(body["messages"])
                if has_personal_info:
                    error_message = (
                        self.personal_info_filter.error_messages.get(info_type, 
                        self.SpanishErrors.GENERAL_ERROR)
                    )
                    raise Exception(error_message)

            # Check prompt for personal information
            if "prompt" in body and isinstance(body["prompt"], str):
                has_personal_info, info_type = self.personal_info_filter.check_personal_info(body["prompt"])
                if has_personal_info:
                    error_message = (
                        self.personal_info_filter.error_messages.get(info_type, 
                        self.SpanishErrors.GENERAL_ERROR)
                    )
                    raise Exception(error_message)

            if self.valves.enable_logging:
                print(f"Processed request body: {body}")

        return body
