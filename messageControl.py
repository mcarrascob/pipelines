"""
title: Personal Information Filter Pipeline
author: modified from open-webui
date: 2024-05-30
version: 1.4
license: MIT
description: Filter pipeline that prevents sending personal information to the OpenAI API with user notifications
requirements: requests, re
"""
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel
import re
import os

class PersonalInfoFilter:
    """Helper class to detect and handle international personal information."""
    
    def __init__(self):
        # Regex patterns for different types of personal information
        self.patterns: Dict[str, str] = {
            # National IDs
            'dni': r'[0-9]{8}[A-Z]',  # Spanish DNI
            'nie': r'[XYZ][0-9]{7}[A-Z]',  # Spanish NIE
            'nif': r'[KLMXYZ][0-9]{7}[A-Z]',  # Spanish NIF
            'nss': r'[0-9]{2}[0-9]{10}',  # Spanish Social Security (NSS)
            'ssn_us': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',  # US SSN
            'ssn_uk': r'\b[A-Z]{2}[0-9]{6}[A-Z]\b',  # UK National Insurance Number
            'ssn_fr': r'\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b',  # French Social Security
            'ssn_de': r'\b[0-9]{2}[0-9]{10}\b',  # German Social Security
            'ssn_it': r'\b[A-Z]{3}[0-9]{2}[A-Z]{2}[0-9]{4}[A-Z]\b',  # Italian Fiscal Code
            
            # International Passports
            'passport_general': r'\b[A-Z]{1,3}[0-9]{6,9}\b',  # Generic international passport
            'passport_eu': r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{5}\b',  # EU passport format
            
            # Phone Numbers
            'phone_international': r'''(?x)
                \b
                (?:
                    # International format with optional country code
                    (?:\+|00)[1-9]\d{0,3}[\s.-]?
                    # Area code
                    (?:\([0-9]{1,4}\)|\d{1,4})
                    [\s.-]?
                    # Local number with flexible grouping
                    (?:\d{2,4}[\s.-]?){2,4}
                    |
                    # National format
                    (?:0\d{1,3}[\s.-]?)?
                    \d{2,4}[\s.-]?
                    (?:\d{2,4}[\s.-]?){1,3}
                )
                \b
            ''',
            
            # Email and Credit Cards (enhanced patterns)
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'credit_card': r'\b(?:\d[ -]*?){13,16}\b',  # Matches most credit card formats
            
            # Additional IDs
            'drivers_license': r'''(?x)
                \b
                (?:
                    # European format
                    [A-Z]{1,3}[-.]?\d{5,8}[-.]?[A-Z]{0,2}
                    |
                    # US format
                    [A-Z]\d{3}[-]\d{3}[-]\d{3}[-]\d{3}
                    |
                    # Generic format
                    [A-Z0-9]{5,20}
                )
                \b
            ''',
            
            # Tax IDs
            'tax_id': r'''(?x)
                \b
                (?:
                    # VAT numbers (EU)
                    [A-Z]{2}\d{8,12}
                    |
                    # Tax ID general format
                    [A-Z0-9]{6,20}
                )
                \b
            '''
        }
        
        # Error messages for each type
        self.error_messages: Dict[str, str] = {
            # National IDs
            'dni': "Se ha detectado un DNI. Por razones de seguridad, no se permite enviar información personal.",
            'nie': "Se ha detectado un NIE. Por razones de seguridad, no se permite enviar información personal.",
            'nif': "Se ha detectado un NIF. Por razones de seguridad, no se permite enviar información personal.",
            'nss': "Se ha detectado un número de Seguridad Social español. Por razones de seguridad, no se permite enviar información personal.",
            'ssn_us': "Se ha detectado un número de Seguro Social estadounidense. Por razones de seguridad, no se permite enviar información personal.",
            'ssn_uk': "Se ha detectado un número de Seguro Nacional británico. Por razones de seguridad, no se permite enviar información personal.",
            'ssn_fr': "Se ha detectado un número de Seguridad Social francés. Por razones de seguridad, no se permite enviar información personal.",
            'ssn_de': "Se ha detectado un número de Seguridad Social alemán. Por razones de seguridad, no se permite enviar información personal.",
            'ssn_it': "Se ha detectado un código fiscal italiano. Por razones de seguridad, no se permite enviar información personal.",
            
            # Passports
            'passport_general': "Se ha detectado un número de pasaporte. Por razones de seguridad, no se permite enviar información personal.",
            'passport_eu': "Se ha detectado un número de pasaporte europeo. Por razones de seguridad, no se permite enviar información personal.",
            
            # Other identifiers
            'phone_international': "Se ha detectado un número de teléfono. Por razones de seguridad, no se permite enviar información personal.",
            'email': "Se ha detectado una dirección de correo electrónico. Por razones de seguridad, no se permite enviar información personal.",
            'credit_card': "Se ha detectado un número de tarjeta de crédito. Por razones de seguridad, no se permite enviar información personal.",
            'drivers_license': "Se ha detectado un número de licencia de conducir. Por razones de seguridad, no se permite enviar información personal.",
            'tax_id': "Se ha detectado un número de identificación fiscal. Por razones de seguridad, no se permite enviar información personal."
        }

    def check_personal_info(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if text contains personal information and return type if found."""
        if not isinstance(text, str):
            return False, None
            
        # Normalize text to improve pattern matching
        normalized_text = text.upper()  # Convert to uppercase for consistent matching
        normalized_text = re.sub(r'\s+', ' ', normalized_text)  # Normalize whitespace
        
        for info_type, pattern in self.patterns.items():
            if re.search(pattern, normalized_text, re.VERBOSE):
                return True, info_type
        return False, None

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        enable_logging: bool = False
        strict_mode: bool = True
        admin_check: bool = True

    class Errors:
        GENERAL_ERROR = "Se ha detectado información personal sensible en su mensaje. Por favor, revise y elimine cualquier dato personal antes de enviarlo."
        INVALID_MESSAGE = "El formato del mensaje no es válido. Por favor, asegúrese de que el mensaje contiene texto válido."

    def __init__(self):
        self.type = "filter"
        self.name = "Enhanced Personal Information Filter"
        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("PERSONAL_INFO_PIPELINES", "*").split(","),
                "enable_logging": bool(os.getenv("PERSONAL_INFO_LOGGING", False)),
                "strict_mode": bool(os.getenv("PERSONAL_INFO_STRICT_MODE", True)),
                "admin_check": bool(os.getenv("PERSONAL_INFO_ADMIN_CHECK", True))
            }
        )
        self.personal_info_filter = PersonalInfoFilter()

    async def on_startup(self):
        print(f"Starting Personal Information Filter Pipeline")
        pass

    async def on_shutdown(self):
        print(f"Shutting down Personal Information Filter Pipeline")
        pass

    def _check_message_content(self, message: dict) -> Tuple[bool, Optional[str]]:
        """Check a single message for personal information."""
        if not isinstance(message, dict):
            return False, None

        content = message.get('content', '')
        if content:
            return self.personal_info_filter.check_personal_info(content)
        return False, None

    def _check_messages(self, messages: List[dict]) -> Tuple[bool, Optional[str]]:
        """Check messages for personal information."""
        if not isinstance(messages, list):
            return False, None

        for message in messages:
            has_personal_info, info_type = self._check_message_content(message)
            if has_personal_info:
                return True, info_type
        return False, None

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Filter personal information from the request body."""
        print(f"Processing request through Personal Information Filter")

        # Skip checks if user is admin and admin_check is False
        if not self.valves.admin_check and user and user.get("role") == "admin":
            return body

        try:
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
            if "prompt" in body:
                has_personal_info, info_type = self.personal_info_filter.check_personal_info(str(body["prompt"]))
                if has_personal_info:
                    error_message = (
                        self.personal_info_filter.error_messages.get(info_type, 
                        self.SpanishErrors.GENERAL_ERROR)
                    )
                    raise Exception(error_message)

            if self.valves.enable_logging:
                print(f"Processed request body: {body}")

            return body

        except Exception as e:
            if self.valves.enable_logging:
                print(f"Error processing request: {str(e)}")
            raise Exception(str(e))

    def log_message(self, message: str):
        """Log a message if logging is enabled."""
        if self.valves.enable_logging:
            print(f"Personal Info Filter: {message}")
