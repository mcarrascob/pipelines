import os
from typing import List, Optional, Dict
from pydantic import BaseModel
import logging
import json
import re

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        target_user_roles: List[str] = ["user"]

    class SpanishErrors:
        FILE_DETECTED = "No se permiten cargas de archivos en este chat. Por favor, comparta el contenido como texto."
        URL_WARNING = "Las URLs no funcionarán ya que el sistema no está conectado a Internet. Por favor, comparta el contenido relevante como texto."
        FILE_IN_FUNCTION = "No se permiten funciones que manipulen archivos reales. Las funciones de ejemplo con archivos en el código están permitidas."
        DOCUMENT_DETECTED = "No se permite la carga de documentos (PDF, Excel, Word, etc.). Por favor, comparta el contenido relevante como texto."

    def __init__(self):
        self.type = "filter"
        self.name = "File Upload Control Filter"
        
        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("FILE_CONTROL_PIPELINES", "*").split(","),
            }
        )
        self.logger = logging.getLogger(__name__)

    def is_code_block(self, text: str) -> bool:
        """Check if the content is within a code block or is code-related."""
        # Check for markdown code blocks (both ``` and ~~~)
        if re.search(r'```[\s\S]*?```|~~~[\s\S]*?~~~', text):
            return True
            
        # Check for HTML code tags
        if re.search(r'<code>[\s\S]*?</code>', text):
            return True
            
        # Check for HTML elements that indicate code
        if re.search(r'<script>|<style>|<html>|<!DOCTYPE|<head>|<body>|</html>|</script>|</style>', text):
            return True
            
        # Check for common programming patterns
        code_patterns = [
            r'function\s+\w+\s*\(',  # Function declarations
            r'const\s+\w+\s*=',      # Const declarations
            r'let\s+\w+\s*=',        # Let declarations
            r'var\s+\w+\s*=',        # Var declarations
            r'class\s+\w+',          # Class declarations
            r'import\s+.*from',      # Import statements
            r'export\s+default',     # Export statements
            r'{\s*[\w\s,:"\']+\s*}', # Object literals
            r'#\w+\s*{',            # CSS selectors
            r'\.\w+\s*{',           # CSS class selectors
            r'@media',              # CSS media queries
            r'<\w+>[\s\S]*</\w+>'   # HTML tags
        ]
        
        return any(re.search(pattern, text) for pattern in code_patterns)

    def check_for_file_content(self, content: str) -> tuple[bool, str]:
        """
        Check if the content contains actual file upload attempts.
        Returns (is_file_content, message).
        """
        # First check if it's code
        if self.is_code_block(content):
            return False, ""

        # Document and file extensions to block (only when they appear as actual files)
        blocked_extensions = [
            # Documents
            ".pdf", ".doc", ".docx", ".rtf", ".odt",
            # Spreadsheets
            ".xls", ".xlsx", ".csv", ".ods",
            # Presentations
            ".ppt", ".pptx", ".odp",
            # Archives
            ".zip", ".rar", ".7z", ".tar", ".gz",
            # Database files
            ".db", ".sqlite", ".sql",
            # Others
            ".exe", ".bin", ".dat"
        ]

        # File upload and handling patterns that indicate actual file uploads
        file_patterns = [
            # Data URIs
            "base64,",
            "data:application/",
            
            # File input patterns
            "input type=\"file\"",
            "accept=\"application/",
            "enctype=\"multipart/form-data\"",
            
            # File API patterns
            "new FileReader(",
            "new File([",
            "new Blob([",
            
            # Binary data patterns
            "%PDF-",  # PDF header
            "PK\x03\x04",  # ZIP header
        ]
        
        # Check for blocked file extensions in a way that indicates actual files
        for ext in blocked_extensions:
            # Look for patterns that suggest actual file uploads, not just mentions
            file_patterns = [
                f"filename=\"*{ext}\"",
                f"uploadFile*{ext}",
                f"download*{ext}",
                f"archivo*{ext}",
            ]
            for pattern in file_patterns:
                if pattern.lower() in content.lower():
                    return True, self.SpanishErrors.DOCUMENT_DETECTED

        # Check for file upload patterns
        for pattern in file_patterns:
            if pattern in content:
                return True, self.SpanishErrors.FILE_DETECTED

        # Check for URLs
        url_patterns = ["http://", "https://"]
        for pattern in url_patterns:
            if pattern in content:
                return False, self.SpanishErrors.URL_WARNING

        return False, ""

    def check_for_file_functions(self, content: str) -> tuple[bool, str]:
        """
        Check if the content contains actual file manipulation functions.
        Returns (has_file_functions, message).
        """
        if self.is_code_block(content):
            return False, ""

        # Only check for actual file manipulation functions, not code examples
        file_functions = [
            # Actual file operations
            "new FileReader(",
            "createObjectURL(",
            "uploadFile(",
            "downloadFile(",
            "saveAs(",
            "readAsDataURL(",
            "readAsBinaryString(",
        ]
        
        for func in file_functions:
            # Look for patterns that suggest actual file operations
            if func in content and not self.is_code_block(content):
                return True, self.SpanishErrors.FILE_IN_FUNCTION

        return False, ""

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")
        print(f"Received body: {body}")
        print(f"User: {user}")

        if user and user.get("role") in self.valves.target_user_roles:
            warnings = set()
            try:
                # Check messages for file content
                if "messages" in body:
                    for message in body["messages"]:
                        if "content" in message and message["content"]:
                            content = str(message["content"])

                            # Check for file content
                            has_file, msg = self.check_for_file_content(content)
                            if has_file:
                                raise Exception(msg)
                            elif msg:  # URL warning
                                warnings.add(msg)

                            # Check for file manipulation functions
                            has_func, msg = self.check_for_file_functions(content)
                            if has_func:
                                raise Exception(msg)

                # Check function calls if present
                if "function_call" in body:
                    function_content = json.dumps(body["function_call"])
                    has_func, msg = self.check_for_file_functions(function_content)
                    if has_func:
                        raise Exception(msg)

                # If we have warnings but no blocking issues, add them to the response
                if warnings:
                    warning_msg = "\n".join(warnings)
                    if "messages" in body:
                        system_msg = {
                            "role": "system",
                            "content": warning_msg
                        }
                        body["messages"].insert(0, system_msg)

            except Exception as e:
                self.logger.error(f"Error processing request: {str(e)}")
                raise

        return body

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
