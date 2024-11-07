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
        """Check if the content is within a code block."""
        # Check for markdown code blocks (both ``` and ~~~)
        markdown_code = bool(re.search(r'```[\s\S]*?```|~~~[\s\S]*?~~~', text))
        # Check for HTML code tags
        html_code = bool(re.search(r'<code>[\s\S]*?</code>', text))
        # Check for common code file extensions
        file_extension = bool(re.search(r'\.(js|html|css|jsx|tsx|vue|svelte)$', text))
        return markdown_code or html_code or file_extension

    def check_for_file_content(self, content: str) -> tuple[bool, str]:
        """
        Check if the content contains actual file upload attempts.
        Returns (is_file_content, message).
        """
        if self.is_code_block(content):
            return False, ""

        # Document and file extensions to block
        blocked_extensions = [
            # Documents
            ".pdf", ".doc", ".docx", ".rtf", ".txt", ".odt",
            # Spreadsheets
            ".xls", ".xlsx", ".csv", ".ods",
            # Presentations
            ".ppt", ".pptx", ".odp",
            # Images
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".ico",
            # Archives
            ".zip", ".rar", ".7z", ".tar", ".gz",
            # Data files
            ".json", ".xml", ".yaml", ".yml", ".ini", ".config",
            # Database files
            ".db", ".sqlite", ".sql",
            # Other common files
            ".md", ".markdown", ".log", ".dat"
        ]

        # File upload and handling patterns
        file_patterns = [
            # Data URIs
            "base64,",
            "data:image/",
            "data:application/",
            "data:text/",
            
            # File input and form patterns
            "input type=\"file\"",
            "accept=\"",
            "enctype=\"multipart",
            "multipart/form-data",
            
            # File API patterns
            "FileReader(",
            "new Blob(",
            "new File(",
            "FormData(",
            "createObjectURL(",
            
            # Common file handling terms
            "upload",
            "download",
            "archivo adjunto",
            "adjuntar archivo",
            "subir archivo",
            "cargar archivo",
            
            # Document-specific patterns
            "application/pdf",
            "application/msword",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats",
            "text/plain",
            "spreadsheet",
            "workbook",
            "document",
            
            # Binary data patterns
            "%PDF-",  # PDF header
            "PK\x03\x04",  # ZIP header
            "\xFF\xD8\xFF",  # JPEG header
            "\x89PNG",  # PNG header
        ]
        
        # Check for blocked file extensions
        for ext in blocked_extensions:
            if ext.lower() in content.lower():
                if "." + ext.lower() in content.lower():  # Make sure it's actually an extension
                    return True, self.SpanishErrors.DOCUMENT_DETECTED

        # Check for file upload patterns
        for pattern in file_patterns:
            if pattern.lower() in content.lower():
                return True, self.SpanishErrors.FILE_DETECTED

        # Check for URLs but don't block them
        url_patterns = ["http://", "https://"]
        for pattern in url_patterns:
            if pattern.lower() in content.lower():
                return False, self.SpanishErrors.URL_WARNING

        return False, ""

    def check_for_file_functions(self, content: str) -> tuple[bool, str]:
        """
        Check if the content contains actual file manipulation functions.
        Returns (has_file_functions, message).
        """
        if self.is_code_block(content):
            return False, ""

        # File manipulation functions and patterns
        file_functions = [
            # Node.js/filesystem
            "fs.read", "fs.write", "readFile", "writeFile",
            "readFileSync", "writeFileSync", "appendFile", "appendFileSync",
            
            # Browser file APIs
            "FileReader", "Blob", "File.createObjectURL",
            "uploadFile", "downloadFile", "saveAs",
            
            # File system operations
            "mkdir", "readdir", "unlink", "rename",
            "copyFile", "createReadStream", "createWriteStream",
            
            # Database file operations
            "open(", ".open(", "sqlite3.open",
            
            # File manipulation
            "file.save", "file.write", "file.read",
            "saveFile", "loadFile", "parseFile",
            
            # File upload/download functions
            "upload.", "download.", "attachment.",
            "uploadToServer", "downloadFromServer",
            
            # Document handling
            "PDFDocument", "ExcelWorkbook", "WordDocument",
            "Spreadsheet", "Workbook", "Document"
        ]
        
        for func in file_functions:
            if func.lower() in content.lower():
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
