"""
AWS Bedrock Agent Pipeline for OpenWebUI
This pipeline integrates AWS Bedrock Agents with OpenWebUI.
"""

import boto3
import json
import uuid
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        AWS_ACCESS_KEY_ID: str = ""
        AWS_SECRET_ACCESS_KEY: str = ""
        AWS_REGION: str = "us-east-1"
        AGENT_ID: str = ""
        AGENT_ALIAS_ID: str = "TSTALIASID"  # or specific alias ID
        SESSION_TIMEOUT: int = 600
        pass

    def __init__(self):
        self.type = "manifold"
        self.id = "aws_bedrock_agent"
        self.name = "AWS Bedrock Agent"
        self.valves = self.Valves()

    def pipelines(self) -> List[dict]:
        return [
            {
                "id": "bedrock_agent",
                "name": "AWS Bedrock Agent",
            }
        ]

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        
        if not self.valves.AGENT_ID:
            return "Error: Agent ID not configured. Please set AGENT_ID in pipeline settings."
        
        try:
            # Initialize Bedrock Agent Runtime client
            session = boto3.Session(
                aws_access_key_id=self.valves.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=self.valves.AWS_SECRET_ACCESS_KEY,
                region_name=self.valves.AWS_REGION
            )
            
            bedrock_agent_runtime = session.client('bedrock-agent-runtime')
            
            # Generate a unique session ID for this conversation
            session_id = str(uuid.uuid4())
            
            # Extract just the user's message (last message in the conversation)
            if messages:
                user_input = messages[-1].get('content', user_message)
            else:
                user_input = user_message
            
            # Invoke the Bedrock Agent
            response = bedrock_agent_runtime.invoke_agent(
                agentId=self.valves.AGENT_ID,
                agentAliasId=self.valves.AGENT_ALIAS_ID,
                sessionId=session_id,
                inputText=user_input
            )
            
            # Process the streaming response
            response_text = ""
            if 'completion' in response:
                for event in response['completion']:
                    if 'chunk' in event:
                        chunk = event['chunk']
                        if 'bytes' in chunk:
                            chunk_text = chunk['bytes'].decode('utf-8')
                            response_text += chunk_text
                        elif 'attribution' in chunk:
                            # Handle citations/sources if needed
                            attribution = chunk['attribution']
                            if 'citations' in attribution:
                                citations = attribution['citations']
                                # You can format citations here
                                pass
                    elif 'trace' in event:
                        # Handle trace information for debugging
                        trace = event['trace']
                        # You can log trace info if needed
                        pass
            
            return response_text if response_text else "No response from agent."
            
        except Exception as e:
            error_msg = f"Error invoking Bedrock Agent: {str(e)}"
            print(error_msg)  # For debugging
            return error_msg

    def get_session_id(self, user_id: str = None) -> str:
        """Generate or retrieve session ID for conversation continuity"""
        # You could implement session persistence here
        return str(uuid.uuid4())
