# services/function_service.py (Refactored version)
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple

from config import FUNCTION_TIMEOUT, OPENAI_API_KEY #type: ignore
from models.event_models import FunctionResultEvent, ErrorEvent, ContentEvent, CompleteEvent #type: ignore
from openai import AsyncOpenAI

# Import the modules directly
from modules.fast_pplx_manager import FastPPLXManager #type: ignore
from modules.fast_pinecone_retrieval import FastPineconeRetrieval #type: ignore

import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

class FunctionService:
    """Service for executing functions asynchronously without HTTP overhead"""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize the modules directly
        self.pplx_manager = FastPPLXManager()
        self.pinecone_retrieval = FastPineconeRetrieval()
    
    async def execute_function(
        self,
        stream_id: str,
        function_name: str,
        arguments: Dict[str, Any],
        stream_manager
    ) -> None:
        """Execute a function and push the result to the stream"""
        try:
            result = None
            formatted_result = None
            
            # Execute the appropriate function directly
            if function_name == "get_information":
                result, formatted_result = await self.execute_get_information(arguments)
            elif function_name == "retrieve_record":
                result = await self.execute_retrieve_record(arguments)
            # Add other function handlers here...
            else:
                raise ValueError(f"Unknown function: {function_name}")
            
            # Push function result event to stream
            await stream_manager.push_event(
                stream_id,
                FunctionResultEvent(
                    request_id=stream_id,
                    function_name=function_name,
                    result=result,
                    formatted_result=formatted_result
                )
            )
            
            # KEY CODE HERE:
            # If we have a formatted result for get_information, force completion
            if function_name == "get_information" and formatted_result:
                # Push the formatted result as content and mark as complete
                await stream_manager.push_event(
                    stream_id,
                    ContentEvent(
                        request_id=stream_id,
                        content=formatted_result,
                        is_complete=True
                    )
                )
                
                # Force an immediate completion event
                await stream_manager.push_event(
                    stream_id,
                    CompleteEvent(
                        request_id=stream_id
                    )
                )
                        
            # If we have a formatted result, push it as content and mark as complete
            elif formatted_result:
                await stream_manager.push_event(
                    stream_id,
                    ContentEvent(
                        request_id=stream_id,
                        content=formatted_result,
                        is_complete=True
                    )
                )
                
                # Push complete event
                await stream_manager.push_event(
                    stream_id,
                    CompleteEvent(
                        request_id=stream_id
                    )
                )
            
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
            # Push error event to stream
            await stream_manager.push_event(
                stream_id,
                ErrorEvent(
                    request_id=stream_id,
                    error=str(e),
                    message=f"An error occurred while executing function {function_name}."
                )
            )
    # Execute the get_information function with internal formatting
    async def execute_get_information(
    self, 
    arguments: Dict[str, Any]
) -> Tuple[Dict[str, Any], str]:
        """Execute the get_information function with internal formatting"""
        try:
            # Extract the question
            question = arguments.get("question", "")
            if not question:
                raise ValueError("No question provided")
                
            # Get information directly from the already initialized pplx_manager
            content, citations = await self.pplx_manager.get_information_async(question)
            
            # Create result dictionary
            if content:
                result = {
                    "success": True,
                    "content": content,
                    "citations": citations or []
                }
            else:
                # Handle null content with fallback
                fallback_content = (
                    f"I couldn't find specific information about '{question}'. "
                    "You might want to try rephrasing your question or consulting "
                    "specific dental research publications."
                )
                
                result = {
                    "success": True,
                    "content": fallback_content,
                    "citations": []
                }
            
            # Format the result using LLM
            formatted_result = await self.format_information(
                result.get("content"),
                result.get("citations", []),
                question
            )
            
            # Return both raw result and formatted result
            return result, formatted_result
                
        except Exception as e:
            logger.error(f"Error in execute_get_information: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "content": f"I encountered an issue while retrieving information. The error was: {str(e)}"
            }
            return error_result, error_result["content"]
    
    async def format_information(
        self,
        content: str,
        citations: List[Any],
        query: str
    ) -> str:
        """Format information using LLM"""
        try:
            # Create formatting prompt
            formatting_prompt = f"""
            Format these search results about "{query}" into a clear, well-structured response:

            Content: {content}
            
            Citations: {json.dumps(citations)}
            
            Requirements:
            1. Complete any cut-off sentences based on context
            2. Format with proper Markdown headings and bullet points
            3. Convert citation references like [1] into proper links
            4. Make your response comprehensive but concise
            5. CRITICAL: In the "References" section use each source’s **title or a clear short descriptor** as the clickable text (never "Source 1", "Source 2", …) and number them.
            """
            
            # Call LLM for formatting
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",  # Use faster model for formatting
                messages=[
                    {"role": "system", "content": "You format search results into clear, well-structured responses."},
                    {"role": "user", "content": formatting_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            # Return formatted content
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error formatting information: {e}")
            # Return a basic formatted version as fallback