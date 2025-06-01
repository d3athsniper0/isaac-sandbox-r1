# services/llm_service.py
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from openai import AsyncOpenAI
from groq import AsyncGroq

from config import OPENAI_API_KEY, GROQ_API_KEY, FUNCTION_SPECS, SYSTEM_PROMPT # type: ignore
from models.event_models import ContentEvent, FunctionCallEvent, ErrorEvent # type: ignore
from models.request_models import ChatRequest # type: ignore    

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with Language Models"""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    
    async def stream_chat_completion(
        self, 
        request: ChatRequest, 
        stream_id: str,
        stream_manager,
        function_service
    ) -> None:
        """Stream a chat completion from the LLM"""
        try:
            # Prepare messages for the LLM
            messages = []
            
            # Add system message if not already present
            if not any(m.role == "system" for m in request.messages):
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            
            # Add the rest of the messages
            messages.extend([m.dict() for m in request.messages])
            
            # Prepare the params for the API call
            params = {
                "model": request.model,
                "messages": messages,
                "stream": True
            }
            
            # Add optional parameters if provided
            for param in ["temperature", "max_tokens"]:
                if hasattr(request, param) and getattr(request, param) is not None:
                    params[param] = getattr(request, param)
            
            # Add function calling capabilities if needed
            if "gpt-4" in request.model or "gpt-3.5" in request.model:
                params["tools"] = FUNCTION_SPECS
                params["tool_choice"] = "auto"
            
            # Stream the completion
            current_content = ""
            async for chunk in self.openai_client.chat.completions.create(**params):
                # Check if we have a delta in the message
                if hasattr(chunk.choices[0], 'delta'):
                    delta = chunk.choices[0].delta
                    
                    # Handle content
                    if hasattr(delta, 'content') and delta.content:
                        current_content += delta.content
                        
                        # Push content event to stream
                        await stream_manager.push_event(
                            stream_id,
                            ContentEvent(
                                request_id=stream_id,
                                content=delta.content,
                                is_complete=False
                            )
                        )
                    
                    # Handle tool calls
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            # Check if this is a complete tool call
                            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                                function_name = tool_call.function.name
                                
                                # Try to parse the arguments
                                try:
                                    arguments = json.loads(tool_call.function.arguments)
                                except Exception:
                                    # If it's not valid JSON yet, it might be incomplete
                                    continue
                                
                                # Push function call event to stream
                                await stream_manager.push_event(
                                    stream_id,
                                    FunctionCallEvent(
                                        request_id=stream_id,
                                        function_name=function_name,
                                        arguments=arguments,
                                        display_message=f"I'm looking up information about {arguments.get('question', function_name)}..."
                                    )
                                )
                                
                                # Start executing the function asynchronously
                                asyncio.create_task(
                                    function_service.execute_function(
                                        stream_id,
                                        function_name,
                                        arguments,
                                        stream_manager
                                    )
                                )
            
            # If no function calls were made, push a complete event
            if not any(isinstance(event, FunctionCallEvent) for event in stream_manager.active_streams.get(stream_id, [])):
                # Push a complete event
                await stream_manager.push_event(
                    stream_id,
                    ContentEvent(
                        request_id=stream_id,
                        content=current_content,
                        is_complete=True
                    )
                )
        
        except Exception as e:
            logger.error(f"Error in stream_chat_completion: {e}")
            # Push error event to stream
            await stream_manager.push_event(
                stream_id,
                ErrorEvent(
                    request_id=stream_id,
                    error=str(e),
                    message="An error occurred while generating the response."
                )
            )