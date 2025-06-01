# models/event_models.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import time

class EventType(str, Enum):
    THINKING = "thinking"        # LLM is processing
    CONTENT = "content"          # Partial content chunk
    FUNCTION_CALL = "function_call"  # Function is being called
    FUNCTION_RESULT = "function_result"  # Function returned result
    COMPLETE = "complete"        # Response is complete
    ERROR = "error"              # Error occurred

class StreamEvent(BaseModel):
    """Base model for all streaming events"""
    type: EventType
    request_id: str  # Unique ID for the request
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))

class ThinkingEvent(StreamEvent):
    """Indicates the LLM is processing"""
    type: EventType = EventType.THINKING
    message: str = "Thinking..."

class ContentEvent(StreamEvent):
    """Contains a content chunk"""
    type: EventType = EventType.CONTENT
    content: str
    is_complete: bool = False

class FunctionCallEvent(StreamEvent):
    """Indicates a function is being called"""
    type: EventType = EventType.FUNCTION_CALL
    function_name: str
    arguments: Dict[str, Any]
    display_message: Optional[str] = None

class FunctionResultEvent(StreamEvent):
    """Contains a function result"""
    type: EventType = EventType.FUNCTION_RESULT
    function_name: str
    result: Dict[str, Any]
    formatted_result: Optional[str] = None
    error: Optional[str] = None

class CompleteEvent(StreamEvent):
    """Indicates the response is complete"""
    type: EventType = EventType.COMPLETE
    content: Optional[str] = None

class ErrorEvent(StreamEvent):
    """Indicates an error occurred"""
    type: EventType = EventType.ERROR
    error: str
    message: str = "An error occurred"