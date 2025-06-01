# utils/stream_utils.py
import json
import logging
from typing import Any, Dict, Generator, Optional
from fastapi import Response
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

def create_sse_event(data: Dict[str, Any], event: Optional[str] = None) -> Dict[str, Any]:
    """Create an SSE event with data and optional event type"""
    result = {"data": json.dumps(data)}
    if event:
        result["event"] = event
    return result

def format_sse_event(data: Dict[str, Any], event: Optional[str] = None) -> str:
    """Format an SSE event as a string"""
    result = f"data: {json.dumps(data)}\n"
    if event:
        result = f"event: {event}\n{result}"
    return f"{result}\n"

def create_sse_response(generator) -> EventSourceResponse:
    """Create an SSE response from a generator"""
    return EventSourceResponse(generator)

def error_event(message: str, code: int = 500) -> Dict[str, Any]:
    """Create an error event"""
    return {"type": "error", "error": message, "code": code}

def thinking_event() -> Dict[str, Any]:
    """Create a thinking event"""
    return {"type": "thinking", "message": "I'm thinking..."}

def function_call_event(func_name: str, display_message: Optional[str] = None) -> Dict[str, Any]:
    """Create a function call event"""
    event = {"type": "function_call", "function_name": func_name}
    if display_message:
        event["display_message"] = display_message
    return event

def content_event(content: str, is_complete: bool = False) -> Dict[str, Any]:
    """Create a content event"""
    return {"type": "content", "content": content, "is_complete": is_complete}

def complete_event() -> Dict[str, Any]:
    """Create a complete event"""
    return {"type": "complete"}