# models/response_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union

class FunctionCall(BaseModel):
    name: str
    arguments: str

class FunctionTool(BaseModel):
    type: str = "function"
    function: FunctionCall

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[FunctionTool]] = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(import time; time.time()))
    model: str
    choices: List[Choice]

class FunctionResponse(BaseModel):
    """Base response model for function calls"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class GetInformationResponse(FunctionResponse):
    """Response model for get_information function"""
    content: Optional[str] = None
    citations: Optional[List[Union[str, Dict[str, str]]]] = None
    formatted_content: Optional[str] = None