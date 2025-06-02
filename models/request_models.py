# models/request_models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "gpt-4o"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1500
    stream: Optional[bool] = False
    user_id: Optional[str] = "anonymous"
    patient_id: Optional[str] = None
    retrieve_context: Optional[bool] = False
    context_query: Optional[str] = None

class StreamRequest(ChatRequest):
    """Request model for streaming endpoint"""
    stream: bool = True
    request_id: Optional[str] = None  # Client can provide request ID for tracking

class SupplierChatRequest(ChatRequest):
    """Request model for supplier-specific chat endpoint"""
    supplier_id: Optional[str] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None

class FunctionRequest(BaseModel):
    """Base model for function call requests"""
    function_name: str
    arguments: Dict[str, Any]
    request_id: str  # To associate with the chat request

class GetInformationRequest(FunctionRequest):
    """Request model for get_information function"""
    function_name: str = "get_information"
    question: str