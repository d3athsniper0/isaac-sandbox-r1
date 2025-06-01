# routers/supplier_router.py
import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
import os

from models.request_models import ChatRequest # type: ignore
from modules.fast_memory import FastMemoryManager # type: ignore
from modules.chat_processor import enhance_chat_completion # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize memory manager
memory_manager = FastMemoryManager(
    index_name=os.getenv("PINECONE_INDEX", "trust")
)

@router.post("/v1/suppliers/{supplier_id}/chat/completions")
async def supplier_chat_completion(supplier_id: str, request: ChatRequest):
    """Create a chat completion with supplier-specific context"""
    try:
        # Add supplier context to the request
        supplier_context = await get_supplier_context(supplier_id, request.messages[-1].content if request.messages else "")
        
        # Modify system prompt to include supplier context
        supplier_request = request.dict()
        messages = [m.dict() for m in request.messages]
        
        # Add supplier context as system message
        if supplier_context:
            supplier_system_msg = {
                "role": "system",
                "content": f"SUPPLIER CONTEXT: You are representing {supplier_context.get('supplier_name', 'this supplier')}. "
                          f"When discussing products or services, prioritize information from this supplier. "
                          f"Available categories: {', '.join(supplier_context.get('categories', {}).keys())}"
            }
            messages.insert(-1, supplier_system_msg)  # Insert before last user message
        
        # Update request with modified messages
        supplier_request["messages"] = [type(request.messages[0])(**msg) for msg in messages]
        supplier_request = ChatRequest(**supplier_request)
        
        # Process with existing chat completion
        return await enhance_chat_completion(supplier_request, memory_manager)
        
    except Exception as e:
        logger.error(f"Error in supplier_chat_completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_supplier_context(supplier_id: str, query: str) -> Dict[str, Any]:
    """Get supplier context for the chat"""
    try:
        # Get supplier overview
        overview = await memory_manager.fast_pinecone_retrieval.retrieve_records_async(
            search_type="supplier_overview",
            query="",
            supplier_id=supplier_id
        )
        
        if overview and len(overview) > 0:
            return overview[0]
        
        return {}
    except Exception as e:
        logger.error(f"Error getting supplier context: {e}")
        return {}