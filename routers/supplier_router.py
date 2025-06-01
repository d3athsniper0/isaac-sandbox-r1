# routers/supplier_router.py
import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
import os

from models.request_models import ChatRequest # type: ignore
from modules.fast_memory import FastMemoryManager # type: ignore
from modules.chat_processor import enhance_chat_completion, FUNCTION_SPECS # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter()

# Define supplier-specific keywords
SUPPLIER_KEYWORDS = [
    'product', 'material', 'composite', 'equipment', 'catalog', 
    'price', 'item', 'brand', 'what does', 'what do you', 
    'offer', 'sell', 'inventory', 'stock', 'available',
    'order', 'purchase', 'buy', 'cost', 'pricing',
    'specification', 'specs', 'feature', 'benefit',
    'recommend', 'recommendation', 'suggest', 'suggestion',
    'blade', 'instrument', 'tool', 'device', 'kit',
    'system', 'solution', 'option', 'alternative',
    'supplier', 'manufacturer', 'vendor', 'provide',
    'dental product', 'dental material', 'dental equipment',
    'surgical', 'microsurgical', 'micro surgical'
]

SUPPLIER_PATTERNS = [
    "do you have", "do you sell", "do you offer",
    "can i get", "can i buy", "can i order",
    "show me", "list of", "what are your",
    "what kind of", "which type of", "what type of",
    "do you recommend", "what do you recommend",
    "looking for", "searching for", "need a"
]

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
                          f"Available categories: {', '.join(supplier_context.get('categories', {}).keys()) if isinstance(supplier_context.get('categories'), dict) else 'Products, Services'}. "
                          f"Supplier ID: {supplier_id}"
            }
            messages.insert(-1, supplier_system_msg)  # Insert before last user message
        
        # Update request with modified messages
        supplier_request["messages"] = [type(request.messages[0])(**msg) for msg in messages]
        
        # Check if we need to force supplier product retrieval
        force_supplier_search = check_supplier_requirements(
            request.messages[-1].content if request.messages else "",
            supplier_id
        )
        
        # If supplier search is needed, add tool configuration
        if force_supplier_search:
            supplier_request["tools"] = FUNCTION_SPECS
            supplier_request["tool_choice"] = {
                "type": "function",
                "function": {"name": "retrieve_supplier_record"}
            }
            
            # Add instruction to force supplier search
            search_instruction = {
                "role": "system",
                "content": f"CRITICAL: The user is asking about supplier products. You MUST use retrieve_supplier_record with search_type='supplier_products', query='{request.messages[-1].content}', and supplier_id='{supplier_id}' to find relevant products. Do not modify the query."
            }
            messages.insert(-1, search_instruction)
            supplier_request["messages"] = [type(request.messages[0])(**msg) for msg in messages]
            
            logger.info(f"[SUPPLIER DEBUG] Forcing retrieve_supplier_record for supplier {supplier_id} with query: {request.messages[-1].content}")
        else:
            # For general queries, still enable tools but set to auto
            supplier_request["tools"] = FUNCTION_SPECS
            supplier_request["tool_choice"] = "auto"
            logger.info(f"[SUPPLIER DEBUG] General query for supplier {supplier_id}, tools enabled with auto mode")
        
        # Add supplier_id to the request
        supplier_request["supplier_id"] = supplier_id
        
        supplier_request = ChatRequest(**supplier_request)
        
        # Process with existing chat completion
        return await enhance_chat_completion(supplier_request, memory_manager)
        
    except Exception as e:
        logger.error(f"Error in supplier_chat_completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def check_supplier_requirements(user_input: str, supplier_id: str) -> bool:
    """
    Check if the user's message requires supplier product retrieval.
    Returns True if supplier product lookup should be forced.
    """
    if not supplier_id or not user_input:
        return False
    
    txt = user_input.lower()
    
    # Check if any supplier keyword is present
    if any(keyword in txt for keyword in SUPPLIER_KEYWORDS):
        logger.info(f"[SUPPLIER DEBUG] Supplier keyword detected in query: '{user_input}' for supplier: {supplier_id}")
        return True
    
    # Check for supplier patterns
    if any(pattern in txt for pattern in SUPPLIER_PATTERNS):
        logger.info(f"[SUPPLIER DEBUG] Supplier pattern detected in query: '{user_input}' for supplier: {supplier_id}")
        return True
    
    return False

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