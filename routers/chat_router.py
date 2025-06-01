# routers/chat_router.py
import asyncio
import json
import logging
import os
from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Dict, List, Any, Optional

from config import DEFAULT_MODEL, FALLBACK_MODEL, MAX_RETRIES #type: ignore
from models.request_models import ChatRequest #type: ignore
from services.llm_service import LLMService #type: ignore
from modules.fast_memory import FastMemoryManager #type: ignore

# Import the chat processor
from modules.chat_processor import enhance_chat_completion, register_chat_processor #type: ignore

# Setup logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize services
llm_service = LLMService()

# Initialize memory manager
memory_manager = FastMemoryManager(
    index_name=os.getenv("PINECONE_INDEX", "trust")
)

@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    """Create a chat completion with context retrieval and function calling support"""
    try:

        # 👇  DEBUG: show the entire inbound payload
        logger.warning(
            "💬  /v1/chat/completions payload ➜ %s",
            json.dumps(request.model_dump(), indent=2)     # pydantic ≥1.10
        )

        # Use the enhanced chat completion function from chat_processor
        chat_completion = await enhance_chat_completion(request, memory_manager)
        
        # Return the response
        return chat_completion
    except Exception as e:
        logger.error(f"Error in create_chat_completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/chat/completions/fallback")
async def create_chat_completion_fallback(request: ChatRequest):
    """Fallback endpoint for chat completions using a simpler model"""
    try:
        # Create a copy of the request with fallback model
        fallback_request = request.dict()
        fallback_request["model"] = FALLBACK_MODEL
        
        # Remove advanced features that might not be supported
        for key in ["functions", "function_call"]:
            if key in fallback_request:
                del fallback_request[key]
                
        # Convert back to ChatRequest
        fallback_request = ChatRequest(**fallback_request)
        
        # Use the enhanced chat completion function with fallback model
        chat_completion = await enhance_chat_completion(fallback_request, memory_manager)
        
        # Return the response
        return chat_completion
    except Exception as e:
        logger.error(f"Error in create_chat_completion_fallback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve-supplier-record")
async def retrieve_supplier_record(request: Dict[str, Any]):
    """Retrieve supplier product and company information"""
    try:
        # Extract parameters
        search_type = request.get("search_type")
        query = request.get("query")
        supplier_id = request.get("supplier_id")
        top_k = request.get("top_k", 10)
        
        # Call the retrieval method with supplier support
        records = await memory_manager.fast_pinecone_retrieval.retrieve_records_async(
            search_type=search_type,
            query=query,
            supplier_id=supplier_id,
            top_k=top_k
        )
        
        # Format and return results
        return {
            "success": True,
            "records": records
        }
    except Exception as e:
        logger.error(f"Error in retrieve_supplier_record: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@router.post("/generate-treatment-plan")
async def generate_treatment_plan(request: Dict[str, Any]):
    """Generate a dental treatment plan based on patient information"""
    try:
        # Implement treatment plan generation logic
        # This might involve calling an external service or using the LLM
        
        # Basic implementation - convert to proper function in production
        system_prompt = "You are a dental assistant creating a treatment plan. Be thorough and professional."
        user_prompt = f"""Generate a comprehensive treatment plan for:
        Patient: {request.get('patient_name', 'Unknown')}
        Age: {request.get('age', 'Unknown')}
        Chief Complaint: {request.get('chief_complaint', 'None')}
        Medical History: {request.get('medical_history', 'None')}
        Dental History: {request.get('dental_history', 'None')}
        """
        
        # Add optional fields if present
        for field in ['current_medications', 'xray_findings', 'budget_constraint', 
                     'time_constraint', 'insurance_info', 'additional_info']:
            if field in request and request[field]:
                user_prompt += f"\n{field.replace('_', ' ').title()}: {request[field]}"
        
        # Create a chat request
        chat_request = ChatRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=DEFAULT_MODEL,
            temperature=0.3,  # Lower temperature for more consistent plans
            max_tokens=1500,
            stream=False
        )
        
        # Get completion
        result = await enhance_chat_completion(chat_request, memory_manager)
        
        # Extract the treatment plan
        if hasattr(result, 'choices') and len(result.choices) > 0:
            treatment_plan = result.choices[0].message.content
            # Format response
            return {
                "success": True,
                "treatment_plan": treatment_plan,
                "patient_name": request.get('patient_name', 'Unknown')
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate treatment plan"
            }
    except Exception as e:
        logger.error(f"Error in generate_treatment_plan: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@router.post("/retrieve-record")
async def retrieve_record(request: Dict[str, Any]):
    """Retrieve patient records from database"""
    try:
        # Extract parameters
        search_type = request.get("search_type")
        query = request.get("query")
        practice_id = request.get("practice_id")
        top_k = request.get("top_k", 10)
        index_name = request.get("index_name", "trust")
        
        # Call the appropriate retrieval method based on search_type
        if not hasattr(memory_manager, "fast_pinecone_retrieval"):
            # Direct retrieval methods if FastPineconeRetrieval not integrated
            if search_type == "patient":
                records = await memory_manager.retrieve_patient_data_async(query, "", top_k)
            elif search_type == "text":
                records = await memory_manager.search_all_patients_async(query, top_k)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported search type: {search_type}"
                }
        else:
            # Use FastPineconeRetrieval if integrated
            records = await memory_manager.fast_pinecone_retrieval.retrieve_records_async(
                search_type, query, practice_id, top_k, index_name
            )
        
        # Format and return results
        return {
            "success": True,
            "records": records
        }
    except Exception as e:
        logger.error(f"Error in retrieve_record: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# New endpoint to get information from external knowledge sources
@router.post("/get-information")
async def get_information(request: Dict[str, Any]):
    """Get information from external knowledge sources"""
    try:
        # Extract the question
        question = request.get("question", "")
        if not question:
            logger.warning("get-information called without a question")
            return {
                "success": False,
                "error": "No question provided"
            }
        
        logger.info(f"get-information called with question: {question}")
        print(f"get-information called with question: {question}")  # Direct console output
        
        # Use the fast_pplx_manager to get information
        from modules.fast_pplx_manager import FastPPLXManager #type: ignore
        
        # Initialize the manager
        try:
            pplx_manager = FastPPLXManager()
        except ValueError as e:
            logger.error(f"Failed to initialize FastPPLXManager: {e}")
            return {
                "success": False,
                "error": f"Failed to initialize Perplexity client: {e}"
            }
        
        # Get information with timeout protection
        try:
            content, citations = await asyncio.wait_for(
                pplx_manager.get_information_async(question),
                timeout=25.0  # 25 second timeout
            )
        except asyncio.TimeoutError:
            logger.error("Timeout while waiting for Perplexity API response")
            return {
                "success": False,
                "error": "Timeout while waiting for response from knowledge source"
            }
        
        # Log the results
        logger.info(f"Perplexity returned content: {bool(content)}")
        if content:
            logger.info(f"Content preview: {content[:50]}...")
        else:
            logger.warning("Perplexity returned null content")
        
        # Check if we got content back
        if content:
            return {
                "success": True,
                "content": content,
                "citations": citations or []
            }
        else:
            # Create fallback content
            fallback_content = (
                f"I couldn't find specific information about '{question}'. "
                "This could be due to API limitations or because the information "
                "isn't available in my knowledge sources. You might want to try "
                "rephrasing your question or consulting specific dental research publications."
            )
            
            logger.warning(f"Using fallback content for question: {question}")
            
            return {
                "success": True,
                "content": fallback_content,
                "citations": []
            }
    except Exception as e:
        logger.error(f"Error in get_information: {e}")
        print(f"Error in get_information: {e}")  # Direct console output
        return {
            "success": False,
            "error": str(e)
        }

# Diagnostic endpoint to check PPLX connection
@router.post("/diagnostics/get-information")
async def check_pplx_connection(request: Dict[str, Any]):
    """Diagnostic endpoint to check PPLX connection with POST request support"""
    try:
        # Extract question from the request
        question = request.get("question")
        if not question:
            return {
                "success": False,
                "error": "No question provided in request",
                "content": "Please provide a question in the request body",
                "citations": []
            }
            
        logger.info(f"Diagnostic endpoint testing question: {question}")
        
        # Import directly to ensure we get the latest version
        from modules.fast_pplx_manager import FastPPLXManager #type: ignore
        
        # Initialize manager
        pplx_manager = FastPPLXManager()
        
        # Test the question
        content, citations = await pplx_manager.get_information_async(question)
        
        # Format the response to match what your test script expects
        return {
            "success": bool(content),
            "content": content or f"No content found for question: {question}",
            "citations": citations or []
        }
    except Exception as e:
        logger.error(f"Error checking PPLX connection: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error occurred: {str(e)}",
            "citations": []
        }

# Define the API endpoints
def register_routes(app):
    """Register any additional routes if needed"""
    # This function can be extended to register more endpoints
    register_chat_processor(app, memory_manager)