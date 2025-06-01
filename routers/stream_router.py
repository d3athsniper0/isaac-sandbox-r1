# routers/stream_router.py
import asyncio
import json
import logging
import uuid
from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse
from typing import Dict, List, Any, Optional

from config import WORKER_CONNECTIONS
from models.request_models import StreamRequest
from models.event_models import ThinkingEvent
from services.stream_manager import StreamManager
from services.llm_service import LLMService
from services.function_service import FunctionService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
stream_manager = StreamManager()
llm_service = LLMService()
function_service = FunctionService()

# Set up semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(WORKER_CONNECTIONS)

@router.post("/v1/chat/completions/stream")
async def stream_chat_completion(request: StreamRequest) -> EventSourceResponse:
    """Stream a chat completion with function calling support"""
    # Generate a request ID if not provided
    request_id = request.request_id or str(uuid.uuid4())
    
    # Create a new stream
    stream_id = stream_manager.create_stream(request_id)
    
    # Acquire semaphore to limit concurrent requests
    async def stream_generator():
        try:
            async with semaphore:
                # Send thinking event
                await stream_manager.push_event(
                    stream_id,
                    ThinkingEvent(request_id=stream_id)
                )
                
                # Start task to stream completion
                task = asyncio.create_task(
                    llm_service.stream_chat_completion(
                        request, stream_id, stream_manager, function_service
                    )
                )
                
                # Register task with stream manager
                stream_manager.register_task(stream_id, task)
                
                # Yield events from the stream
                async for event in stream_manager.get_generator(stream_id):
                    yield event
                    
        except Exception as e:
            logger.error(f"Error in stream_generator: {e}")
            yield json.dumps({"type": "error", "error": str(e)})
            
        finally:
            # Ensure stream is closed properly
            await stream_manager.close_stream(stream_id)
    
    # Return SSE response
    return EventSourceResponse(stream_generator())

@router.post("/v1/chat/stream/{stream_id}/abort")
async def abort_stream(stream_id: str):
    """Abort an ongoing stream"""
    if stream_id not in stream_manager.active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    # Close the stream
    await stream_manager.close_stream(stream_id)
    
    return {"status": "success", "message": f"Stream {stream_id} aborted successfully"}