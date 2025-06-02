# services/stream_manager.py
import asyncio
import json
import uuid
import time
import logging
from typing import Dict, List, Any, AsyncGenerator, Optional
from fastapi import WebSocket
from sse_starlette.sse import EventSourceResponse

from models.event_models import ( #type: ignore
    StreamEvent, ThinkingEvent, ContentEvent, FunctionCallEvent,
    FunctionResultEvent, CompleteEvent, ErrorEvent
)

logger = logging.getLogger(__name__)

class StreamManager:
    """Manages event streams for chat completions"""
    
    def __init__(self):
        self.active_streams: Dict[str, asyncio.Queue] = {}
        self.stream_tasks: Dict[str, asyncio.Task] = {}
        
        # New tracking variables
        self.last_get_info_times: Dict[str, float] = {}
        self.last_get_info_results: Dict[str, str] = {}
    
    def create_stream(self, request_id: Optional[str] = None) -> str:
        """Create a new event stream and return the stream ID"""
        stream_id = request_id or str(uuid.uuid4())
        self.active_streams[stream_id] = asyncio.Queue()
        return stream_id
    
    async def push_event(self, stream_id: str, event: StreamEvent) -> None:
        """Push an event to the stream"""
        if stream_id not in self.active_streams:
            logger.warning(f"Attempted to push event to nonexistent stream: {stream_id}")
            return
        
        # Track get_information events
        if isinstance(event, FunctionResultEvent) and event.function_name == "get_information":
            self.last_get_info_times[stream_id] = time.time()
            if event.formatted_result:
                self.last_get_info_results[stream_id] = event.formatted_result
            elif event.result and isinstance(event.result, dict) and "content" in event.result:
                # Fallback to raw content if formatted is not available
                content = event.result["content"]
                self.last_get_info_results[stream_id] = f'<div class="research-container"><p>{content}</p></div>'
        
        await self.active_streams[stream_id].put(event)
    
    async def get_generator(self, stream_id: str) -> AsyncGenerator[str, None]:
        """Return a generator that yields events from the stream"""
        if stream_id not in self.active_streams:
            logger.error(f"Attempted to get generator for nonexistent stream: {stream_id}")
            yield json.dumps({"error": "Stream not found"})
            return
        
        queue = self.active_streams[stream_id]
        
        try:
            while True:
                # Wait for the next event with a timeout
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=60.0)
                    
                    # Convert event to dict for serialization
                    event_dict = event.dict()
                    
                    # Yield the event
                    yield json.dumps(event_dict)
                    
                    # If this is a complete or error event, break the loop
                    if event.type in ["complete", "error"]:
                        break
                        
                except asyncio.TimeoutError:
                    # No event received for 60 seconds, send a keepalive
                    yield ":keepalive"
                    
                    # ADD THE CIRCUIT BREAKER HERE
                    # Check if we have a stalled get_information result
                    if (stream_id in self.last_get_info_times and 
                        stream_id in self.last_get_info_results and
                        time.time() - self.last_get_info_times[stream_id] > 15):
                        
                        logger.warning(f"Circuit breaker triggered for stream {stream_id} - forcing research results")
                        
                        # Force content event with the research results
                        content_event = ContentEvent(
                            request_id=stream_id,
                            content=self.last_get_info_results[stream_id],
                            is_complete=True
                        )
                        yield json.dumps(content_event.dict())
                        
                        # Force a complete event
                        complete_event = CompleteEvent(request_id=stream_id)
                        yield json.dumps(complete_event.dict())
                        
                        # Clear the tracking data to prevent multiple triggers
                        if stream_id in self.last_get_info_times:
                            del self.last_get_info_times[stream_id]
                        if stream_id in self.last_get_info_results:
                            del self.last_get_info_results[stream_id]
                        
                        # Break the loop to end the stream
                        break

        except asyncio.CancelledError:
            logger.info(f"Stream {stream_id} was cancelled")
        except Exception as e:
            logger.error(f"Error in stream generator: {e}")
            yield json.dumps({"type": "error", "error": str(e)})
        finally:
            # Clean up the stream
            await self.close_stream(stream_id)
    
    async def close_stream(self, stream_id: str) -> None:
        """Close and clean up the stream"""
        if stream_id in self.active_streams:
            # Send a final complete event if none was sent
            try:
                await self.active_streams[stream_id].put(
                    CompleteEvent(request_id=stream_id)
                )
            except Exception:
                pass
            
            # Remove from active streams
            del self.active_streams[stream_id]
        
        # Cancel any associated tasks
        if stream_id in self.stream_tasks:
            self.stream_tasks[stream_id].cancel()
            del self.stream_tasks[stream_id]
    
    def create_sse_response(self, stream_id: str) -> EventSourceResponse:
        """Create an SSE response from the stream"""
        return EventSourceResponse(self.get_generator(stream_id))
    
    def register_task(self, stream_id: str, task: asyncio.Task) -> None:
        """Register a task associated with a stream"""
        self.stream_tasks[stream_id] = task