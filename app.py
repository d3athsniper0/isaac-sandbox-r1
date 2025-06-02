# app.py
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio

from routers import chat_router, stream_router, whatsapp_router  # type: ignore
from config import DEBUG # type: ignore

from routers.whatsapp_router import cleanup_old_sessions #type: ignore

# Import the WhatsApp syncer module
from modules.whatsapp_syncer_module import register_whatsapp_syncer # type: ignore

# Import supplier router
from routers import supplier_router # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create FastAPI app
app = FastAPI(
    title="AI Dental Assistant API",
    description="Streaming API for the AI Dental Assistant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router.router)
app.include_router(stream_router.router)
app.include_router(whatsapp_router.router, prefix="/whatsapp")  # WhatsApp router
app.include_router(supplier_router.router) # Supplier router

# Register the WhatsApp syncer
register_whatsapp_syncer(app)

# Start session cleanup task
@app.on_event("startup")
async def start_cleanup():
    asyncio.create_task(cleanup_old_sessions())

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the AI Dental Assistant API",
        "documentation": "/docs",
        "health": "/health"
    }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        workers=4
    )