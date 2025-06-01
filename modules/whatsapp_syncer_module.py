from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import os
import sys
import json
from datetime import datetime
import logging
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the WhatsAppSyncer
from modules.whatsapp_syncer import WhatsAppSyncer # type: ignore

# Create a router for whatsapp sync routes
router = APIRouter(prefix="/whatsapp-sync", tags=["whatsapp-sync"])

# Create a scheduler
scheduler = AsyncIOScheduler()

# Authentication dependency
async def verify_auth_key(request: Request):
    auth_key = request.headers.get('X-Auth-Key')
    expected_key = os.environ.get("WHATSAPP_SYNC_AUTH_KEY")
    
    if not expected_key:
        logger.warning("WHATSAPP_SYNC_AUTH_KEY environment variable not set. API endpoint will not be secured.")
        return True
    
    if auth_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

@router.post("/sync")
async def manual_sync(
    request: Request, 
    background_tasks: BackgroundTasks,
    authorized: bool = Depends(verify_auth_key)
):
    """Endpoint to manually trigger a WhatsApp conversation sync"""
    try:
        # Get the full_sync parameter from request
        data = {}
        try:
            data = await request.json()
        except Exception:
            # If request has no body or invalid JSON, use empty dict
            data = {}
            
        full_sync = data.get('full_sync', False)
        
        logger.info(f"Starting manual WhatsApp sync job (full_sync={full_sync})...")
        
        # Run sync in background task
        background_tasks.add_task(run_sync_job, full_sync)
        
        return JSONResponse({
            "success": True, 
            "message": "WhatsApp conversation sync started in background"
        })
        
    except Exception as e:
        logger.error(f"Error in manual WhatsApp sync: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({
            "success": False, 
            "error": str(e)
        }, status_code=500)

async def run_sync_job(full_sync: bool = False) -> Dict[str, Any]:
    """Run a WhatsApp sync job asynchronously"""
    try:
        logger.info(f"Running WhatsApp sync job (full_sync={full_sync})...")
        
        # Initialize syncer
        syncer = WhatsAppSyncer()
        
        # Set up database
        syncer.setup_database()
        
        # Run sync
        results = syncer.sync_conversations(full_sync=full_sync)
        
        # Close connection
        syncer.close()
        
        logger.info(f"WhatsApp sync job completed with results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error in WhatsApp sync job: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

async def scheduled_sync_job():
    """Background job to sync WhatsApp conversations on a schedule"""
    try:
        logger.info("Starting scheduled WhatsApp sync job...")
        await run_sync_job(full_sync=False)
    except Exception as e:
        logger.error(f"Error in scheduled WhatsApp sync: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def register_whatsapp_syncer(app):
    """Register the WhatsApp syncer router and scheduler with the FastAPI app"""
    # Register router
    app.include_router(router)
    
    # Setup startup event to start scheduler
    @app.on_event("startup")
    async def start_scheduler():
        # Schedule the sync job if enabled
        if os.environ.get("ENABLE_WHATSAPP_SYNC_SCHEDULER", "false").lower() == "true":
            # Get sync interval from environment (default: 1 hour)
            sync_interval = int(os.environ.get("WHATSAPP_SYNC_INTERVAL_HOURS", 1))
            
            # Add the job to the scheduler
            scheduler.add_job(
                func=scheduled_sync_job,
                trigger="interval",
                hours=sync_interval,
                id="whatsapp_sync_job"
            )
            
            # Start the scheduler
            scheduler.start()
            logger.info(f"WhatsApp sync scheduler started with {sync_interval}-hour interval")
    
    # Setup shutdown event to stop scheduler
    @app.on_event("shutdown")
    async def stop_scheduler():
        if scheduler.running:
            scheduler.shutdown()
            logger.info("WhatsApp sync scheduler stopped")
    
    logger.info("WhatsApp syncer registered successfully")
    
    return app