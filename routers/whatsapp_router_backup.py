# routers/whatsapp_router.py
from fastapi import APIRouter, Request, BackgroundTasks

router = APIRouter()

@router.get("/webhook")
async def verify_webhook(request: Request):
    """Verify webhook for WhatsApp API setup"""
    return "Webhook endpoint placeholder"

@router.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming messages from WhatsApp"""
    return {"status": "ok"}