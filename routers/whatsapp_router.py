# routers/whatsapp_router.py
import os
import json
import logging
import asyncio
import httpx
import time
from datetime import datetime
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple

from models.request_models import ChatRequest, Message #type: ignore
from config import OPENAI_API_KEY, DEFAULT_MODEL #type: ignore

# Redis storage module
from modules.fast_memory import FastMemoryManager  # type: ignore

# Formatting functions
from utils.whatsapp_formatter import strip_html, format_function_result_for_whatsapp, split_long_message #type: ignore

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# WhatsApp API Settings
WHATSAPP_API_URL = os.getenv("WHATSAPP_API_URL")
WHATSAPP_API_TOKEN = os.getenv("WHATSAPP_API_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "your_verify_token")

# Create an instance of FastMemoryManager
memory_manager = FastMemoryManager(
    index_name=os.getenv("PINECONE_INDEX", "trust")
)

# Define inactivity timeout (1 hour in seconds)
INACTIVITY_TIMEOUT = 3600  # 1 hour

# Welcome message for first-time users
WELCOME_MESSAGE = """I'm Isaac, the Trust AI Dental Co-Pilot.

Key Commands:
/reset - Start new conversation
/invite - Share WhatsApp link with friends
/help - Display commands

Questions: Contact support@trustdentistry.ai"""

# Help message content (same as welcome for now)
HELP_MESSAGE = WELCOME_MESSAGE

# Share WhatsApp link
WHATSAPP_LINK = f"https://wa.me/{WHATSAPP_PHONE_ID}"

# Share WhatsApp link message
WHATSAPP_LINK_MESSAGE = f"Share this link with your friends and family: {WHATSAPP_LINK}"

# User session storage 
user_sessions: Dict[str, List[Dict[str, str]]] = {}

class WhatsAppMessage(BaseModel):
    messaging_product: str
    recipient_type: str
    to: str
    type: str
    text: Dict[str, str]

# Migration function to save in-memory sessions to Redis
async def migrate_sessions_to_redis():
    """Migrate all in-memory user sessions to Redis for persistence"""
    migrated_count = 0
    failed_count = 0
    
    logger.info(f"Starting migration of {len(user_sessions)} WhatsApp conversations to Redis")
    
    for phone_number, messages in user_sessions.items():
        try:
            # Clean phone number for consistency
            clean_phone = phone_number
            if clean_phone.startswith("whatsapp:"):
                clean_phone = clean_phone[9:]
                
            # Store in Redis using the same format as your web interface
            success = await memory_manager.store_conversation_async(clean_phone, messages)
            
            if success:
                migrated_count += 1
                logger.info(f"Successfully migrated conversation for {clean_phone}")
            else:
                failed_count += 1
                logger.error(f"Failed to migrate conversation for {clean_phone}")
                
        except Exception as e:
            failed_count += 1
            logger.error(f"Error migrating conversation for {phone_number}: {str(e)}")
    
    logger.info(f"Migration complete: {migrated_count} conversations migrated, {failed_count} failed")
    return {
        "success": True,
        "migrated_count": migrated_count,
        "failed_count": failed_count,
        "total_attempted": len(user_sessions)
    }

# Retry function
async def post_with_retry(url, json_data, max_retries=2, timeout=60.0):
    """Make a POST request with retry logic"""
    attempt = 0
    while attempt <= max_retries:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.post(url, json=json_data)
        except httpx.ReadTimeout:
            attempt += 1
            if attempt > max_retries:
                logger.error(f"Request to {url} timed out after {max_retries} attempts")
                raise
            logger.warning(f"Request timed out, retrying ({attempt}/{max_retries})...")
            # Wait a bit before retrying
            await asyncio.sleep(1)

async def send_whatsapp_message(recipient_phone: str, message_text: str):
    """Send a message via Twilio WhatsApp API"""
    try:
        # If recipient_phone starts with "whatsapp:", use it as is, otherwise add it
        if not recipient_phone.startswith("whatsapp:"):
            recipient_phone = f"whatsapp:{recipient_phone}"
        
        # For Twilio, we use Basic Auth with Account SID and Auth Token
        # The API token should be the base64 of "AccountSID:AuthToken"
        # But httpx can do this for us with auth parameter
        
        # Prepare the message payload for Twilio
        payload = {
            "To": recipient_phone,
            "From": f"whatsapp:{WHATSAPP_PHONE_ID}",
            "Body": message_text
        }
        
        # Get Twilio account SID from environment
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        if not account_sid:
            logger.error("TWILIO_ACCOUNT_SID environment variable not set")
            return False
            
        # Extract auth token from the WHATSAPP_API_TOKEN
        auth_token = WHATSAPP_API_TOKEN
        
        # Make the API request with Basic Auth
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
                data=payload,  # Twilio uses form data
                auth=(account_sid, auth_token)  # Use Basic Auth
            )
            
            if response.status_code >= 300:
                logger.error(f"WhatsApp API error: {response.status_code} - {response.text}")
                return False
                
            logger.info(f"Message sent successfully to {recipient_phone}")
            return True
            
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def handle_media_message(sender: str, message: Dict[str, Any]):
    """Handle media messages (not supported in this implementation)"""
    # Inform the user that we can't process media yet
    await send_whatsapp_message(
        sender, 
        "I'm sorry, I can't process images or other media at the moment. Please send your question as text."
    )

async def execute_get_information(args):
    """Execute the get_information function by calling the API endpoint"""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://isaac-lite-r1-045fb83e2eb2.herokuapp.com/get-information",  # Adjust to actual endpoint
                json=args
            )
            
            if response.status_code != 200:
                logger.error(f"Error calling get_information API: {response.status_code}")
                return {
                    "success": False,
                    "content": "Sorry, I couldn't retrieve that information."
                }
            
            return response.json()
    except Exception as e:
        logger.error(f"Error in execute_get_information: {e}")
        return {
            "success": False,
            "content": f"Error retrieving information: {str(e)}"
        }
    
async def get_conversation_history(phone_number: str) -> Tuple[List[Dict[str, str]], bool, datetime]:
    """
    Retrieve conversation history from Redis
    Returns: (conversation_history, is_first_time, last_timestamp)
    """
    try:
        # Generate Redis key for WhatsApp user
        redis_key = f"whatsapp:{phone_number}"
        
        # Try to get conversation from Redis
        conversation_data = await memory_manager.get_conversation_async(redis_key)
        
        # Check if this is the user's first time
        is_first_time = False
        last_timestamp = None
        
        if not conversation_data:
            # No existing conversation, return empty list
            is_first_time = True
            return [], is_first_time, None
        
        # Process conversation history
        try:
            # Check for timestamps in conversation
            last_message = conversation_data[-1]
            if "timestamp" in last_message:
                last_timestamp = datetime.fromtimestamp(last_message["timestamp"])
            else:
                last_timestamp = None
        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"Error parsing conversation timestamps: {e}")
            last_timestamp = None
        
        return conversation_data, is_first_time, last_timestamp
        
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        # Return empty history on error
        return [], False, None

async def store_conversation_history(phone_number: str, conversation: List[Dict[str, str]]):
    """Store conversation history in Redis"""
    try:
        # Generate Redis key for WhatsApp user
        redis_key = f"whatsapp:{phone_number}"
        
        # Store conversation in Redis
        success = await memory_manager.store_conversation_async(redis_key, conversation)
        if not success:
            logger.error(f"Failed to store conversation for {phone_number}")
        
        return success
    except Exception as e:
        logger.error(f"Error storing conversation history: {e}")
        return False

async def should_reset_conversation(last_timestamp) -> bool:
    """Check if conversation should be reset due to inactivity"""
    if not last_timestamp:
        return False
    
    # Calculate time difference
    now = datetime.now()
    time_diff = now - last_timestamp
    
    # Reset if more than INACTIVITY_TIMEOUT seconds (1 hour)
    return time_diff.total_seconds() > INACTIVITY_TIMEOUT

async def process_user_message(phone_number: str, message_text: str, profile_name: Optional[str] = None):
    """Process a message from a user and send a response"""
    try:
        # Clean phone number - remove "whatsapp:" prefix if present
        if phone_number.startswith("whatsapp:"):
            phone_number = phone_number[9:]
            
        logger.info(f"Processing message from {phone_number}: {message_text}")
        
        # Retrieve conversation history
        conversation, is_first_time, last_timestamp = await get_conversation_history(phone_number)
        
        # Check for commands

        # Check for typos in code command
        if message_text.startswith("/code") and not message_text.startswith("/code "):
            # This matches things like "/codeyHCK0" or "/codeABC123" without a space
            
            # Extract what appears to be the code
            code_part = message_text[5:]  # Remove "/code" prefix
            
            # Send correction message
            await send_whatsapp_message(
                phone_number, 
                f"Wrong /code command format. I believe you meant:\n\n/code {code_part}\n\nPlease try again with a space after '/code'."
            )
            return

        elif message_text.lower().strip() == "/reset":
            # Clear conversation and confirm
            conversation = []
            await store_conversation_history(phone_number, conversation)
            await send_whatsapp_message(phone_number, "New Conversation.")
            return
        
        elif message_text.lower().strip() == "/help":
            # Send help message
            await send_whatsapp_message(phone_number, HELP_MESSAGE)
            # Don't clear conversation, just add this exchange
            conversation.append({"role": "user", "content": message_text})
            conversation.append({"role": "assistant", "content": HELP_MESSAGE})
            conversation[-1]["timestamp"] = time.time()
            await store_conversation_history(phone_number, conversation)
            return
        
        # Send invite message
        elif message_text.lower().strip() == "/invite":
            # Send invite message
            await send_whatsapp_message(phone_number, WHATSAPP_LINK_MESSAGE)
            return

        # Store affiliate program code
        elif message_text.lower().strip().startswith("/code "):
            # Extract the affiliate code
            affiliate_code = message_text.strip()[6:].strip()  # Remove "/code " and any extra spaces
            
            # Store the code in user metadata in Redis
            if not conversation:
                conversation = []
            
            # Add referred_by field to track the affiliate
            conversation.append({
                "role": "system", 
                "content": f"User was referred by affiliate code: {affiliate_code}",
                "timestamp": time.time(),
                "referred_by": affiliate_code
            })
            
            # Store the conversation with the affiliation tracking
            await store_conversation_history(phone_number, conversation)
            
            # Send acknowledgment
            await send_whatsapp_message(phone_number, f"Thank you! You've been connected through invitation code: {affiliate_code}")
            return
        
        # Check for inactivity reset
        if await should_reset_conversation(last_timestamp):
            # Reset conversation due to inactivity
            conversation = []
            await send_whatsapp_message(
                phone_number, 
                "Starting a new conversation due to inactivity."
            )
        
        # Send welcome message for first-time users
        if is_first_time:
            await send_whatsapp_message(phone_number, WELCOME_MESSAGE)
            # Add welcome message to conversation history
            conversation.append({"role": "assistant", "content": WELCOME_MESSAGE, "timestamp": time.time()})
            await store_conversation_history(phone_number, conversation)
        
        # Add user message to history with timestamp
        conversation.append({
            "role": "user", 
            "content": message_text,
            "timestamp": time.time()
        })
        
        # Store user metadata if available
        if profile_name:
            # Add user metadata to the latest message
            conversation[-1]["user_name"] = profile_name
        
        # Create a chat request
        chat_request = ChatRequest(
            messages=[Message(role=msg["role"], content=msg["content"]) for msg in conversation if "role" in msg and "content" in msg],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,  # Shorter for WhatsApp
            stream=False,
            user_id=phone_number # Use phone number as user ID
        )
        
        # Call the API to get the response
        try:
            response = await post_with_retry(
                "https://isaac-lite-r1-045fb83e2eb2.herokuapp.com/v1/chat/completions",
                chat_request.dict()
            )
        except httpx.ReadTimeout:
            logger.error("Request to chat completions timed out")
            await send_whatsapp_message(
                phone_number, 
                "I'm taking longer than expected to process your message. Please try a shorter or simpler question."
            )
            return
            
        if response.status_code != 200:
            error_msg = f"Error getting AI response: {response.status_code}"
            logger.error(error_msg)
            await send_whatsapp_message(phone_number, "Sorry, I'm having trouble processing your request right now.")
            return
            
        # Parse the JSON response
        try:
            result = response.json()
            logger.info(f"Received API response: {json.dumps(result)[:200]}...")
        except json.JSONDecodeError:
            logger.error("Failed to parse API response as JSON")
            await send_whatsapp_message(phone_number, "Sorry, I received an invalid response from the API.")
            return
            
        # Get the assistant's response with defensive checks
        assistant_response = ""
        if (result and isinstance(result, dict) and 
            "choices" in result and result["choices"] and 
            isinstance(result["choices"], list) and 
            "message" in result["choices"][0] and 
            "content" in result["choices"][0]["message"]):
            
            assistant_response = result["choices"][0]["message"]["content"]
            
            # Check for tool calls
            message = result["choices"][0]["message"]
            if (isinstance(message, dict) and 
                "tool_calls" in message and 
                message["tool_calls"] is not None and 
                isinstance(message["tool_calls"], list)):
                
                tool_calls = message["tool_calls"]
                for tool_call in tool_calls:
                    if ("function" in tool_call and 
                        isinstance(tool_call["function"], dict) and 
                        "name" in tool_call["function"]):
                        
                        if tool_call["function"]["name"] == "get_information":
                            try:
                                args = json.loads(tool_call["function"]["arguments"])
                                info_result = await execute_get_information(args)
                                assistant_response = format_function_result_for_whatsapp(info_result)
                            except Exception as e:
                                logger.error(f"Error formatting get_information result: {e}")
                        
                        elif tool_call["function"]["name"] == "retrieve_record":
                            assistant_response = "I'm sorry, patient record access is not available via WhatsApp for security reasons."
        
        # If we couldn't extract a response for some reason
        if not assistant_response:
            assistant_response = "I'm not sure how to respond to that."
        
        # Clean the response for WhatsApp
        assistant_response = strip_html(assistant_response)
        
        # Add assistant response to conversation history with timestamp
        conversation.append({
            "role": "assistant", 
            "content": assistant_response,
            "timestamp": time.time()
        })
        
        # Store updated conversation in Redis
        await store_conversation_history(phone_number, conversation)
        
        # Split the message if needed and send via WhatsApp
        if len(assistant_response) > 1000:
            chunks = split_long_message(assistant_response)
            for chunk in chunks:
                await send_whatsapp_message(phone_number, chunk)
        else:
            await send_whatsapp_message(phone_number, assistant_response)
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            await send_whatsapp_message(phone_number, "I encountered an error processing your request.")
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}")

# Session cleanup
async def cleanup_old_sessions():
    """Periodically clean up inactive sessions"""
    while True:
        try:
            # Wait for 1 hour before cleaning up
            await asyncio.sleep(3600)
            
            # Find sessions older than 24 hours
            current_time = time.time()
            to_remove = []
            
            for phone, session in user_sessions.items():
                # Check last activity time
                if "last_activity" in session and current_time - session["last_activity"] > 86400:
                    to_remove.append(phone)
            
            # Remove inactive sessions
            for phone in to_remove:
                del user_sessions[phone]
                
            logger.info(f"Cleaned up {len(to_remove)} inactive sessions")
                
        except Exception as e:
            logger.error(f"Error in session cleanup: {e}")


# Debug endpoint to check current user sessions
# Usage: curl -k -X GET https://isaac-lite-r1-045fb83e2eb2.herokuapp.com/whatsapp/debug/sessions
@router.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to check current user sessions"""
    return {
        "session_count": len(user_sessions),
        "session_users": list(user_sessions.keys()),
        "sample": {k: v[:2] for k, v in list(user_sessions.items())[:2]} if user_sessions else {}
    }

# Endpoint to trigger the migration
# Usage: curl -k -X POST https://isaac-lite-r1-045fb83e2eb2.herokuapp.com/whatsapp/migrate -H "Authorization: Bearer migration_secret_key"
@router.post("/migrate")
async def migrate_sessions(request: Request):
    """Endpoint to trigger migration of in-memory sessions to Redis"""
    # Only allow this from localhost or internal network for security
    client_host = request.client.host
    if client_host not in ["127.0.0.1", "localhost"]:
        # Check if it's an internal request
        auth_header = request.headers.get("Authorization")
        if auth_header != f"Bearer {os.getenv('MIGRATION_SECRET', 'migration_secret_key')}":
            raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Run the migration
    result = await migrate_sessions_to_redis()
    return result

@router.get("/webhook")
async def verify_webhook(request: Request):
    """Verify webhook for WhatsApp API setup"""
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    
    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            logger.info("Webhook verified")
            return int(challenge) if challenge else "Webhook verified"
        else:
            logger.error("Webhook verification failed")
            raise HTTPException(status_code=403, detail="Verification failed")
    
    return "Hello, this is the WhatsApp webhook endpoint."

@router.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming messages from WhatsApp"""
    try:
        # First, try to get the content type
        content_type = request.headers.get("content-type", "")
        logger.info(f"Received webhook with content type: {content_type}")
        
        # Try to parse as JSON for Meta/Facebook format
        if "application/json" in content_type:
            try:
                body = await request.json()
                logger.info(f"Received JSON webhook: {json.dumps(body)[:200]}...")
                
                # Process Meta/Facebook format
                if "object" in body and body["object"] == "whatsapp_business_account":
                    # Process each entry
                    for entry in body.get("entry", []):
                        # Process each change in the entry
                        for change in entry.get("changes", []):
                            if change.get("field") == "messages":
                                # Process each message
                                for message in change.get("value", {}).get("messages", []):
                                    sender = message.get("from")
                                    message_type = message.get("type")
                                    
                                    # Try to get profile name
                                    contacts = change.get("value", {}).get("contacts", [])
                                    profile_name = None
                                    if contacts and len(contacts) > 0:
                                        profile_name = contacts[0].get("profile", {}).get("name")
                                    
                                    if message_type == "text":
                                        text = message.get("text", {}).get("body", "")
                                        background_tasks.add_task(process_user_message, sender, text, profile_name)
                                    elif message_type in ["image", "audio", "document", "video"]:
                                        background_tasks.add_task(handle_media_message, sender, message)
            except json.JSONDecodeError:
                logger.warning("Could not parse request body as JSON")
        
        # If it's not JSON or parsing failed, try form data (Twilio format)
        if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            form_data = await request.form()
            logger.info(f"Received form data: {dict(form_data)}")
            
            # For Twilio, the relevant fields are From, Body, and ProfileName
            if "From" in form_data and "Body" in form_data:
                sender = form_data["From"]
                text = form_data["Body"]
                profile_name = form_data.get("ProfileName")
                
                logger.info(f"Processing Twilio message from {sender}: {text}")
                background_tasks.add_task(process_user_message, sender, text, profile_name)
        
        # For debugging in case of unknown format
        else:
            # Log the raw request body for debugging
            body_text = await request.body()
            logger.info(f"Raw request body: {body_text[:200]}...")
        
        # Return a 200 OK to acknowledge receipt
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        # Even with an error, return 200 OK to acknowledge receipt
        # This prevents WhatsApp from retrying the message repeatedly
        return {"status": "error", "message": str(e)}