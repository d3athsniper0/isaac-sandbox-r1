# By default, this script will only fetch new conversations.
# To fetch all conversations regardless of previous syncs, use the --full_sync flag.
# Example: python whatsapp_syncer.py --full_sync

import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import redis
import ssl
from urllib.parse import urlparse
import logging
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class WhatsAppSyncer:
    """Class to sync WhatsApp conversations from Redis to PostgreSQL"""
    
    def __init__(self):
        # Initialize Redis connection for conversations
        self.redis_client = self._get_redis_client()
        
        # Initialize Redis connection for caller info (could be the same Redis instance)
        self.caller_redis_client = self._get_redis_client(is_caller_redis=True)
        
        # Initialize PostgreSQL connection
        self.db_conn = self._get_db_connection()
    
    def _get_db_connection(self):
        """Create a connection to the PostgreSQL database"""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL environment variable is required")
            
        # Add sslmode=require parameter if not already in the URL
        if 'sslmode=' not in db_url:
            if '?' in db_url:
                db_url += "&sslmode=require"
            else:
                db_url += "?sslmode=require"
        
        conn = psycopg2.connect(
            db_url,
            cursor_factory=RealDictCursor
        )
        return conn
    
    def _get_redis_client(self, is_caller_redis=False):
        """Initialize Redis client from environment variable"""
        if is_caller_redis:
            redis_url = os.getenv("CALLER_REDIS_URL", os.getenv("REDIS_URL"))
        else:
            redis_url = os.getenv("REDIS_URL")
            
        if not redis_url:
            logger.warning("REDIS_URL environment variable not set. Cannot fetch conversations.")
            return None
        
        # Parse Redis URL to check if it's using rediss:// (TLS/SSL)
        parsed_url = urlparse(redis_url)
        if parsed_url.scheme == 'rediss':
            # For SSL connections, disable cert validation if needed
            return redis.from_url(redis_url, ssl_cert_reqs=ssl.CERT_NONE)
        else:
            return redis.from_url(redis_url)
    
    def _get_caller_name(self, phone_number):
        """Get caller name from Redis if available"""
        if not self.caller_redis_client or not phone_number:
            return None
        
        clean_phone = phone_number
        # Remove 'whatsapp:' prefix if present
        if clean_phone.startswith("whatsapp:"):
            clean_phone = clean_phone[9:]
        
        try:
            # Check if number exists in Redis
            if self.caller_redis_client.exists(clean_phone):
                caller_info = self.caller_redis_client.get(clean_phone)
                info_str = caller_info.decode('utf-8') if isinstance(caller_info, bytes) else caller_info
                info = json.loads(info_str)
                return info.get("name", None)
        except Exception as e:
            logger.error(f"Error getting caller name from Redis: {str(e)}")
        
        return None
    
    def setup_database(self):
        """Create necessary tables if they don't exist"""
        cursor = self.db_conn.cursor()
        
        # Create the whatsapp table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS whatsapp (
            id SERIAL PRIMARY KEY,
            phone_number TEXT UNIQUE NOT NULL,
            user_name TEXT,
            message_count INTEGER,
            last_message_timestamp TIMESTAMP,
            conversations JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Index for faster lookups by phone number
        CREATE INDEX IF NOT EXISTS whatsapp_phone_number_idx ON whatsapp(phone_number);
        """)
        
        self.db_conn.commit()
        cursor.close()
        logger.info("Database setup complete.")
    
    def conversation_exists(self, phone_number):
        """Check if a conversation already exists in the database for this phone number"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT id, last_message_timestamp FROM whatsapp WHERE phone_number = %s", (phone_number,))
        result = cursor.fetchone()
        cursor.close()
        return result
    
    def get_all_whatsapp_keys(self):
        """Get all Redis keys that contain WhatsApp conversations"""
        if not self.redis_client:
            logger.error("Redis client is not initialized")
            return []
        
        # Get all keys matching WhatsApp conversation patterns
        whatsapp_keys = []
        
        # Pattern 1: "conversation:whatsapp:+XXXXXXXXXX"
        keys = self.redis_client.keys("conversation:whatsapp:*")
        whatsapp_keys.extend([key.decode('utf-8') if isinstance(key, bytes) else key for key in keys])
        
        # Pattern 2: "conversation:+XXXXXXXXXX" (we need to filter out non-WhatsApp ones)
        keys = self.redis_client.keys("conversation:+*")
        # Convert bytes to string if needed
        keys = [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
        # Filter out the ones that are already captured by the whatsapp prefix
        keys = [key for key in keys if not key.replace("conversation:", "conversation:whatsapp:") in whatsapp_keys]
        whatsapp_keys.extend(keys)
        
        logger.info(f"Found {len(whatsapp_keys)} WhatsApp conversation keys in Redis")
        return whatsapp_keys
    
    def extract_phone_number(self, redis_key):
        """Extract phone number from Redis key"""
        # For keys like "conversation:whatsapp:+XXXXXXXXXX"
        if redis_key.startswith("conversation:whatsapp:"):
            return redis_key.replace("conversation:whatsapp:", "")
        # For keys like "conversation:+XXXXXXXXXX"
        elif redis_key.startswith("conversation:"):
            return redis_key.replace("conversation:", "")
        return None
    
    def extract_user_name_from_conversations(self, conversations):
        """Extract user name from conversation data if available"""
        if not conversations:
            return None
        
        # Look for user_name in user messages
        for message in conversations:
            if message.get("role") == "user" and "user_name" in message:
                return message.get("user_name")
        
        return None
    
    def filter_conversations(self, conversations, last_timestamp=None):
        """
        Filter conversation data:
        - Remove system messages
        - Filter by timestamp if provided
        """
        if not conversations:
            return []
        
        # Convert timestamp to float if it's a datetime
        if last_timestamp and isinstance(last_timestamp, datetime):
            last_timestamp = last_timestamp.timestamp()
        
        filtered = []
        for message in conversations:
            # Skip system messages
            if message.get("role") == "system":
                continue
                
            # Skip messages older than last_timestamp if provided
            if last_timestamp and "timestamp" in message and message["timestamp"] <= last_timestamp:
                continue
                
            filtered.append(message)
            
        return filtered
    
    def get_conversation_data(self, redis_key, last_timestamp=None):
        """
        Get and process conversation data from Redis
        
        Args:
            redis_key: Redis key for the conversation
            last_timestamp: Only fetch messages newer than this timestamp
        """
        try:
            # Get raw conversation data from Redis
            raw_data = self.redis_client.get(redis_key)
            if not raw_data:
                logger.warning(f"No data found for key: {redis_key}")
                return None
            
            # Parse the data
            data_str = raw_data.decode('utf-8') if isinstance(raw_data, bytes) else raw_data
            conversations = json.loads(data_str)
            
            # Filter the conversations
            filtered_conversations = self.filter_conversations(conversations, last_timestamp)
            
            # Extract metadata
            phone_number = self.extract_phone_number(redis_key)
            user_name = self.extract_user_name_from_conversations(conversations)
            
            # If user_name not found in conversations, try to get it from caller info
            if not user_name:
                user_name = self._get_caller_name(phone_number)
            
            # Calculate message count and last timestamp
            message_count = len(filtered_conversations)
            last_message_timestamp = None
            
            if filtered_conversations:
                # Find the latest timestamp
                for message in reversed(filtered_conversations):
                    if "timestamp" in message:
                        last_message_timestamp = datetime.fromtimestamp(message["timestamp"])
                        break
            
            return {
                "phone_number": phone_number,
                "user_name": user_name,
                "conversations": filtered_conversations,
                "message_count": message_count,
                "last_message_timestamp": last_message_timestamp
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation data for {redis_key}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def insert_or_update_conversation(self, conversation_data):
        """Insert or update a conversation in the database"""
        if not conversation_data:
            return False
        
        try:
            phone_number = conversation_data["phone_number"]
            user_name = conversation_data["user_name"]
            message_count = conversation_data["message_count"]
            last_message_timestamp = conversation_data["last_message_timestamp"]
            conversations = conversation_data["conversations"]
            
            cursor = self.db_conn.cursor()
            
            # Check if conversation exists
            cursor.execute(
                "SELECT id, message_count, conversations FROM whatsapp WHERE phone_number = %s", 
                (phone_number,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing conversation
                existing_conversations = existing["conversations"]
                # Append new messages to existing conversations
                if existing_conversations:
                    # Merge conversations
                    all_conversations = existing_conversations + conversations
                else:
                    all_conversations = conversations
                
                cursor.execute("""
                UPDATE whatsapp SET
                    user_name = COALESCE(%s, user_name),
                    message_count = %s,
                    last_message_timestamp = %s,
                    conversations = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE phone_number = %s
                """, (
                    user_name,
                    message_count + (existing["message_count"] or 0),
                    last_message_timestamp,
                    Json(all_conversations),
                    phone_number
                ))
                result = "updated"
            else:
                # Insert new conversation
                cursor.execute("""
                INSERT INTO whatsapp (
                    phone_number, user_name, message_count,
                    last_message_timestamp, conversations
                ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    phone_number,
                    user_name,
                    message_count,
                    last_message_timestamp,
                    Json(conversations)
                ))
                result = "inserted"
            
            self.db_conn.commit()
            cursor.close()
            
            return result
            
        except Exception as e:
            logger.error(f"Error inserting/updating conversation for {conversation_data.get('phone_number')}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.db_conn.rollback()
            return False
    
    def sync_conversations(self, full_sync=False):
        """
        Sync WhatsApp conversations from Redis to PostgreSQL
        
        Args:
            full_sync: If True, sync all conversations regardless of previous syncs
        """
        logger.info(f"Starting WhatsApp conversation sync (full_sync={full_sync})...")
        
        # Get all WhatsApp conversation keys from Redis
        redis_keys = self.get_all_whatsapp_keys()
        
        # Track metrics
        new_count = 0
        updated_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process each conversation
        for idx, redis_key in tqdm(enumerate(redis_keys), total=len(redis_keys), desc="Processing WhatsApp conversations"):
            try:
                # Extract phone number from the Redis key
                phone_number = self.extract_phone_number(redis_key)
                if not phone_number:
                    logger.warning(f"Could not extract phone number from key: {redis_key}")
                    skipped_count += 1
                    continue
                
                # Check if conversation already exists to determine last timestamp
                last_timestamp = None
                if not full_sync:
                    existing = self.conversation_exists(phone_number)
                    if existing and existing["last_message_timestamp"]:
                        last_timestamp = existing["last_message_timestamp"]
                
                # Get conversation data from Redis
                conversation_data = self.get_conversation_data(redis_key, last_timestamp)
                
                # If no new messages, skip
                if not conversation_data or not conversation_data["conversations"]:
                    logger.info(f"No new messages for {phone_number}, skipping.")
                    skipped_count += 1
                    continue
                
                # Insert or update in database
                result = self.insert_or_update_conversation(conversation_data)
                
                if result == "inserted":
                    logger.info(f"Inserted new conversation for {phone_number} with {conversation_data['message_count']} messages")
                    new_count += 1
                elif result == "updated":
                    logger.info(f"Updated conversation for {phone_number} with {conversation_data['message_count']} new messages")
                    updated_count += 1
                else:
                    logger.error(f"Error processing conversation for {phone_number}")
                    error_count += 1
                
            except Exception as e:
                logger.error(f"Error processing Redis key {redis_key}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                error_count += 1
        
        logger.info(f"\nSync complete. Summary:")
        logger.info(f"  - New conversations: {new_count}")
        logger.info(f"  - Updated conversations: {updated_count}")
        logger.info(f"  - Skipped conversations: {skipped_count}")
        logger.info(f"  - Errors: {error_count}")
        
        return {
            "new": new_count,
            "updated": updated_count,
            "skipped": skipped_count,
            "errors": error_count,
            "total": len(redis_keys)
        }
    
    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()

# Main execution
if __name__ == "__main__":
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Sync WhatsApp conversations from Redis to PostgreSQL")
        parser.add_argument("--full_sync", action="store_true", help="Sync all conversations regardless of previous syncs")
        args = parser.parse_args()
        
        logger.info(f"Starting WhatsApp conversation sync at {datetime.now()}")
        syncer = WhatsAppSyncer()
        
        # Setup the database
        syncer.setup_database()
        
        # Sync conversations
        results = syncer.sync_conversations(full_sync=args.full_sync)
        
        # Close connections
        syncer.close()
        
        logger.info(f"Sync completed at {datetime.now()}")
        logger.info(f"Summary: {results}")
        
    except Exception as e:
        logger.error(f"Error in WhatsApp conversation sync: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
