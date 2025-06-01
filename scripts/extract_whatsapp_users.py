#!/usr/bin/env python3
import json
import csv
import re
import sys
import os
from typing import Dict, List, Tuple, Set, Optional
import argparse

def extract_phone_from_key(key: str) -> Optional[str]:
    """
    Extract phone number from Redis key.
    Handles formats like "conversation:+96878686876" or "conversation:whatsapp:+919000796210"
    """
    # Handle both formats
    if "whatsapp:" in key:
        match = re.search(r'whatsapp:(\+\d+)', key)
    else:
        match = re.search(r'conversation:(\+\d+)', key)

    if match:
        return match.group(1)
    return None

def extract_username_from_messages(messages: List[Dict]) -> Optional[str]:
    """
    Extract user_name from message objects if available
    """
    for message in messages:
        if message.get("role") == "user" and "user_name" in message:
            return message.get("user_name")
    return None

def process_redis_dump(file_path: str, output_path: str) -> None:
    """
    Process Redis dump JSON file and extract phone numbers and usernames
    """
    # Dictionary to store phone numbers and usernames
    phone_users: Dict[str, str] = {}
    
    try:
        print(f"Reading JSON dump from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process keys
        keys = data.get("keys", [])
        data_dict = data.get("data", {})
        
        print(f"Found {len(keys)} keys in dump file")
        
        # Counter for progress tracking
        counter = 0
        conversation_keys = [k for k in keys if "conversation:" in k]
        print(f"Found {len(conversation_keys)} conversation keys")
        
        for key in conversation_keys:
            counter += 1
            if counter % 100 == 0:
                print(f"Processed {counter}/{len(conversation_keys)} conversation keys")
                
            phone = extract_phone_from_key(key)
            if not phone:
                continue
                
            # Get data for this key
            key_data = data_dict.get(key, {})
            
            if key_data.get("type") == "string":
                value = key_data.get("value", [])
                if isinstance(value, list):
                    username = extract_username_from_messages(value)
                    if username:
                        phone_users[phone] = username
                    else:
                        # Only add phone without username if we don't already have it
                        if phone not in phone_users:
                            phone_users[phone] = ""
        
        # Write results to CSV
        print(f"Writing {len(phone_users)} unique phone numbers to {output_path}")
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Phone Number", "User Name"])
            
            for phone, username in phone_users.items():
                writer.writerow([phone, username])
                
        print(f"Successfully exported data to {output_path}")
        print(f"Total unique phone numbers: {len(phone_users)}")
        print(f"Phone numbers with usernames: {sum(1 for username in phone_users.values() if username)}")
        
    except Exception as e:
        print(f"Error processing Redis dump: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Extract WhatsApp phone numbers and usernames from Redis dump')
    parser.add_argument('input_file', help='Path to the Redis dump JSON file')
    parser.add_argument('-o', '--output', help='Path to the output CSV file (default: whatsapp_users.csv)', 
                        default='whatsapp_users.csv')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    process_redis_dump(args.input_file, args.output)

if __name__ == "__main__":
    main()
