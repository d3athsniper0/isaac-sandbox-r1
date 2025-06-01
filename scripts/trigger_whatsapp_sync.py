#!/usr/bin/env python
"""
WhatsApp Conversation Sync API Client

This script triggers a WhatsApp conversation sync through the API endpoint.
It can be used to manually initiate a sync from a remote machine
or scheduled as a cron job.

Usage:
  python trigger_whatsapp_sync.py           # Sync only new messages (incremental)
  python trigger_whatsapp_sync.py --full    # Sync all messages (full sync)
"""

import os
import sys
import json
import argparse
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def trigger_sync(base_url, auth_key, full_sync=False):
    """
    Trigger a WhatsApp conversation sync via API
    
    Args:
        base_url (str): Base URL of the API (e.g., 'https://your-app.herokuapp.com')
        auth_key (str): Authentication key for the endpoint
        full_sync (bool): Whether to sync all messages or just new ones
    
    Returns:
        dict: API response
    """
    # Construct the full URL - note the updated path!
    url = f"{base_url.rstrip('/')}/whatsapp-sync/sync"
    
    # Set up headers
    headers = {
        "X-Auth-Key": auth_key,
        "Content-Type": "application/json"
    }
    
    # Create payload
    payload = {
        "full_sync": full_sync
    }
    
    print(f"Triggering WhatsApp sync at {url} (full_sync={full_sync})...")
    
    try:
        # Make the API request
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30  # Timeout after 30 seconds
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse and return response
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {str(e)}")
        if hasattr(e, 'response') and e.response:
            print(f"Status code: {e.response.status_code}")
            try:
                print(f"Response: {e.response.json()}")
            except:
                print(f"Response text: {e.response.text}")
        return None

def main():
    """Main function to parse arguments and trigger the sync"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trigger a WhatsApp conversation sync via API")
    parser.add_argument("--full", action="store_true", help="Sync all messages instead of just new ones")
    parser.add_argument("--url", help="Base URL of the API (e.g., 'https://your-app.herokuapp.com')")
    parser.add_argument("--key", help="Authentication key for the endpoint")
    args = parser.parse_args()
    
    # Get base URL from args or environment
    base_url = args.url or os.getenv("WHATSAPP_SYNC_API_URL")
    if not base_url:
        print("Error: API URL not provided. Use --url option or set WHATSAPP_SYNC_API_URL environment variable.")
        sys.exit(1)
    
    # Get auth key from args or environment
    auth_key = args.key or os.getenv("WHATSAPP_SYNC_AUTH_KEY")
    if not auth_key:
        print("Error: Authentication key not provided. Use --key option or set WHATSAPP_SYNC_AUTH_KEY environment variable.")
        sys.exit(1)
    
    # Trigger the sync
    result = trigger_sync(base_url, auth_key, full_sync=args.full)
    
    # Display results
    if result:
        print("\nSync triggered successfully!")
        print(f"Success: {result.get('success', False)}")
        print(f"Message: {result.get('message', 'No message')}")
        
        # If results are included in the response
        if 'results' in result:
            stats = result['results']
            print("\nSync Statistics:")
            print(f"- New conversations: {stats.get('new', 0)}")
            print(f"- Updated conversations: {stats.get('updated', 0)}")
            print(f"- Skipped conversations: {stats.get('skipped', 0)}")
            print(f"- Errors: {stats.get('errors', 0)}")
            print(f"- Total: {stats.get('total', 0)}")
    else:
        print("Failed to trigger sync.")
        sys.exit(1)

if __name__ == "__main__":
    main()