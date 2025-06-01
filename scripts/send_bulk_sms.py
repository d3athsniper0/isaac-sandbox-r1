#!/usr/bin/env python3
import os
import sys
import argparse
import csv
import time
from dotenv import load_dotenv
from typing import List, Dict

class ImprovedSMSSender:
    def __init__(self, credentials_dict=None):
        """Initialize with either provided credentials or from environment variables"""
        try:
            from twilio.rest import Client
        except ImportError:
            print("Error: Twilio package not installed. Install it with 'pip install twilio'")
            sys.exit(1)
            
        if credentials_dict:
            self.account_sid = credentials_dict['TWILIO_ACCOUNT_SID']
            self.auth_token = credentials_dict['TWILIO_AUTH_TOKEN']
            self.from_number = credentials_dict['TWILIO_FROM_NUMBER']
        else:
            load_dotenv()
            self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            self.from_number = os.getenv('TWILIO_FROM_NUMBER', '+19253973469')
        
        if not all([self.account_sid, self.auth_token, self.from_number]):
            raise ValueError("Twilio credentials not found")
            
        self.client = Client(self.account_sid, self.auth_token)
    
    def _format_phone(self, phone: str) -> str:
        """
        Format phone number to E.164 format.
        Improved to handle international numbers correctly.
        """
        # Strip any non-digit characters except the leading +
        if phone.startswith('+'):
            # Keep the + but remove any other non-digits
            digits = '+' + ''.join(filter(str.isdigit, phone[1:]))
        else:
            # Remove any non-digits
            digits = ''.join(filter(str.isdigit, phone))
            # If it's a 10-digit US number, add +1
            if len(digits) == 10:
                digits = f"+1{digits}"
            # If it starts with 1 and is 11 digits, add +
            elif len(digits) == 11 and digits.startswith('1'):
                digits = f"+{digits}"
            # Otherwise, it's probably an international number without +
            else:
                digits = f"+{digits}"
        
        return digits
    
    def send_sms(self, to_phone: str, message: str) -> dict:
        """
        Send an SMS message.
        
        Args:
            to_phone: Recipient's phone number
            message: Message content
            
        Returns:
            dict: Response containing success status and details
        """
        try:
            to_number = self._format_phone(to_phone)
            
            response = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            
            return {
                "success": True,
                "message_sid": response.sid,
                "details": {
                    "recipient": to_number
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

def read_recipients(csv_path: str) -> List[Dict[str, str]]:
    """Read recipient data from CSV file"""
    recipients = []
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check for required column
            if not reader.fieldnames or "Phone Number" not in reader.fieldnames:
                raise ValueError(f"CSV file {csv_path} missing required 'Phone Number' column")
            
            for row in reader:
                phone = row.get("Phone Number", "").strip()
                username = row.get("User Name", "").strip()
                
                if phone:  # Only add if phone number is not empty
                    recipients.append({
                        "phone": phone,
                        "name": username
                    })
    
    except Exception as e:
        print(f"Error reading {csv_path}: {str(e)}")
        sys.exit(1)
        
    return recipients

def read_message_template(file_path: str) -> str:
    """Read message content from a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading message file {file_path}: {str(e)}")
        sys.exit(1)

def send_bulk_messages(csv_file: str, message_file: str, dry_run: bool = False, delay: float = 1.0):
    """Send messages to all recipients in the CSV file"""
    # Load recipients and message
    recipients = read_recipients(csv_file)
    message_template = read_message_template(message_file)
    
    print(f"Loaded {len(recipients)} recipients from {csv_file}")
    print(f"Message template ({len(message_template)} characters):")
    print("-" * 50)
    print(message_template)
    print("-" * 50)
    
    # Initialize counters
    success_count = 0
    failure_count = 0
    
    # Confirm before sending
    if not dry_run:
        confirm = input(f"You're about to send messages to {len(recipients)} recipients. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
    
    # Initialize SMS sender
    try:
        sms_sender = ImprovedSMSSender()
    except ValueError as e:
        print(f"Error initializing Twilio client: {e}")
        print("Make sure TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER are set in your environment")
        sys.exit(1)
    
    # Send messages
    print(f"Preparing to send messages to {len(recipients)} recipients")
    
    if dry_run:
        print("DRY RUN MODE: No messages will actually be sent")
    
    for i, recipient in enumerate(recipients, 1):
        phone = recipient["phone"]
        name = recipient["name"] or "there"
        
        # Personalize message if {name} placeholder exists
        personalized_message = message_template.replace("{name}", name)
        
        print(f"[{i}/{len(recipients)}] Sending to {phone}...", end="", flush=True)
        
        if dry_run:
            print(" SKIPPED (dry run)")
            print(f"    Would send: {personalized_message[:50]}..." if len(personalized_message) > 50 else personalized_message)
            success_count += 1
            continue
        
        # Actually send the message
        try:
            result = sms_sender.send_sms(phone, personalized_message)
            
            if result["success"]:
                print(" SUCCESS")
                success_count += 1
            else:
                print(f" FAILED: {result.get('error', 'Unknown error')}")
                failure_count += 1
            
            # Sleep to avoid rate limiting
            time.sleep(delay)
                
        except Exception as e:
            print(f" ERROR: {str(e)}")
            failure_count += 1
    
    # Print summary
    print("\nSending complete!")
    print(f"Successfully sent: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Total: {len(recipients)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Send SMS messages to recipients in a CSV file')
    parser.add_argument('csv_file', help='Path to CSV file containing recipient phone numbers')
    parser.add_argument('message_file', help='Path to text file containing message to send')
    parser.add_argument('--dry-run', action='store_true', help='Run without actually sending messages')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between messages in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' does not exist")
        sys.exit(1)
        
    if not os.path.exists(args.message_file):
        print(f"Error: Message file '{args.message_file}' does not exist")
        sys.exit(1)
    
    # Send messages
    send_bulk_messages(args.csv_file, args.message_file, args.dry_run, args.delay)

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    main()
