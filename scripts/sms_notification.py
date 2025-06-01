import os
from twilio.rest import Client
from dotenv import load_dotenv

class SMSSender:
    def __init__(self, credentials_dict=None):
        if credentials_dict:
            self.account_sid = credentials_dict['TWILIO_ACCOUNT_SID']
            self.auth_token = credentials_dict['TWILIO_AUTH_TOKEN']
            self.from_number = credentials_dict['TWILIO_FROM_NUMBER']
        else:
            load_dotenv()
            self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            self.from_number = '+19253973469'
        
        if not all([self.account_sid, self.auth_token, self.from_number]):
            raise ValueError("Twilio credentials not found")
            
        self.client = Client(self.account_sid, self.auth_token)
    
    def _format_phone(self, phone: str) -> str:
        """Format phone number to E.164 format."""
        digits = ''.join(filter(str.isdigit, phone))
        
        if len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) == 11 and digits.startswith('1'):
            return f"+{digits}"
        raise ValueError(f"Invalid phone number format: {phone}")
    
    def send_sms(self, to_phone: str, message: str) -> dict:
        """
        Send a simple SMS message.
        
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