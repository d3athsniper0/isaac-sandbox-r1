# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API settings
#API_BASE_URL = os.getenv("API_BASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# LLM settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")

# Function settings
FUNCTION_TIMEOUT = int(os.getenv("FUNCTION_TIMEOUT", "15"))  # seconds
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# Redis settings (for caching)
REDIS_URL = os.getenv("REDIS_URL", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", "1800"))  # 30 minutes

# System settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
WORKER_CONNECTIONS = int(os.getenv("WORKER_CONNECTIONS", "100"))

# System prompt
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "dental_assistant_prompt.txt")
SYSTEM_PROMPT = ""
if os.path.exists(SYSTEM_PROMPT_PATH):
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    SYSTEM_PROMPT = "You are Isaac, the Trust AI Dental Assistant..."  # Default fallback

# Function definitions
# Define the function specifications for the OpenAI functions API
FUNCTION_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "generate_treatment_plan",
            "description": "Generate a dental treatment plan based on patient information",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string", "description": "Patient's full name"},
                    "age": {"type": "string", "description": "Patient's age"},
                    "chief_complaint": {"type": "string", "description": "Patient's main concern or reason for visit"},
                    "medical_history": {"type": "string", "description": "Relevant medical conditions"},
                    "dental_history": {"type": "string", "description": "Relevant dental conditions and history"},
                    "current_medications": {"type": "string", "description": "Current medications being taken by patient"},
                    "xray_findings": {"type": "string", "description": "Findings from dental x-rays"},
                    "budget_constraint": {"type": "string", "description": "Patient's budget limitations if any"},
                    "time_constraint": {"type": "string", "description": "Any time constraints for treatment completion"},
                    "insurance_info": {"type": "string", "description": "Insurance coverage details"},
                    "additional_info": {"type": "string", "description": "Any additional relevant information"},
                    "notification": {
                        "type": "object",
                        "properties": {
                            "wants_sms": {"type": "boolean", "description": "Whether the patient wants SMS notifications"},
                            "phone_number": {"type": "string", "description": "Patient's phone number for notifications"}
                        }
                    }
                },
                "required": ["patient_name", "age", "chief_complaint", "medical_history", "dental_history"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_record",
            "description": "Retrieves patient records from Pinecone using various search criteria",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_type": {
                        "type": "string",
                        "enum": ["id", "patient", "patient_name", "text", "medication", "condition"],
                        "description": "Type of search to perform: id (File_ID), patient (Patient_Case_ID), patient_name, text (semantic search), medication, or condition"
                    },
                    "query": {
                        "type": "string",
                        "description": "The search term (e.g., file ID, patient name, or search text)"
                    },
                    "practice_id": {
                        "type": "string",
                        "description": "Optional practice ID to filter results"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)"
                    },
                    "index_name": {
                        "type": "string",
                        "description": "Optional index name to filter results"
                    }
                },
                "required": ["search_type", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_recipient",
            "description": "Verify a recipient using their 4-digit verification code and retrieve their contact information",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "4-digit verification code to identify the recipient"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Send an email or SMS notification to a verified recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "notification_type": {"type": "string", "enum": ["email", "sms"], "description": "Type of notification to send"},
                    "to": {"type": "string", "description": "Recipient email or phone number"},
                    "content": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string", "description": "Email subject (required for email)"},
                            "body": {"type": "string", "description": "Message content"}
                        },
                        "required": ["body"]
                    }
                },
                "required": ["notification_type", "to", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_information",
            "description": "Get information from external sources about dental topics",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question to get information about"}
                },
                "required": ["question"]
            }
        }
    }
]