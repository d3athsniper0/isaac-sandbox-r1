#!/usr/bin/env python3
import os
import json
import time
import requests
import logging
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# Backend configuration
API_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_URL}/v1/chat/completions"
# Map function names to their respective endpoints
FUNCTION_ENDPOINTS = {
    "get_information": f"{API_URL}/get-information",
    "retrieve_record": f"{API_URL}/retrieve-record",
    #"generate_treatment_plan": f"{API_URL}/generate-treatment-plan",
    #"update_case": f"{API_URL}/update-case",
    "send_notification": f"{API_URL}/send-notification",
    "verify_recipient": f"{API_URL}/verify-recipient"
}

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ANSI color codes for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

class ComprehensiveDentalChatBot:
    def __init__(self, model="gpt-4o-mini", patient_id=None, verbose=False):
        self.model = model
        self.patient_id = patient_id
        # Disable verbose mode by default
        self.verbose = False # Set to True to enable verbose logging
        self.session_id = f"session_{int(time.time())}"
        # raw_history keeps all messages (user, assistant, function calls, etc.)
        self.raw_history = []
        #logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    def add_message(self, role, content, meta=None):
        # Ensure content is always a non-None string.
        if content is None:
            content = ""
        message = {"role": role, "content": str(content)}
        if meta:
            message.update(meta)
        self.raw_history.append(message)

    def _sanitize_history(self):
        """
        Build the sanitized history for the API call.
        Only include:
          - For user/assistant/system: {role, content}
          - For function messages: {role, name, content}
        Also, if content is None, replace it with an empty string.
        """
        sanitized = []
        for msg in self.raw_history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content") or ""
            if role in ["system", "user", "assistant"]:
                sanitized.append({
                    "role": role,
                    "content": content
                })
            elif role == "function":
                name = msg.get("name", "unknown_function")
                sanitized.append({
                    "role": "function",
                    "name": name,
                    "content": content
                })
        return sanitized

    def _build_payload(self):
        """
        Construct the payload from the sanitized history.
        You may choose to limit or summarize older messages if needed.
        """
        payload = {
            "messages": self._sanitize_history(),
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 1500,
            "user_id": self.session_id
        }
        if self.patient_id:
            payload["patient_id"] = self.patient_id
            payload["retrieve_context"] = True
        return payload

    def _call_api(self, endpoint, payload):
        """
        Generic method to call an API endpoint with retry logic.
        """
        headers = {"Content-Type": "application/json"}
        for attempt in range(MAX_RETRIES):
            try:
                if self.verbose:
                    print(Colors.BOLD + "Payload:" + Colors.ENDC, json.dumps(payload, indent=2))
                response = requests.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                print(Colors.RED + f"API call error (attempt {attempt+1}): {e}" + Colors.ENDC)
                time.sleep(RETRY_DELAY)
        return None

    def _call_function(self, function_name, arguments):
        """
        Call a specific backend function API.
        """
        if function_name not in FUNCTION_ENDPOINTS:
            print(Colors.RED + f"Function '{function_name}' not configured." + Colors.ENDC)
            return {"error": f"Function '{function_name}' not configured."}
        endpoint = FUNCTION_ENDPOINTS[function_name]
        return self._call_api(endpoint, arguments) or {}

    
    def generate_email_payload(self, email_request, full_content):
        """
        Ask the LLM to extract from the full content the section that addresses the user's
        email request and generate an appropriate email subject.
        The LLM is instructed to return a JSON object with keys "subject" and "body".
        """
        prompt = (
            "You are an assistant that extracts relevant sections from a given text and generates "
            "an appropriate email subject based on a user's request. "
            "Given the user's email request and the full content, extract only the section that directly "
            "addresses the request and generate an appropriate subject line summarizing that information. "
            "Return your answer in valid JSON format with keys 'subject' and 'body'.\n\n"
            f"User email request: {email_request}\n\n"
            f"Full content: {full_content}\n\n"
            "Your JSON response:"
        )
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "gpt-4o-mini-mini",
            "temperature": 0.5,
            "max_tokens": 150,
            "user_id": self.session_id
        }
        result = self._call_api(CHAT_ENDPOINT, payload)
        if result and "choices" in result and result["choices"]:
            response_text = result["choices"][0]["message"].get("content", "").strip()
            try:
                email_payload = json.loads(response_text)
                # Validate that both subject and body are present
                if "subject" in email_payload and "body" in email_payload:
                    return email_payload
            except Exception as e:
                print(Colors.RED + f"Error parsing email payload: {e}" + Colors.ENDC)
        
        # Second attempt with a simplified prompt—no hardcoded subject.
        simple_prompt = (
            "Generate a JSON object with keys 'subject' and 'body' based solely on the following text. "
            "The subject should be a concise, descriptive summary, and the body should include the key details from the text.\n\n"
            f"{full_content}\n\n"
            "Your JSON response:"
        )
        payload["messages"] = [{"role": "user", "content": simple_prompt}]
        result = self._call_api(CHAT_ENDPOINT, payload)
        if result and "choices" in result and result["choices"]:
            response_text = result["choices"][0]["message"].get("content", "").strip()
            try:
                email_payload = json.loads(response_text)
                if "subject" in email_payload and "body" in email_payload:
                    return email_payload
            except Exception as e:
                print(Colors.RED + f"Error parsing email payload on second attempt: {e}" + Colors.ENDC)
        # As a last resort, return an empty subject and the full content as the body.
        return {"subject": "", "body": full_content}

    
    def process_function_calls(self, tool_calls):
        """
        Process all tool calls from the assistant's response.
        For each tool call, parse the function name and arguments,
        call the backend function, and append the response.
        """
        for call in tool_calls:
            try:
                function_details = call.get("function", {})
                function_name = function_details.get("name", "unknown_function")
                arguments_str = function_details.get("arguments", "{}")
                arguments = json.loads(arguments_str)
            except Exception as e:
                print(Colors.RED + f"Error processing tool call: {e}" + Colors.ENDC)
                continue
            print(Colors.BLUE + f"Calling function: {function_name}" + Colors.ENDC)
            result = self._call_function(function_name, arguments)

            # Append the function response as a message
            self.add_message("function", json.dumps(result), {"name": function_name})

            # Update last_email_content based on the function type
            if function_name in ["get_information", "retrieve_record", "generate_treatment_plan", "update_case"]:
                # For these functions, use the returned content as the email content
                self.last_email_content = result.get("content", "")
                print(Colors.GREEN + f"Last email content: {self.last_email_content}" + Colors.ENDC)

            # Force email sending after verify_recipient
            if function_name == "verify_recipient" and result.get("success"):
                recipient = result.get("recipient", {})
                if recipient.get("email"):
                    # Use the stored literature search result
                    email_body = getattr(self, "last_email_content", "")
                    if not email_body or len(email_body) < 50:
                        print(Colors.YELLOW + "Warning: Email content is missing or too short." + Colors.ENDC)
                        email_body = "Detailed information not found. Please check the conversation for details."
                    # Dynamically generate subject line from the literature content
                    
                    # Use the LLM to generate the final email payload (subject and filtered body)
                    # Pass the user's email request (if available) and the full content.
                    email_request = getattr(self, "last_email_request", "Please send all the information.")
                    email_payload = self.generate_email_payload(email_request, email_body)

                    notification_args = {
                        "notification_type": "email",
                        "to": recipient["email"],
                        "content": {
                            "subject": email_payload["subject"],
                            "body": email_payload["body"]
                        }
                    }
                    notif_result = self._call_function("send_notification", notification_args)
                    self.add_message("function", json.dumps(notif_result), {"name": "send_notification"})
                    confirmation = f"I've sent an email to {recipient.get('name', '')} at {recipient['email']} with the requested information."
                    self.add_message("assistant", confirmation)
                    # Optionally, clear the last_email_content after sending
                    self.last_email_content = ""
                    return

    def get_response(self, user_input):
        """
        Main method to get a response:
          1. Add the user's message.
          2. Build and send the sanitized payload.
          3. If tool calls are present in the assistant's reply, process them
             and then re-call the completions endpoint.
          4. Return the final assistant message.
        """
        self.add_message("user", user_input)
        payload = self._build_payload()
        result = self._call_api(CHAT_ENDPOINT, payload)
        if not result:
            return {"role": "assistant", "content": "Error: No response from API."}
        if "choices" in result and result["choices"]:
            message = result["choices"][0]["message"]
            # Process tool calls if they exist
            if "tool_calls" in message and message["tool_calls"]:
                self.add_message("assistant", message.get("content") or "", {"tool_calls": message["tool_calls"]})
                self.process_function_calls(message["tool_calls"])
                # Rebuild payload after processing function calls
                payload = self._build_payload()
                result = self._call_api(CHAT_ENDPOINT, payload)
                if result and "choices" in result and result["choices"]:
                    message = result["choices"][0]["message"]
                    self.add_message("assistant", message.get("content") or "")
                    return message
                else:
                    return {"role": "assistant", "content": "Error: No valid final response."}
            else:
                self.add_message("assistant", message.get("content") or "")
                return message
        return {"role": "assistant", "content": "Error: Unexpected response format."}

    def run_terminal(self):
        """
        Terminal-based interaction loop.
        This method lets you run the chatbot in the terminal.
        """
        print(Colors.BOLD + "Trust AI Dental Assistant" + Colors.ENDC)
        while True:
            try:
                user_input = input(Colors.BOLD + "You: " + Colors.ENDC).strip()
                if user_input.lower() in ["quit", "exit"]:
                    print(Colors.BLUE + "Exiting..." + Colors.ENDC)
                    break
                response = self.get_response(user_input)
                # Ensure we output a non-null string
                print(Colors.GREEN + "Assistant: " + (response.get("content") or "") + Colors.ENDC)
            except KeyboardInterrupt:
                print(Colors.BLUE + "\nExiting chat session..." + Colors.ENDC)
                break
            except Exception as e:
                print(Colors.RED + f"Error: {e}" + Colors.ENDC)

if __name__ == "__main__":
    # Instantiate the bot with your preferred settings.
    bot = ComprehensiveDentalChatBot(model="gpt-4o-mini", verbose=True)
    bot.run_terminal()
