import os
import json
import logging
from typing import Dict, Optional, Tuple, List, Any
import httpx
import asyncio
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

class FastPPLXManager:
    """Optimized version of PPLXManager with async capabilities"""
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.api_key = os.getenv("PPLX_API_KEY")
        if not self.api_key:
            error_msg = "PPLX_API_KEY not found in environment variables"
            logger.error(error_msg)
            print(error_msg)  # Also print for direct visibility
            raise ValueError(error_msg)
        
        self.url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use provided client or create a new one
        self.http_client = http_client or httpx.AsyncClient(timeout=30.0)
        
        # Set model from environment variable
        if os.getenv("KNOWLEDGE_MODEL") == "sonar-pro":
            self.model = "sonar-pro"
        else:
            self.model = "sonar"
        
        # Debug log initialization 
        logger.info(f"FastPPLXManager initialized with model: {self.model}")
        print(f"FastPPLXManager initialized with model: {self.model}")  # Direct visibility
    
    async def get_information_async(self, question: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Asynchronous version of get_information that uses httpx for better performance.
        
        Args:
            question (str): The question to ask
            
        Returns:
            Tuple[Optional[str], Optional[List[str]]]: A tuple containing (answer, citations)
        """
        try:
            # Log request
            logger.info(f"Sending question to Perplexity: {question}")
            print(f"Sending question to Perplexity: {question}")  # Direct visibility
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Be precise and concise."
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.2,
                "top_p": 0.9,
                "search_domain_filter": None,
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "year",
                "top_k": 0,
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 1,
                "response_format": None
            }

            # Use AsyncClient for non-blocking HTTP request
            response = await self.http_client.post(
                self.url, 
                json=payload, 
                headers=self.headers
            )
            
            # Log response status
            logger.info(f"Perplexity API response status: {response.status_code}")
            print(f"Perplexity API response status: {response.status_code}")
            
            response.raise_for_status()
            
            # Get JSON data
            data = response.json()
            
            # Debug log raw response
            logger.debug(f"Perplexity API raw response: {data}")
            
            # Extract content and citations from response
            content = None
            citations = []
            
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0].get('message', {}).get('content')
                citations = data.get('citations', [])
                
                # Log extracted content and citations
                logger.info(f"Extracted content: {content[:50]}...")
                logger.info(f"Extracted citations: {citations}")
                print(f"Extracted content: {content[:50] if content else 'None'}...")
                print(f"Extracted citations: {citations}")
            else:
                logger.warning("No choices found in Perplexity response")
                print("No choices found in Perplexity response")
            
            return content, citations

        except httpx.RequestError as e:
            error_msg = f"Error making request to Perplexity AI: {e}"
            logger.error(error_msg)
            print(error_msg)  # Direct visibility
            return None, None
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON response: {e}"
            logger.error(error_msg)
            print(error_msg)  # Direct visibility
            return None, None
        except Exception as e:
            error_msg = f"Unexpected error in get_information_async: {e}"
            logger.error(error_msg)
            print(error_msg)  # Direct visibility
            return None, None
    
    # Maintain compatibility with original synchronous method
    def get_information(self, question: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Synchronous wrapper for backward compatibility.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.get_information_async(question))