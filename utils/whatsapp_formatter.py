# utils/whatsapp_formatter.py
import re
from typing import Dict, Any, List

def strip_html(text: str) -> str:
    """Remove HTML tags and convert to WhatsApp-friendly format"""
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Fix escaped characters
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    
    # Convert some simple markdown-like formatting that WhatsApp supports
    # Bold
    text = re.sub(r'<(b|strong)>(.*?)</\1>', '*\\2*', text)
    # Italic
    text = re.sub(r'<(i|em)>(.*?)</\1>', '_\\2_', text)
    # Code/monospace
    text = re.sub(r'<(code|pre)>(.*?)</\1>', '```\\2```', text)
    
    return text

def format_function_result_for_whatsapp(result: Dict[str, Any]) -> str:
    """Format function results for WhatsApp display"""
    if not result or not isinstance(result, dict):
        return "Sorry, I couldn't process that request."
    
    # For get_information results
    if "content" in result and "success" in result:
        # This is from get_information function
        content = result.get("content", "")
        citations = result.get("citations", [])
        
        # Clean the content text
        content = strip_html(content)
        
        # Format content with proper line breaks
        formatted_text = content
        
        # Add references at the bottom
        if citations and len(citations) > 0:
            formatted_text += "\n\nReferences:\n"
            for i, citation in enumerate(citations, 1):
                if isinstance(citation, str):
                    formatted_text += f"{i}. {citation}\n"
                elif isinstance(citation, dict) and "url" in citation:
                    formatted_text += f"{i}. {citation.get('title', 'Source')} - {citation['url']}\n"
        
        return formatted_text
    
    # For other function results, just return the content if available
    elif "content" in result:
        return strip_html(result["content"])
    
    # If it's a raw JSON object, try to extract key info
    try:
        if "success" in result and result["success"] == True:
            if isinstance(result.get("content"), str):
                return strip_html(result["content"])
            # If content is a dictionary or other structure, convert to string
            return "Information found. " + strip_html(str(result.get("content", "")))
    except:
        pass
    
    # Generic fallback formatting - convert to string but hide implementation details
    return "I found some information for you:\n\n" + strip_html(str(result))

def split_long_message(text: str, max_length: int = 1000) -> List[str]:
    """Split a long message into chunks for WhatsApp"""
    # First try to split at paragraph breaks
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max_length
        if len(current_chunk) + len(paragraph) + 2 > max_length:
            # If current_chunk is not empty, add it to chunks
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # The paragraph itself is too long, split it
                words = paragraph.split(' ')
                for word in words:
                    if len(current_chunk) + len(word) + 1 > max_length:
                        chunks.append(current_chunk)
                        current_chunk = word
                    else:
                        if current_chunk:
                            current_chunk += ' ' + word
                        else:
                            current_chunk = word
        else:
            # Add paragraph with a paragraph break
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks