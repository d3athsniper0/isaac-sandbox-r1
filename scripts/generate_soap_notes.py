# Transcribe audio file and generate dental SOAP notes
import os
import json
from dotenv import load_dotenv
from io import BytesIO
import requests
import argparse
from openai import OpenAI
import re
from datetime import datetime, timezone
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import time

load_dotenv()

class EmailSender:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('SENDGRID_API_KEY')
        if not self.api_key:
            raise ValueError("SendGrid API key not found in environment variables")
        
         # Main sender identity (Isaac)
        self.sender_email = "ai@trustdentistry.ai"
        self.sender_name = "Isaac from Trust AI"
        
        # Secondary sender identity (Zoe)
        self.sender_email_2 = "support@trustdentistry.ai"
        self.sender_name_2 = "Zoe from Trust AI"

        self.client = SendGridAPIClient(self.api_key)

    def send_email(self, to_email: str, subject: str, content: str, recipient_name: str = None, 
           use_sender_2: bool = False, template_type: str = "literature") -> dict:
        """
        Send a simple email.
        """
        
        try:
            # Format content as HTML is removed since we're already formatting before calling this method
            
            # Select the appropriate sender based on use_sender_2 flag
            from_email = (self.sender_email_2, self.sender_name_2) if use_sender_2 else (self.sender_email, self.sender_name)
        
            message = Mail(
                from_email=from_email,
                to_emails=to_email,
                subject=subject or "Information from Trust AI Dental Assistant",  # Add fallback
                html_content=content
            )
            
            response = self.client.send(message)
            
            return {
                "success": True,
                "status_code": response.status_code,
                "details": {
                    "recipient": to_email,
                    "sender": from_email[0]
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

def format_transcript_with_speakers(response_json):
    """Format the transcript to clearly show different speakers"""
    
    # Check if we have the expected response structure
    if not response_json.get("text") or not response_json.get("words"):
        return response_json.get("text", "No transcript available")
    
    # Extract words with speaker information
    words = response_json.get("words", [])
    
    if not words:
        return response_json.get("text", "No transcript available")
    
    # If no speaker information is available in any words, use a simple analysis
    if not any("speaker" in word for word in words):
        # Fall back to a more basic approach to identify speakers
        return analyze_and_format_speakers(response_json.get("text", ""))
    
    # Get unique speakers
    speakers = set()
    for word in words:
        if "speaker" in word:
            speakers.add(word["speaker"])
    
    # Assign speaker labels
    speaker_labels = {}
    for i, speaker in enumerate(sorted(speakers)):
        speaker_labels[speaker] = f"Speaker {i+1}"
    
    # Group words by speaker and create paragraphs
    current_speaker = None
    formatted_lines = []
    current_line = ""
    
    for word in words:
        word_speaker = word.get("speaker")
        
        # Start a new paragraph for a new speaker
        if word_speaker != current_speaker and word_speaker is not None:
            if current_line:  # Add the previous line if it exists
                formatted_lines.append(current_line)
            
            # Start a new line with speaker label
            current_speaker = word_speaker
            current_line = f"{speaker_labels[current_speaker]}: {word.get('text', '')}"
        else:
            # Continue the current line
            if current_line:
                current_line += f" {word.get('text', '')}"
            else:
                current_line = word.get('text', '')
    
    # Add the last line
    if current_line:
        formatted_lines.append(current_line)
    
    # Join all lines
    formatted_transcript = "\n\n".join(formatted_lines)
    
    # Add a header explaining the speaker identification
    header = "--- Transcript with Identified Speakers ---\n\n"
    
    return header + formatted_transcript

def analyze_and_format_speakers(raw_text):
    """Use heuristics to identify speakers when API doesn't provide speaker labels"""
    import re
    
    # Clean up excessive whitespace
    clean_text = re.sub(r'\s+', ' ', raw_text).strip()
    
    # Try to identify question marks as potential patient questions
    sentences = re.split(r'([.!?])\s+', clean_text)
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
    
    # Identify potential patient questions (sentences ending with question marks)
    formatted_lines = []
    current_speaker = "Dentist"
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Switch speaker for questions
        if sentence.endswith('?'):
            if current_speaker != "Patient":
                current_speaker = "Patient"
                formatted_lines.append(f"\nPatient: {sentence}")
            else:
                formatted_lines.append(sentence)
        else:
            # After a patient question, the next sentence is likely from the dentist
            if current_speaker == "Patient":
                current_speaker = "Dentist"
                formatted_lines.append(f"\nDentist: {sentence}")
            else:
                formatted_lines.append(sentence)
    
    # Better format the final text
    formatted_text = ' '.join(formatted_lines)
    formatted_text = re.sub(r'\n\s+', '\n', formatted_text)
    
    # Now manually analyze the transcript based on common dialogue patterns in dental settings
    # Look for specific phrases that typically indicate patient vs dentist
    
    # Post-process the text to combine adjacent sentences from the same speaker
    formatted_text = re.sub(r'(Dentist: .*?[.!])\s+(?!Patient:)', r'\1 ', formatted_text)
    formatted_text = re.sub(r'(Patient: .*?[.!])\s+(?!Dentist:)', r'\1 ', formatted_text)
    
    header = "--- Transcript with Identified Speakers (Pattern-Based Analysis) ---\n\n"
    
    return header + formatted_text

def manual_speaker_identification(transcription):
    """Alternative method using OpenAI to identify speakers in a transcript"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""
    This is a transcript of a conversation between a dentist and a patient. Please identify who is speaking each part 
    and format the transcript with clear speaker labels (Dentist: and Patient:).
    
    The dentist typically:
    - Explains procedures and recommendations
    - Answers questions
    - Discusses treatment options
    
    The patient typically:
    - Asks questions
    - Expresses concerns
    - Makes decisions about treatment
    
    Original transcript:
    {transcription}
    
    Please format this as a clean dialogue with proper spacing (no extra spaces) and clear speaker labels.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an expert at identifying speakers in medical transcripts and formatting them clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in manual speaker identification: {e}")
        return transcription

def transcribe_audio(audio_file):
    """Transcribe audio file using ElevenLabs API with speaker diarization"""
    import requests
    import json
    
    # Get API key from environment variables
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    # If audio_file is a URL, download it
    if audio_file.startswith(("http://", "https://")):
        response = requests.get(audio_file)
        audio_data = response.content
    else:
        # If it's a local file path
        with open(audio_file, "rb") as f:
            audio_data = f.read()
    
    # Prepare the API request
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key
    }
    
    # Create form data with the audio file
    files = {"file": ("audio_file", audio_data)}
    data = {
        "model_id": "scribe_v1",
        "tag_audio_events": "true",
        "language_code": "eng",
        "diarize": "true"  # Enable speaker identification
    }
    
    # Make the API request
    response = requests.post(url, headers=headers, files=files, data=data)
    
    # Check for errors
    if response.status_code != 200:
        error_message = f"ElevenLabs API Error: {response.status_code} - {response.text}"
        print(error_message)
        raise Exception(error_message)
    
    # Parse the response
    result = response.json()
    
    # Save the raw API response for debugging
    #with open("api_response.json", "w") as f:
    #    json.dump(result, f, indent=2)
    
    # Get the basic transcript
    basic_transcript = result.get("text", "")
    
    # Clean up extra spaces
    basic_transcript = ' '.join(basic_transcript.split())
    
    # Try to format with speaker identification from API
    api_formatted = format_transcript_with_speakers(result)
    
    # If the API formatting doesn't distinguish speakers well, use our backup method
    if "Speaker 1" in api_formatted and "Speaker 2" not in api_formatted:
        print("API speaker diarization insufficient, using backup speaker identification...")
        return manual_speaker_identification(basic_transcript)
    
    return api_formatted

def clean_soap_note(soap_note):
    """Remove lines with 'Not mentioned in transcript' or 'Information not provided'"""
    # Split note into lines
    lines = soap_note.split('\n')
    
    # Keep track of lines to remove
    lines_to_keep = []
    skip_next_empty = False
    
    for i, line in enumerate(lines):
        # Check if line contains phrases to remove
        if re.search(r'(not mentioned in transcript|information not provided|no information provided|None mentioned)', line.lower()):
            skip_next_empty = True
            continue
        
        # Skip empty line following a removed line
        if skip_next_empty and line.strip() == '':
            skip_next_empty = False
            continue
            
        lines_to_keep.append(line)
    
    # Join the remaining lines
    cleaned_note = '\n'.join(lines_to_keep)

    # Ensure proper spacing between sections by adding a blank line after certain headers
    section_headers = [
        r'Provider Name:.*?\n',
        r'S \(Subjective\):.*?\n',
        r'O \(Objective\):.*?\n',
        r'A \(Assessment\):.*?\n',
        r'P \(Plan\):.*?\n',
        r'Additional Notes:.*?\n'
    ]

    for header_pattern in section_headers:
        # Replace header followed by content without blank line
        cleaned_note = re.sub(f"({header_pattern})([^\n])", r"\1\n\2", cleaned_note)
    
    # Remove any bullet points that are now empty
    cleaned_note = re.sub(r'- \s*\n', '', cleaned_note)
    
    # Remove any double blank lines
    cleaned_note = re.sub(r'\n\n\n+', '\n\n', cleaned_note)
    
    # Clean up empty sections (heading followed immediately by another heading)
    section_patterns = [
        r'(S \(Subjective\):)\s*\n\n(O \(Objective\):)',
        r'(O \(Objective\):)\s*\n\n(A \(Assessment\):)',
        r'(A \(Assessment\):)\s*\n\n(P \(Plan\):)',
    ]
    
    for pattern in section_patterns:
        cleaned_note = re.sub(pattern, r'\1\n\n\2', cleaned_note)
    
    # Ensure "Additional Notes:" is preceded by a blank line
    cleaned_note = re.sub(r'([^\n])\n(Additional Notes:)', r'\1\n\n\2', cleaned_note)
    
    return cleaned_note

def generate_dental_soap_note(transcription, provider_name, current_date_time, patient_name=None):
    """Generate dental SOAP note from transcription using OpenAI API"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    patient_name_line = f"Patient Name: {patient_name}" if patient_name and patient_name != "[To be filled by the provider]" else "Patient Name: [To be filled by the provider]"

    # Dental-specific prompt with anti-hallucination guardrails
    prompt = f"""
    You are a dental professional assistant. Convert the following dental appointment transcription into a formal, detailed SOAP note.
    
    CRITICAL INSTRUCTION: ONLY include information that is EXPLICITLY stated in the transcription. DO NOT add any details, findings, or medical information that is not directly mentioned in the transcript. Healthcare documentation must be factual and accurate.
    
    If certain sections cannot be completed because the information is not present in the transcript, indicate this with "Not mentioned in transcript" or "Information not provided".
    
    Use proper dental terminology and follow this structure:

    Patient Name: {patient_name_line}
    Date and Time of Visit (UTC): {current_date_time}
    Provider Name: {provider_name}
    Appointment Type: ONLY if mentioned
    
    S (Subjective): 
    - Chief complaint in patient's own words - ONLY if explicitly stated
    - History of present illness - ONLY mention what was discussed
    - Dental history - ONLY if mentioned
    - Medical history - ONLY if mentioned
    - Pain assessment - ONLY if mentioned
    
    O (Objective):
    - Vital signs - ONLY if recorded and mentioned
    - Extra-oral examination findings - ONLY if mentioned
    - Intra-oral examination - ONLY include teeth that were specifically discussed
    - Periodontal status - ONLY if mentioned
    - Description of affected teeth - ONLY those mentioned in transcript
    - Radiographic findings - ONLY if mentioned
    
    A (Assessment):
    - Diagnosis - ONLY what was explicitly stated or clearly implied
    - Caries risk assessment - ONLY if mentioned
    - Periodontal classification - ONLY if mentioned
    
    P (Plan):
    - Treatment plan with codes - ONLY what was discussed
    - Restorative procedures - ONLY if mentioned
    - Periodontal therapy - ONLY if mentioned
    - Prescription medications - ONLY if mentioned
    - Oral hygiene instructions - ONLY if mentioned
    - Recall interval - ONLY if mentioned

    Additional Notes:
    - Include important contextual information (if any) from the conversation that doesn't fit neatly into the SOAP format
    - Do NOT repeat information that is already included in the SOAP note

    **Signature:**  
    [Insert Provider Signature]  
    [Insert Provider Credentials]  
    
    EXAMPLES OF HALLUCINATIONS TO AVOID:
    1. Do not invent specific teeth numbers unless mentioned
    2. Do not invent periodontal measurements unless specifically stated
    3. Do not assume diagnoses that weren't mentioned
    4. Do not include examinations that weren't performed
    5. Do not add medications or treatment codes not mentioned
    
    For any information not included in the transcript, note "Not mentioned in transcript" rather than inventing details.
    
    Format the SOAP note professionally as it would appear in a dental patient record.
    Do not include any dashed lines in the SOAP note and maintain the same spacing between sections. "Additional Notes" should be a separate section.
    
    Transcription:
    {transcription}
    """
    
    try:
        response = client.chat.completions.create(
            model="o3-mini",  # Use appropriate model
            messages=[
                {"role": "system", "content": "You are a dental professional assistant converting patient transcriptions into formal dental SOAP notes. You must ONLY include information that is explicitly stated in the transcript. NEVER add details that aren't mentioned. This is critical for patient safety."},
                {"role": "user", "content": prompt}
            ],
            #temperature=0.1,  # Very low temperature for consistent, factual output
            #max_tokens=2500
        )
        
        # Process the SOAP note to remove "Not mentioned" lines
        raw_soap_note = response.choices[0].message.content
        cleaned_soap_note = clean_soap_note(raw_soap_note)
        return cleaned_soap_note
    except Exception as e:
        print(f"Error generating SOAP note: {e}")
        return f"Error generating SOAP note: {e}"

def format_soap_note_as_html(soap_note):
    """Format a SOAP note as HTML using OpenAI's GPT-4.1 model"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""
    Format the following dental SOAP note into clean, professional HTML for email viewing.
    Use appropriate HTML elements, styles, and structure.
    Preserve all sections, content, and formatting from the original SOAP note.
    CRITICAL:Do NOT change or add any medical information.
    
    Original SOAP note:
    {soap_note}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an HTML formatter for medical documents. Convert plain text SOAP notes into well-structured, professional HTML while preserving all medical content exactly as provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=3000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error formatting SOAP note as HTML: {e}")
        # Fallback to basic HTML formatting
        return f"<pre>{soap_note}</pre>"

def send_soap_note_email(email_address, soap_note, provider_name, patient_name, current_date_time):
    """Send the SOAP note as an HTML-formatted email"""
    try:
        # Format SOAP note as HTML
        html_soap_note = format_soap_note_as_html(soap_note)
        
        # Create email sender
        email_sender = EmailSender()
        
        # Extract patient information for subject line if not provided
        if not patient_name or patient_name == "[To be filled by the provider]":
            extracted_name = "Patient"
            patient_name_match = re.search(r"Patient Name:\s*(.+?)\s*\n", soap_note)
            if patient_name_match:
                extracted_name = patient_name_match.group(1).strip()
                if extracted_name == "[To be filled by the provider]":
                    extracted_name = "Patient"
            patient_name = extracted_name
        
        # Create subject line
        subject = f"Dental SOAP Note - {patient_name} - {current_date_time}"
        
        # Send email
        result = email_sender.send_email(
            to_email=email_address,
            subject=subject,
            content=html_soap_note,
            template_type="default"
        )
        
        if result["success"]:
            print(f"SOAP note successfully emailed to {email_address}")
        else:
            print(f"Failed to email SOAP note: {result.get('error', 'Unknown error')}")
            
        return result["success"]
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def main(audio_file, output_file, provider_name, current_date_time, email_address=None, patient_name=None):
    try:
        # Step 1: Transcribe the audio
        # Measure the time taken to transcribe the audio
        start_time = time.time()
        print("Transcribing audio...")
        transcription = transcribe_audio(audio_file)
        end_time = time.time()
        print(f"Time taken to transcribe audio: {end_time - start_time} seconds")
        
        if not transcription or transcription.strip() == "":
            print("Error: Received empty transcription from ElevenLabs API")
            return
        
        # Save raw transcription 
        transcription_file = output_file.replace(".txt", "_transcription.txt")
        with open(transcription_file, "w", encoding="utf-8") as f:
            f.write(transcription)
        print(f"Raw transcription saved to {transcription_file}")
        
        # Step 2: Generate dental SOAP note
        print("Generating dental SOAP note using Isaac-Lite-R1...")
        soap_note = generate_dental_soap_note(transcription, provider_name, current_date_time, patient_name)
        
        # Step 3: Save SOAP note
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(soap_note)
        print(f"Dental SOAP note saved to {output_file}")
        
        # Step 4: Perform verification check
        print("Performing hallucination verification check...")
        verification_result = verify_no_hallucinations(transcription, soap_note)
        verification_file = output_file.replace(".txt", "_verification.txt")
        with open(verification_file, "w", encoding="utf-8") as f:
            f.write(verification_result)
        print(f"Verification report saved to {verification_file}")

        # Send email if address provided
        if email_address:
            if not os.getenv('SENDGRID_API_KEY'):
                print("Warning: SENDGRID_API_KEY environment variable is not set. Email will not be sent.")
            else:
                print(f"Sending SOAP note to {email_address}...")
                send_soap_note_email(email_address, soap_note, provider_name, patient_name, current_date_time)
        
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

def verify_no_hallucinations(transcription, soap_note):
    """Verify the SOAP note doesn't contain hallucinations by comparing with transcript"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""
    You are a dental verification expert. Your task is to analyze a dental transcript and the resulting SOAP note to identify any potential hallucinations or fabricated information.
    
    CRITICAL: Healthcare documentation must ONLY contain information that is explicitly stated or very clearly implied in the transcript.
    
    Please review both texts and identify ANY instances where the SOAP note contains information that is not present in the transcript.
    
    Focus on these critical areas:
    1. Patient identifiers or demographics not mentioned in transcript
    2. Medical history elements not discussed
    3. Diagnostic findings not mentioned
    4. Treatment details or recommendations not discussed
    5. Specific teeth numbers or locations not specified in transcript
    6. Measurements or clinical parameters not mentioned
    7. Medications or prescriptions not discussed
    
    For every potential hallucination, quote the suspicious text from the SOAP note and explain why it appears to be fabricated.
    
    If no hallucinations are found, state this clearly.
    
    Original Transcript:
    ---
    {transcription}
    ---
    
    SOAP Note:
    ---
    {soap_note}
    ---
    
    Provide your analysis in a clear, structured format with a summary of findings at the end.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a dental verification expert whose sole purpose is to identify potential hallucinations in healthcare documentation by comparing a transcript to a SOAP note. Be extremely thorough and critical."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error performing verification: {str(e)}"

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Transcribe audio and generate dental SOAP notes")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file or URL")
    parser.add_argument("--o", type=str, required=True, help="Output file for dental SOAP note")
    # Add this line after the existing parser.add_argument statements
    parser.add_argument("--email", type=str, help="Email address to send the SOAP note to (optional)")

    # Insert provider name
    parser.add_argument("--provider", type=str, required=True, help="Provider name")

    # Provide patient name, default to "[To be filled by the provider]" if not provided
    parser.add_argument("--patient", type=str, help="Patient name (optional)")

    args = parser.parse_args()
    
    if not args.patient:
        args.patient = "[To be filled by the provider]"

    
    # Current date and time
    # Universal clock time
    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Check if API keys are set
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("Error: ELEVENLABS_API_KEY environment variable is not set.")
        print("Please set it in your .env file or export it in your environment.")
        exit(1)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in your .env file or export it in your environment.")
        exit(1)
    
    print("Starting process...")
    print(f"Audio: {args.audio}")
    print(f"Output: {args.o}")
    
    # Display usage instructions
    #print("Usage: python transcribe_audio.py --audio <audio_file> --o <output_file>")
    
    # If provider name does not contain "Dr." or "Dr", add it
    if not re.search(r"Dr\.", args.provider, re.IGNORECASE):
        args.provider = "Dr. " + args.provider
    
    main(args.audio, args.o, args.provider, current_date_time, args.email, args.patient)