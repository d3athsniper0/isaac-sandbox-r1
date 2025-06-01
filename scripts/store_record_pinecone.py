# Usage: python store_record_pinecone.py record <file_path>
# Usage: python store_record_pinecone.py protocol --directory <directory_path>
# Usage: python store_record_pinecone.py protocol --file <file_path>

import sys
import json
import os
import datetime
from datetime import datetime
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

def create_embeddings(text_content):
    """
    Generate embeddings for the provided text content.
    
    Parameters:
    - text_content (str): Text to generate embeddings for
    
    Returns:
    - list: The embedding vector
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.embeddings.create(
        input=text_content,
        model="text-embedding-3-small"
    )
    
    return response.data[0].embedding

def extract_key_metadata(metadata, patient_record):
    """
    Extract key fields from metadata and patient record for efficient filtering,
    optimized for dental records and queries.
    
    Parameters:
    - metadata (dict): Original metadata dictionary
    - patient_record (dict): Patient record dictionary
    
    Returns:
    - dict: Dictionary with key metadata fields
    """
    # Core metadata fields
    key_metadata = {
        "Practice_ID": metadata.get("Practice_ID", ""),
        "File_ID": metadata.get("File_ID", ""),
        "Patient_Case_ID": metadata.get("Patient_Case_ID", ""),
        "Patient_Name": metadata.get("Patient_Name", ""),
        "Content_Type": metadata.get("Content_Type", ""),
        "Update_Number": metadata.get("Update_Number", ""),
        "Timestamp": metadata.get("Timestamp", ""),
        "Sequence_Number": metadata.get("Sequence_Number", "")
    }
    
    content_type = metadata.get("Content_Type", "")
    
    # For Intraoral_Photograph
    if content_type == "Intraoral_Photograph" and "synthesis" in patient_record:
        synthesis = patient_record["synthesis"]
        
        # Extract severity level
        if "case_summary" in synthesis and "severity_level" in synthesis["case_summary"]:
            key_metadata["severity_level"] = synthesis["case_summary"]["severity_level"]
        
        # Extract clinical findings - soft tissue
        if "clinical_findings" in synthesis and "soft_tissue_assessment" in synthesis["clinical_findings"]:
            key_metadata["soft_tissue_findings"] = synthesis["clinical_findings"]["soft_tissue_assessment"]
        
        # Extract clinical findings - hard tissue
        if "clinical_findings" in synthesis and "hard_tissue_assessment" in synthesis["clinical_findings"]:
            key_metadata["hard_tissue_findings"] = synthesis["clinical_findings"]["hard_tissue_assessment"]
        
        # Extract primary concerns
        if "diagnostic_assessment" in synthesis and "primary_concerns" in synthesis["diagnostic_assessment"]:
            key_metadata["primary_concerns"] = synthesis["diagnostic_assessment"]["primary_concerns"]
        
        # Extract differential diagnosis
        if "diagnostic_assessment" in synthesis and "differential_diagnosis" in synthesis["diagnostic_assessment"]:
            key_metadata["differential_diagnosis"] = synthesis["diagnostic_assessment"]["differential_diagnosis"]
    
    # For X_ray, Pano, CBCT
    elif content_type in ["X_ray", "Pano", "CBCT"] and "synthesis" in patient_record:
        synthesis = patient_record["synthesis"]
        
        # Extract clinical narrative
        if "clinical_narrative" in synthesis:
            if "summary" in synthesis["clinical_narrative"]:
                key_metadata["clinical_summary"] = synthesis["clinical_narrative"]["summary"]
            
            if "immediate_concerns" in synthesis["clinical_narrative"]:
                concerns = []
                for concern in synthesis["clinical_narrative"]["immediate_concerns"]:
                    if "finding" in concern:
                        concerns.append(concern["finding"])
                if concerns:
                    key_metadata["immediate_concerns"] = concerns
        
        # Extract teeth information
        if "consensus_findings" in synthesis:
            findings = synthesis["consensus_findings"]
            
            # Extract teeth affected
            if "teeth_affected" in findings:
                key_metadata["teeth_affected"] = findings["teeth_affected"]
            
            # Extract dental conditions
            if "confirmed_conditions" in findings:
                condition_types = []
                condition_locations = []
                for condition in findings["confirmed_conditions"]:
                    if "type" in condition:
                        condition_types.append(condition["type"])
                    if "location" in condition:
                        condition_locations.append(condition["location"])
                
                if condition_types:
                    key_metadata["condition_types"] = condition_types
                if condition_locations:
                    key_metadata["condition_locations"] = condition_locations
            
            # Extract risk factors
            if "risk_factors" in findings:
                risk_factors = []
                for category, risks in findings["risk_factors"].items():
                    if risks and risks[0] != "none identified":
                        risk_factors.extend(risks)
                if risk_factors:
                    key_metadata["risk_factors"] = risk_factors
    
    # For Treatment_Plan
    elif content_type == "Treatment_Plan" and "patient_record" in patient_record:
        record = patient_record["patient_record"]
        
        # Extract patient info (medical and dental)
        if "patient_info" in record:
            patient_info = record["patient_info"]
            
            # Extract chief complaint
            if "chief_complaint" in patient_info:
                key_metadata["chief_complaint"] = patient_info["chief_complaint"]
            
            # Extract dental history
            if "dental_history" in patient_info:
                key_metadata["dental_history"] = patient_info["dental_history"]
            
            # Extract medications (secondary but still useful)
            if "current_medications" in patient_info:
                med_text = patient_info["current_medications"]
                if isinstance(med_text, str):
                    medications = [m.strip() for m in med_text.replace(";", ",").split(",")]
                    key_metadata["medications"] = medications
                elif isinstance(med_text, list):
                    key_metadata["medications"] = med_text
            
            # Extract medical history (secondary but still useful)
            if "medical_history" in patient_info:
                med_history = patient_info["medical_history"]
                if isinstance(med_history, str):
                    conditions = [c.strip() for c in med_history.replace(";", ",").split(",")]
                    key_metadata["conditions"] = conditions
                elif isinstance(med_history, list):
                    key_metadata["conditions"] = med_history
            
            # Extract budget and time constraints
            if "budget_constraint" in patient_info:
                key_metadata["budget_constraint"] = patient_info["budget_constraint"]
            if "time_constraint" in patient_info:
                key_metadata["time_constraint"] = patient_info["time_constraint"]
        
        # Extract treatment information
        if "treatment_plan" in record and "synthesis" in record["treatment_plan"]:
            synthesis = record["treatment_plan"]["synthesis"]
            
            # Extract diagnosis
            if "diagnosis" in synthesis:
                diagnosis = synthesis["diagnosis"]
                if "condition" in diagnosis:
                    key_metadata["diagnosis"] = diagnosis["condition"]
                if "severity" in diagnosis:
                    key_metadata["diagnosis_severity"] = diagnosis["severity"]
                if "supporting_findings" in diagnosis:
                    key_metadata["supporting_findings"] = diagnosis["supporting_findings"]
            
            # Extract treatment approaches
            if "treatment_approaches" in synthesis:
                approaches = synthesis["treatment_approaches"]
                procedures = []
                philosophies = []
                for approach in approaches:
                    if "primary_procedure" in approach:
                        procedures.append(approach["primary_procedure"])
                    if "philosophy" in approach:
                        philosophies.append(approach["philosophy"])
                
                if procedures:
                    key_metadata["procedures"] = procedures
                if philosophies:
                    key_metadata["treatment_philosophies"] = philosophies
    
    # For Patient_Info (if we ever have this as a separate content type)
    elif content_type == "Patient_Info" and "patient_info" in patient_record:
        patient_info = patient_record["patient_info"]
        
        # Extract chief complaint
        if "chief_complaint" in patient_info:
            key_metadata["chief_complaint"] = patient_info["chief_complaint"]
        
        # Extract dental history
        if "dental_history" in patient_info:
            key_metadata["dental_history"] = patient_info["dental_history"]
        
        # Extract medications
        if "current_medications" in patient_info:
            med_text = patient_info["current_medications"]
            if isinstance(med_text, str):
                medications = [m.strip() for m in med_text.replace(";", ",").split(",")]
                key_metadata["medications"] = medications
            elif isinstance(med_text, list):
                key_metadata["medications"] = med_text
        
        # Extract medical history
        if "medical_history" in patient_info:
            med_history = patient_info["medical_history"]
            if isinstance(med_history, str):
                conditions = [c.strip() for c in med_history.replace(";", ",").split(",")]
                key_metadata["conditions"] = conditions
            elif isinstance(med_history, list):
                key_metadata["conditions"] = med_history
    
    return key_metadata

def generate_text_for_embedding(patient_record, content_type):
    """
    Generate a comprehensive text representation of the dental patient record for embedding,
    optimizing for dental terminology and searchability.
    
    Parameters:
    - patient_record (dict): Patient record dictionary
    - content_type (str): Type of content (X_ray, Pano, CBCT, Treatment_Plan, Intraoral_Photograph, etc.)
    
    Returns:
    - str: Text representation for embedding generation
    """
    text_parts = []
    
    # For Treatment_Plan records
    if content_type == "Treatment_Plan" and "patient_record" in patient_record:
        record = patient_record["patient_record"]
        
        # Add patient information
        if "patient_info" in record:
            info = record["patient_info"]
            text_parts.append(f"Patient Chief Complaint: {info.get('chief_complaint', '')}")
            text_parts.append(f"Medical History: {info.get('medical_history', '')}")
            text_parts.append(f"Dental History: {info.get('dental_history', '')}")
            text_parts.append(f"Current Medications: {info.get('current_medications', '')}")
            text_parts.append(f"Budget Constraint: {info.get('budget_constraint', '')}")
            text_parts.append(f"Time Constraint: {info.get('time_constraint', '')}")
            text_parts.append(f"X-ray Findings: {info.get('xray_findings', '')}")
            text_parts.append(f"Additional Info: {info.get('additional_info', '')}")
        
        # Add treatment plan synthesis
        if "treatment_plan" in record and "synthesis" in record["treatment_plan"]:
            synthesis = record["treatment_plan"]["synthesis"]
            
            # Add diagnosis
            if "diagnosis" in synthesis:
                diagnosis = synthesis["diagnosis"]
                text_parts.append(f"Diagnosis: {diagnosis.get('condition', '')}")
                text_parts.append(f"Severity: {diagnosis.get('severity', '')}")
                
                if "supporting_findings" in diagnosis:
                    text_parts.append("Supporting findings: " + ", ".join(diagnosis["supporting_findings"]))
                
                if "notes" in diagnosis:
                    text_parts.append(f"Diagnosis Notes: {diagnosis['notes']}")
            
            # Add treatment approaches
            if "treatment_approaches" in synthesis:
                for i, approach in enumerate(synthesis["treatment_approaches"]):
                    text_parts.append(f"Treatment Approach {i+1}: {approach.get('primary_procedure', '')}")
                    text_parts.append(f"Philosophy: {approach.get('philosophy', '')}")
                    text_parts.append(f"Priority Level: {approach.get('priority_level', '')}")
                    text_parts.append(f"Rationale: {approach.get('rationale', '')}")
                    text_parts.append(f"Timeline: {approach.get('timeline', '')}")
                    
                    # Add pros and cons
                    if "pros_cons" in approach:
                        pros_cons = approach["pros_cons"]
                        if "pros" in pros_cons:
                            text_parts.append("Pros: " + ", ".join(pros_cons["pros"]))
                        if "cons" in pros_cons:
                            text_parts.append("Cons: " + ", ".join(pros_cons["cons"]))
                    
                    # Add risks
                    if "risks" in approach:
                        text_parts.append("Risks: " + ", ".join(approach["risks"]))
            
            # Add clinical evidence
            if "clinical_evidence" in synthesis:
                evidence = synthesis["clinical_evidence"]
                
                if "success_rates" in evidence:
                    for procedure, rate in evidence["success_rates"].items():
                        text_parts.append(f"Success Rate for {procedure}: {rate}")
                
                if "key_studies" in evidence:
                    text_parts.append("Key Studies: " + ", ".join(evidence["key_studies"]))
                
                if "contraindications" in evidence:
                    text_parts.append("Contraindications: " + ", ".join(evidence["contraindications"]))
            
            # Add treatment phases
            if "treatment_phases" in synthesis:
                for phase in synthesis["treatment_phases"]:
                    phase_num = phase.get("phase_number", "")
                    phase_name = phase.get("phase", "")
                    text_parts.append(f"Treatment Phase {phase_num}: {phase_name}")
                    
                    if "procedures" in phase:
                        text_parts.append("Procedures: " + ", ".join(phase["procedures"]))
                    
                    if "clinical_markers" in phase:
                        text_parts.append("Clinical Markers: " + ", ".join(phase["clinical_markers"]))
                    
                    if "success_criteria" in phase:
                        text_parts.append("Success Criteria: " + ", ".join(phase["success_criteria"]))
            
            # Add follow up protocol
            if "follow_up_protocol" in synthesis:
                follow_up = synthesis["follow_up_protocol"]
                
                if "maintenance_schedule" in follow_up:
                    schedule = follow_up["maintenance_schedule"]
                    text_parts.append(f"Maintenance Frequency: {schedule.get('frequency', '')}")
                
                if "warning_signs" in follow_up:
                    text_parts.append(f"Warning Signs: {follow_up['warning_signs']}")
            
            # Add estimated cost
            if "estimated_cost" in synthesis:
                text_parts.append(f"Estimated Cost: {synthesis['estimated_cost']}")
    
    # For X_ray, Pano, and CBCT records
    elif content_type in ["X_ray", "Pano", "CBCT"] and "synthesis" in patient_record:
        synthesis = patient_record["synthesis"]
        
        # Add clinical narrative
        if "clinical_narrative" in synthesis:
            narrative = synthesis["clinical_narrative"]
            text_parts.append(f"Clinical Summary: {narrative.get('summary', '')}")
            
            # Add immediate concerns
            if "immediate_concerns" in narrative:
                for i, concern in enumerate(narrative["immediate_concerns"]):
                    text_parts.append(f"Finding {i+1}: {concern.get('finding', '')}")
                    text_parts.append(f"Clinical Significance: {concern.get('clinical_significance', '')}")
                    text_parts.append(f"Recommended Action: {concern.get('recommended_action', '')}")
                    text_parts.append(f"Urgency: {concern.get('urgency', '')}")
                    text_parts.append(f"Rationale: {concern.get('rationale', '')}")
        
        # Add notation conflicts (crucial for consistent tooth identification)
        if "notation_conflicts_resolved" in synthesis:
            conflicts = synthesis["notation_conflicts_resolved"]
            
            if "verified_quadrants" in conflicts:
                quadrants = conflicts["verified_quadrants"]
                for quadrant, teeth in quadrants.items():
                    if teeth:
                        text_parts.append(f"Teeth in {quadrant}: " + ", ".join(teeth))
        
        # Add consensus findings
        if "consensus_findings" in synthesis:
            findings = synthesis["consensus_findings"]
            
            if "teeth_affected" in findings:
                text_parts.append("Teeth affected: " + ", ".join(findings["teeth_affected"]))
            
            # Add confirmed conditions
            if "confirmed_conditions" in findings:
                for i, condition in enumerate(findings["confirmed_conditions"]):
                    text_parts.append(f"Condition {i+1}: {condition.get('type', '')}, Severity: {condition.get('severity', '')}")
                    text_parts.append(f"Location: {condition.get('location', '')}")
                    text_parts.append(f"Description: {condition.get('radiographic_description', '')}")
                    text_parts.append(f"Clinical Implications: {condition.get('clinical_implications', '')}")
            
            # Add risk factors
            if "risk_factors" in findings:
                for category, risks in findings["risk_factors"].items():
                    if risks and risks[0] != "none identified":
                        text_parts.append(f"Risk Factor - {category}: " + ", ".join(risks))
            
            # Add uncertain findings
            if "uncertain_findings" in findings:
                for i, finding in enumerate(findings["uncertain_findings"]):
                    text_parts.append(f"Uncertain Finding {i+1} Location: {finding.get('location', '')}")
                    
                    if "description" in finding:
                        desc = finding["description"]
                        text_parts.append(f"Radiographic Appearance: {desc.get('radiographic_appearance', '')}")
                        
                        if "possible_interpretations" in desc:
                            text_parts.append("Possible Interpretations: " + ", ".join(desc["possible_interpretations"]))
                        
                        if "clinical_significance" in desc:
                            text_parts.append(f"Clinical Significance: {desc['clinical_significance']}")
                    
                    if "differential_considerations" in finding:
                        text_parts.append("Differential Considerations: " + ", ".join(finding["differential_considerations"]))
        
        # Add structural assessment
        if "structural_assessment" in synthesis:
            assessment = synthesis["structural_assessment"]
            
            if "alignment" in assessment:
                alignment = assessment["alignment"]
                text_parts.append(f"Alignment: {alignment.get('consensus_view', '')}")
            
            if "bone_levels" in assessment:
                bone = assessment["bone_levels"]
                text_parts.append(f"Bone Levels: {bone.get('overall_status', '')}")
                
                if "critical_areas" in bone:
                    for area in bone["critical_areas"]:
                        text_parts.append(f"Critical Area Location: {area.get('location', '')}")
                        text_parts.append(f"Severity: {area.get('severity', '')}")
                        text_parts.append(f"Periodontal Implications: {area.get('periodontal_implications', '')}")
        
        # Add clinical recommendations
        if "clinical_recommendations" in synthesis:
            recommendations = synthesis["clinical_recommendations"]
            
            if "priorities" in recommendations:
                for priority in recommendations["priorities"]:
                    text_parts.append(f"Priority: {priority.get('priority', '')}")
                    text_parts.append(f"Procedure: {priority.get('procedure', '')}")
                    
                    if "rationale" in priority:
                        rationale = priority["rationale"]
                        text_parts.append(f"Urgency Reason: {rationale.get('urgency_reason', '')}")
                        text_parts.append(f"Delay Risks: {rationale.get('delay_risks', '')}")
                    
                    if "treatment_considerations" in priority:
                        text_parts.append("Treatment Considerations: " + ", ".join(priority["treatment_considerations"]))
        
        # Add diagnostic quality
        if "diagnostic_quality" in synthesis:
            quality = synthesis["diagnostic_quality"]
            
            if "technical_limitations" in quality:
                for limitation in quality["technical_limitations"]:
                    text_parts.append(f"Technical Limitation: {limitation.get('limitation', '')}")
                    text_parts.append(f"Impact: {limitation.get('impact', '')}")
            
            if "additional_imaging_recommendations" in quality:
                for rec in quality["additional_imaging_recommendations"]:
                    text_parts.append(f"Recommended Imaging: {rec.get('type', '')} for {rec.get('region', '')}")
                    text_parts.append(f"Rationale: {rec.get('rationale', '')}")
                    text_parts.append(f"Priority: {rec.get('priority', '')}")
    
    # For Intraoral_Photograph records
    elif content_type == "Intraoral_Photograph" and "synthesis" in patient_record:
        synthesis = patient_record["synthesis"]
        
        # Add case summary
        if "case_summary" in synthesis:
            summary = synthesis["case_summary"]
            text_parts.append(f"Severity Level: {summary.get('severity_level', '')}")
            text_parts.append(f"Primary Diagnosis: {summary.get('primary_diagnosis', '')}")
            
            if "key_verification_needs" in summary:
                text_parts.append("Key Verification Needs: " + ", ".join(summary["key_verification_needs"]))
        
        # Add clinical findings
        if "clinical_findings" in synthesis:
            findings = synthesis["clinical_findings"]
            
            if "hard_tissue_assessment" in findings:
                text_parts.append("Hard Tissue Assessment: " + ", ".join(findings["hard_tissue_assessment"]))
            
            if "soft_tissue_assessment" in findings:
                text_parts.append("Soft Tissue Assessment: " + ", ".join(findings["soft_tissue_assessment"]))
            
            if "existing_restorations" in findings:
                text_parts.append("Existing Restorations: " + ", ".join(findings["existing_restorations"]))
        
        # Add diagnostic assessment
        if "diagnostic_assessment" in synthesis:
            assessment = synthesis["diagnostic_assessment"]
            
            if "primary_concerns" in assessment:
                text_parts.append("Primary Concerns: " + ", ".join(assessment["primary_concerns"]))
            
            if "differential_diagnosis" in assessment:
                text_parts.append("Differential Diagnosis: " + ", ".join(assessment["differential_diagnosis"]))
        
        # Add risk stratification
        if "risk_stratification" in synthesis:
            risks = synthesis["risk_stratification"]
            
            if "immediate_risks" in risks:
                text_parts.append("Immediate Risks: " + ", ".join(risks["immediate_risks"]))
            
            if "long_term_risks" in risks:
                text_parts.append("Long Term Risks: " + ", ".join(risks["long_term_risks"]))
        
        # Add treatment considerations
        if "treatment_considerations" in synthesis:
            considerations = synthesis["treatment_considerations"]
            
            if "diagnostic_requirements" in considerations:
                text_parts.append("Diagnostic Requirements: " + ", ".join(considerations["diagnostic_requirements"]))
            
            if "intervention_options" in considerations:
                text_parts.append("Intervention Options: " + ", ".join(considerations["intervention_options"]))
    
    # For Patient_Info (if we ever have this as a separate content type)
    elif content_type == "Patient_Info" and "patient_info" in patient_record:
        patient_info = patient_record["patient_info"]
        text_parts.append(f"Chief Complaint: {patient_info.get('chief_complaint', '')}")
        text_parts.append(f"Medical History: {patient_info.get('medical_history', '')}")
        text_parts.append(f"Dental History: {patient_info.get('dental_history', '')}")
        text_parts.append(f"Current Medications: {patient_info.get('current_medications', '')}")
        text_parts.append(f"Additional Info: {patient_info.get('additional_info', '')}")
    
    # If no matches or empty text parts, fall back to converting the entire record to JSON
    if not text_parts:
        return json.dumps(patient_record)
    
    return "\n".join(text_parts)

def store_record(metadata, patient_record, file_id):
    """
    Store a patient record in Pinecone with the given metadata and patient_record.

    Parameters:
    - metadata (dict): Dictionary containing metadata (e.g., File_ID, Update_Type, etc.).
    - patient_record (dict): Dictionary containing findings (e.g., case_summary, clinical_findings, etc.).
    - file_id (str): Unique identifier for the record (e.g., "IOP336").
    """
    # Load API keys from .env file
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")
    
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)

    # Specify the index name
    index_name = "trust"

    # Connect to the index
    index = pc.Index(index_name)
    
    # Extract content type for proper text generation
    content_type = metadata.get("Content_Type", "")
    
    # Generate text for embedding
    text_for_embedding = generate_text_for_embedding(patient_record, content_type)
    
    # Generate embedding vector
    embedding_vector = create_embeddings(text_for_embedding)
    
    # Extract key metadata for filtering
    key_metadata = extract_key_metadata(metadata, patient_record)
    
    # Serialize patient_record to a JSON string
    patient_record_str = json.dumps(patient_record)
    
    # Check if serialized record fits within metadata size limit (40KB)
    if len(patient_record_str) > 40000:
        print("Warning: Patient record exceeds Pinecone metadata size limit.")
        print("Storing truncated record in metadata. Full record should be stored elsewhere.")
        # Store a reference or truncated version
        key_metadata["record_reference"] = f"full_record:{file_id}"
        # You might want to implement additional storage for the full record
    else:
        # Add the full patient record to metadata
        key_metadata["patient_record"] = patient_record_str

    # Upsert the record with the file_id as ID, embedding vector, and metadata
    index.upsert(vectors=[(file_id, embedding_vector, key_metadata)])
    print(f"Successfully stored record with File_ID: {file_id}")

def upload_or_update_protocol(protocol_name, protocol_content, protocol_id=None):
    """Upload or update a protocol document in Pinecone."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Create text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    # Generate protocol_id if not provided
    if not protocol_id:
        protocol_id = f"protocol_{protocol_name.lower().replace(' ', '_')}"
    
    # First, try to delete existing protocol chunks if this is an update
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX", "trust"))
        
        # Query for existing protocol chunks to delete
        existing_filter = {"Protocol_ID": {"$eq": protocol_id}}
        results = index.query(
            vector=[0.0] * 1536,  # Dummy vector for metadata-only query
            filter=existing_filter,
            top_k=100,
            include_metadata=True
        )
        
        # Get IDs to delete
        ids_to_delete = [match['id'] for match in results['matches']]
        
        if ids_to_delete:
            # Delete existing chunks before uploading new ones
            index.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} existing chunks for protocol: {protocol_name}")
    except Exception as e:
        print(f"Warning: Error checking/deleting existing protocol: {e}")
    
    # Split into chunks for better retrieval
    chunks = text_splitter.split_text(protocol_content)
    chunk_ids = []
    
    # Upload each chunk
    for i, chunk in enumerate(chunks):
        chunk_id = f"{protocol_id}_chunk_{i}"
        chunk_ids.append(chunk_id)
        
        metadata = {
            "Protocol_ID": protocol_id,
            "Protocol_Name": protocol_name,
            "Protocol_Section": i,
            "Total_Sections": len(chunks),
            "Content_Type": "Protocol",
            "Timestamp": datetime.now().isoformat(),
        }
        
        # Create record and upload
        protocol_record = {"protocol_content": chunk}
        response = store_record(
            metadata=metadata,
            patient_record=protocol_record,
            file_id=chunk_id
        )
        
        print(f"Uploaded chunk {i+1}/{len(chunks)} for protocol: {protocol_name}")
    
    return {
        "protocol_id": protocol_id,
        "chunks_uploaded": len(chunks),
        "chunk_ids": chunk_ids
    }

#######################SUPPLIER DATA #######################################################

def store_supplier_record(metadata, supplier_record, chunk_id):
    """Modified version of store_record for supplier data."""
    
    # Generate text for embedding (simpler than patient records)
    text_for_embedding = supplier_record.get("content", "")
    
    # Generate embedding
    embedding_vector = create_embeddings(text_for_embedding)
    
    # Prepare metadata (no size limits to worry about like patient records)
    final_metadata = metadata.copy()
    final_metadata["supplier_record"] = json.dumps(supplier_record)
    
    # Store in Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("trust")  # Same index, different namespace or metadata
    
    index.upsert(vectors=[(chunk_id, embedding_vector, final_metadata)])
    print(f"Stored supplier chunk: {chunk_id}")

def store_supplier_data(supplier_id, supplier_name, content_file_path, supplier_info=None):
    """
    Store supplier product catalog and information in Pinecone.
    
    Parameters:
    - supplier_id (str): Unique identifier for supplier (e.g., "dental_corp_123")
    - supplier_name (str): Display name of supplier
    - content_file_path (str): Path to the 4MB text file
    - supplier_info (dict): Optional additional supplier metadata
    """
    
    # Read the supplier content
    with open(content_file_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()
    
    # Use your existing text splitter with supplier-optimized settings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Larger chunks for product info
        chunk_overlap=200,  # More overlap to maintain context
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # Smart separators
    )
    
    chunks = text_splitter.split_text(raw_content)
    
    # Store each chunk with supplier-specific metadata
    for i, chunk in enumerate(chunks):
        chunk_id = f"{supplier_id}_chunk_{i}"
        
        # Detect content type automatically
        content_type = detect_content_type(chunk)
        
        metadata = {
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "Content_Type": "Supplier_Data",
            "Data_Category": content_type,  # "Product", "Company_Info", "Policy", etc.
            "Chunk_Number": i,
            "Total_Chunks": len(chunks),
            "Timestamp": datetime.now().isoformat(),
            "File_Source": os.path.basename(content_file_path)
        }
        
        # Add any additional supplier info
        if supplier_info:
            metadata.update(supplier_info)
        
        # Create the record structure
        supplier_record = {
            "content": chunk,
            "supplier_context": {
                "supplier_id": supplier_id,
                "supplier_name": supplier_name,
                "content_category": content_type
            }
        }
        
        # Store using your existing function (slightly modified)
        store_supplier_record(metadata, supplier_record, chunk_id)

def detect_content_type(chunk_text):
    """Automatically categorize content type based on keywords."""
    text_lower = chunk_text.lower()
    
    # Product-related keywords
    if any(word in text_lower for word in ['product', 'item', 'catalog', 'price', 'sku', 'model', 'specification']):
        return "Product"
    
    # Company info keywords  
    elif any(word in text_lower for word in ['about us', 'mission', 'vision', 'company', 'founded', 'history']):
        return "Company_Info"
    
    # Policy/procedure keywords
    elif any(word in text_lower for word in ['policy', 'procedure', 'terms', 'conditions', 'warranty', 'return']):
        return "Policy"
    
    # Service keywords
    elif any(word in text_lower for word in ['service', 'support', 'installation', 'training', 'consultation']):
        return "Service"
    
    else:
        return "General"

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Store records in Pinecone")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Record command
    record_parser = subparsers.add_parser("record", help="Store a patient record")
    record_parser.add_argument("file", help="JSON file path containing the record")
    
    # Protocol command
    protocol_parser = subparsers.add_parser("protocol", help="Store protocol documents")
    protocol_parser.add_argument("--directory", help="Directory containing protocol files")
    protocol_parser.add_argument("--file", help="Single protocol file to upload")

    # Supplier command
    supplier_parser = subparsers.add_parser("supplier", help="Store supplier data")
    supplier_parser.add_argument("--supplier_id", required=True, help="Unique supplier identifier")
    supplier_parser.add_argument("--supplier_name", required=True, help="Supplier display name")
    supplier_parser.add_argument("--file", required=True, help="Supplier data file path")
    supplier_parser.add_argument("--info", help="Additional supplier info as JSON string")
    
    args = parser.parse_args()
    
    if args.command == "record":
        # Existing record storing functionality
        if not os.path.isfile(args.file):
            print(f"Error: File '{args.file}' does not exist")
            sys.exit(1)
            
        try:
            # Read and parse the JSON file
            with open(args.file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Extract metadata and findings
            metadata = data.get("metadata", {})
            patient_record = data.get("patient_record", {})
            
            # Extract File_ID from metadata
            file_id = metadata.get("File_ID")
            if not file_id:
                raise ValueError("File_ID not found in metadata")
            
            # Store the record in Pinecone
            store_record(metadata, patient_record, file_id)
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in file - {str(e)}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
            
    elif args.command == "supplier":
        if not os.path.isfile(args.file):
            print(f"Error: File '{args.file}' does not exist")
            sys.exit(1)
        
        supplier_info = None
        if args.info:
            try:
                supplier_info = json.loads(args.info)
            except json.JSONDecodeError:
                print("Error: Invalid JSON in --info parameter")
                sys.exit(1)
        
        store_supplier_data(args.supplier_id, args.supplier_name, args.file, supplier_info)
        print(f"Successfully stored supplier data for: {args.supplier_name}")
    
    elif args.command == "protocol":
        if args.directory:
            # Process all protocol files in the directory
            if not os.path.isdir(args.directory):
                print(f"Error: Directory '{args.directory}' does not exist")
                sys.exit(1)
                
            protocol_files = [f for f in os.listdir(args.directory) if f.endswith('.txt')]
            if not protocol_files:
                print(f"No .txt protocol files found in '{args.directory}'")
                sys.exit(1)
                
            for filename in protocol_files:
                protocol_path = os.path.join(args.directory, filename)
                protocol_name = os.path.splitext(filename)[0]
                
                # Read protocol content
                with open(protocol_path, 'r', encoding='utf-8') as f:
                    protocol_content = f.read()
                
                # Upload protocol
                response = upload_or_update_protocol(protocol_name, protocol_content)
                print(f"Uploaded protocol '{protocol_name}': {response['chunks_uploaded']} chunks")
                
        elif args.file:
            # Process a single protocol file
            if not os.path.isfile(args.file):
                print(f"Error: File '{args.file}' does not exist")
                sys.exit(1)
                
            protocol_name = os.path.splitext(os.path.basename(args.file))[0]
            
            # Read protocol content
            with open(args.file, 'r', encoding='utf-8') as f:
                protocol_content = f.read()
                
            # Upload protocol
            response = upload_or_update_protocol(protocol_name, protocol_content)
            print(f"Uploaded protocol '{protocol_name}': {response['chunks_uploaded']} chunks")
        
        else:
            print("Error: Either --directory or --file must be specified")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
