# modules/fast_pinecone_retrieval.py

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)
load_dotenv()

class FastPineconeRetrieval:
    """Optimized version of pinecone_retrieval with async capabilities"""
    
    def __init__(self, pinecone_client: Optional[Pinecone] = None):
        # Initialize Pinecone client
        self.pc = pinecone_client or Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Initialize OpenAI client for embeddings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Create a semaphore to limit concurrent embedding operations
        self.semaphore = asyncio.Semaphore(5)
    
    def _get_index(self, index_name: str):
        """Get Pinecone index with error handling."""
        try:
            return self.pc.Index(index_name)
        except Exception as e:
            logger.error(f"Error connecting to Pinecone index {index_name}: {e}")
            raise
    
    async def create_embeddings_async(self, text_content: str) -> List[float]:
        """
        Async wrapper for generating embeddings.
        """
        async with self.semaphore:
            # Use AsyncOpenAI client for better performance
            client = AsyncOpenAI(api_key=self.openai_api_key)
            response = await client.embeddings.create(
                input=text_content,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
    
    # Maintain compatibility with original method
    def create_embeddings(self, text_content: str) -> List[float]:
        """Synchronous wrapper for backward compatibility."""
        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            input=text_content,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    async def retrieve_record_by_id_async(self, case_id: str, index_name: str = "trust") -> Optional[Dict]:
        """Async version of retrieve_record_by_id."""
        try:
            # Get the appropriate index
            index = self._get_index(index_name)
            
            # Use asyncio.to_thread to avoid blocking the event loop
            result = await asyncio.to_thread(
                index.fetch,
                ids=[case_id]
            )
            
            if case_id in result['vectors']:
                metadata = result['vectors'][case_id]['metadata']
                
                # Extract original metadata (excluding the "patient_record" field)
                original_metadata = {k: v for k, v in metadata.items() if k != "patient_record"}
                
                # Parse patient_record from the stored JSON string
                if "patient_record" in metadata:
                    patient_record_str = metadata["patient_record"]
                    patient_record = json.loads(patient_record_str)
                else:
                    patient_record = {"error": "Full record not available in vector store"}
                
                return {"metadata": original_metadata, "patient_record": patient_record}
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in retrieve_record_by_id_async: {str(e)}")
            return None
    
    async def retrieve_records_by_patient_id_async(
        self, 
        patient_case_id: str, 
        practice_id: Optional[str] = None, 
        index_name: str = "trust"
    ) -> List[Dict]:
        """Async version of retrieve_records_by_patient_id."""
        try:
            # Get the appropriate index
            index = self._get_index(index_name)
            
            # Build filter
            filter_dict = {"Patient_Case_ID": {"$eq": patient_case_id}}
            
            # Add practice ID filter if provided
            if practice_id:
                filter_dict["Practice_ID"] = {"$eq": practice_id}
            
            # Use asyncio.to_thread to avoid blocking the event loop
            results = await asyncio.to_thread(
                index.query,
                vector=[0.0] * 1536,  # Placeholder vector
                filter=filter_dict,
                top_k=100,
                include_metadata=True
            )
            
            return self._process_query_results(results)
                
        except Exception as e:
            logger.error(f"Error in retrieve_records_by_patient_id_async: {str(e)}")
            return []
    
    async def retrieve_records_by_patient_name_async(
        self, 
        patient_name: str, 
        practice_id: Optional[str] = None, 
        top_k: int = 100, 
        index_name: str = "trust"
    ) -> List[Dict]:
        """Async version of retrieve_records_by_patient_name."""
        
        try:
            # Get the appropriate index
            index = self._get_index(index_name)
            
            # Build filter
            filter_dict = {"Patient_Name": {"$eq": patient_name}}
            
            # Add practice ID filter if provided
            if practice_id:
                filter_dict["Practice_ID"] = {"$eq": practice_id}
            
            # Use asyncio.to_thread to avoid blocking the event loop
            results = await asyncio.to_thread(
                index.query,
                vector=[0.0] * 1536,  # Placeholder vector
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
            
            records = self._process_query_results(results)
            
            # If no exact matches, try semantic search
            if not records:
                # Generate embedding for patient name
                query_embedding = await self.create_embeddings_async(f"Patient named {patient_name}")
                
                # Build filter for practice_id only
                filter_dict = {}
                if practice_id:
                    filter_dict["Practice_ID"] = {"$eq": practice_id}
                
                # Query with embedding
                results = await asyncio.to_thread(
                    index.query,
                    vector=query_embedding,
                    filter=filter_dict if filter_dict else None,
                    top_k=top_k,
                    include_metadata=True
                )
                
                records = self._process_query_results(results, filter_name=patient_name)
            
            return records
                
        except Exception as e:
            logger.error(f"Error in retrieve_records_by_patient_name_async: {str(e)}")
            return []
    
    async def search_by_text_async(
        self, 
        query_text: str, 
        practice_id: Optional[str] = None, 
        top_k: int = 10, 
        index_name: str = "trust"
    ) -> List[Dict]:
        """Async version of search_by_text."""
        try:
            # Get the appropriate index
            index = self._get_index(index_name)
            
            # Generate embedding for query
            query_embedding = await self.create_embeddings_async(query_text)
            
            # Build filter if practice_id is provided
            filter_dict = {}
            if practice_id:
                filter_dict["Practice_ID"] = {"$eq": practice_id}
            
            # Query Pinecone
            results = await asyncio.to_thread(
                index.query,
                vector=query_embedding,
                filter=filter_dict if filter_dict else None,
                top_k=top_k,
                include_metadata=True
            )
            
            return self._process_query_results(results)
                
        except Exception as e:
            logger.error(f"Error in search_by_text_async: {str(e)}")
            return []
    
    # Add the missing methods for medication and condition searches
    async def search_by_medication_async(
        self, 
        medication: str, 
        practice_id: Optional[str] = None, 
        top_k: int = 10, 
        index_name: str = "trust"
    ) -> List[Dict]:
        """Async version of search_by_medication."""
        try:
            # Get the appropriate index
            index = self._get_index(index_name)
            
            # Build filter
            filter_dict = {"medications": {"$in": [medication]}}
            
            # Add practice ID filter if provided
            if practice_id:
                filter_dict["Practice_ID"] = {"$eq": practice_id}
            
            # Query Pinecone
            results = await asyncio.to_thread(
                index.query,
                vector=[0.0] * 1536,  # Placeholder vector
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
            
            return self._process_query_results(results)
                
        except Exception as e:
            logger.error(f"Error in search_by_medication_async: {str(e)}")
            return []
    
    async def search_by_condition_async(
        self, 
        condition: str, 
        practice_id: Optional[str] = None, 
        top_k: int = 10, 
        index_name: str = "trust"
    ) -> List[Dict]:
        """Async version of search_by_condition."""
        try:
            # Get the appropriate index
            index = self._get_index(index_name)
            
            # Build filter
            filter_dict = {"conditions": {"$in": [condition]}}
            
            # Add practice ID filter if provided
            if practice_id:
                filter_dict["Practice_ID"] = {"$eq": practice_id}
            
            # Query Pinecone
            results = await asyncio.to_thread(
                index.query,
                vector=[0.0] * 1536,  # Placeholder vector
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
            
            return self._process_query_results(results)
                
        except Exception as e:
            logger.error(f"Error in search_by_condition_async: {str(e)}")
            return []
    
    def _process_query_results(self, results, filter_name=None):
        """Process query results into a consistent format."""
        records = []
        seen_patient_ids = set()
        
        for match in results['matches']:
            metadata = match['metadata']
            
            # If filtering by name, check if the name contains the search term
            if filter_name and "Patient_Name" in metadata:
                if filter_name.lower() not in metadata["Patient_Name"].lower():
                    continue
            
            # Skip duplicates if we're tracking by patient ID
            patient_case_id = metadata.get("Patient_Case_ID")
            if filter_name and patient_case_id:
                if patient_case_id in seen_patient_ids:
                    continue
                seen_patient_ids.add(patient_case_id)
            
            # Extract original metadata
            original_metadata = {k: v for k, v in metadata.items() if k != "patient_record"}
            
            # Parse patient_record from the stored JSON string if available
            if "patient_record" in metadata:
                patient_record_str = metadata["patient_record"]
                patient_record = json.loads(patient_record_str)
            else:
                # Handle case where record was too large for metadata
                patient_record = {"error": "Full record not available in vector store"}
            
            records.append({
                "id": match['id'],
                "metadata": original_metadata,
                "patient_record": patient_record,
                "score": match['score']
            })
        
        return records
    
    # Unified async method for all record retrieval operations
    async def retrieve_records_async(
        self, 
        search_type: str, 
        query: str, 
        practice_id: Optional[str] = None,
        top_k: int = 10,
        index_name: str = "trust"
    ) -> Any:
        """
        Unified async method for all record retrieval operations.
        """
        
        # Print the search type and query
        print("--------------------------------")
        print(f"Trying to retrieve records with search_type: {search_type}, query: {query}", flush=True)
        sys.stderr.write(f"Trying to retrieve records with search_type: {search_type}, query: {query}\n")
        sys.stderr.flush()
        print("--------------------------------")
        
        logger.info(f"Trying to retrieve records with search_type: {search_type}, query: {query}")
        logger.debug(f"Trying to retrieve records with search_type: {search_type}, query: {query}")
        
        try:
            if search_type == "id":
                result = await self.retrieve_record_by_id_async(query, index_name)
                return [result] if result else []
                
            elif search_type == "patient":
                return await self.retrieve_records_by_patient_id_async(query, practice_id, index_name)
                
            elif search_type == "patient_name":
                return await self.retrieve_records_by_patient_name_async(query, practice_id, top_k, index_name)
                
            elif search_type == "text":
                return await self.search_by_text_async(query, practice_id, top_k, index_name)
                
            elif search_type == "medication":
                return await self.search_by_medication_async(query, practice_id, top_k, index_name)
                
            elif search_type == "condition":
                return await self.search_by_condition_async(query, practice_id, top_k, index_name)
                
            else:
                logger.error(f"Invalid search_type: {search_type}")
                return []
                
        except Exception as e:
            logger.error(f"Error in retrieve_records_async: {str(e)}")
            return []