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

# Add a global variable for pinecone index name
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "trust")  # Use explicit variable

# Separate index for suppliers
SUPPLIER_INDEX_NAME = "njor"

class FastPineconeRetrieval:
    """Optimized version of pinecone_retrieval with async capabilities and supplier support"""
    
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

    # ========================================
    # SUPPLIER-SPECIFIC METHODS (NEW)
    # ========================================
    
    async def search_supplier_products_async(
        self,
        supplier_id: str,
        query: str,
        top_k: int = 10,
        index_name: str = SUPPLIER_INDEX_NAME
    ) -> List[Dict]:
        """
        Search for products from a specific supplier.
        
        Args:
            supplier_id: Unique supplier identifier
            query: Search query (e.g., "composite resin", "dental implants")
            top_k: Number of results to return
            index_name: Pinecone index name
            
        Returns:
            List of supplier product records
        """
        try:
            logger.info(f"Searching supplier {supplier_id} products for: {query}")
            
            # Get the appropriate index
            index = self._get_index(index_name)
            
            # Generate embedding for query
            query_embedding = await self.create_embeddings_async(query)
            
            # Build filter for supplier products
            filter_dict = {
                "supplier_id": {"$eq": supplier_id},
                "Content_Type": {"$eq": "Supplier_Data"}
            }
            
            # Query Pinecone
            results = await asyncio.to_thread(
                index.query,
                vector=query_embedding,
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
            
            return self._process_supplier_results(results)
            
        except Exception as e:
            logger.error(f"Error in search_supplier_products_async: {str(e)}")
            return []
    
    async def search_supplier_by_category_async(
        self,
        supplier_id: str,
        category: str,
        top_k: int = 10,
        index_name: str = SUPPLIER_INDEX_NAME
    ) -> List[Dict]:
        """
        Search supplier data by specific category.
        
        Args:
            supplier_id: Unique supplier identifier
            category: Data category ("Product", "Company_Info", "Policy", "Service", "General")
            top_k: Number of results to return
            index_name: Pinecone index name
            
        Returns:
            List of supplier records for that category
        """
        try:
            logger.info(f"Searching supplier {supplier_id} category: {category}")
            
            # Get the appropriate index
            index = self._get_index(index_name)
            
            # Build filter for supplier category
            filter_dict = {
                "supplier_id": {"$eq": supplier_id},
                "Content_Type": {"$eq": "Supplier_Data"},
                "Data_Category": {"$eq": category}
            }
            
            # Query Pinecone (using placeholder vector for metadata-only search)
            results = await asyncio.to_thread(
                index.query,
                vector=[0.0] * 1536,  # Placeholder vector
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
            
            return self._process_supplier_results(results)
            
        except Exception as e:
            logger.error(f"Error in search_supplier_by_category_async: {str(e)}")
            return []
    
    async def get_supplier_overview_async(
        self,
        supplier_id: str,
        index_name: str = SUPPLIER_INDEX_NAME
    ) -> Dict[str, Any]:
        """
        Get a comprehensive overview of a supplier's data.
        
        Args:
            supplier_id: Unique supplier identifier
            index_name: Pinecone index name
            
        Returns:
            Dictionary with supplier overview and sample content from each category
        """
        try:
            logger.info(f"Getting overview for supplier: {supplier_id}")
            
            # Get samples from each category
            categories = ["Product", "Company_Info", "Policy", "Service", "General"]
            overview = {
                "supplier_id": supplier_id,
                "categories": {},
                "total_chunks": 0
            }
            
            for category in categories:
                category_results = await self.search_supplier_by_category_async(
                    supplier_id, category, top_k=3, index_name=index_name
                )
                
                if category_results:
                    overview["categories"][category] = {
                        "count": len(category_results),
                        "sample_content": category_results[0].get("content", "")[:200] + "..."
                    }
                    overview["total_chunks"] += len(category_results)
            
            # Get supplier name from first available record
            first_result = await self.search_supplier_products_async(
                supplier_id, "", top_k=1, index_name=index_name
            )
            
            if first_result:
                overview["supplier_name"] = first_result[0].get("metadata", {}).get("supplier_name", "")
            
            return overview
            
        except Exception as e:
            logger.error(f"Error in get_supplier_overview_async: {str(e)}")
            return {"error": str(e)}
    
    async def search_all_suppliers_async(
        self,
        query: str,
        top_k: int = 10,
        index_name: str = SUPPLIER_INDEX_NAME
    ) -> List[Dict]:
        """
        Search across all suppliers for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            index_name: Pinecone index name
            
        Returns:
            List of supplier records from all suppliers
        """
        try:
            logger.info(f"Searching all suppliers for: {query}")
            
            # Get the appropriate index
            index = self._get_index(index_name)
            
            # Generate embedding for query
            query_embedding = await self.create_embeddings_async(query)
            
            # Build filter for all supplier data
            filter_dict = {
                "Content_Type": {"$eq": "Supplier_Data"}
            }
            
            # Query Pinecone
            results = await asyncio.to_thread(
                index.query,
                vector=query_embedding,
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
            
            return self._process_supplier_results(results, include_supplier_info=True)
            
        except Exception as e:
            logger.error(f"Error in search_all_suppliers_async: {str(e)}")
            return []
    
    def _process_supplier_results(self, results, include_supplier_info=False):
        """Process supplier query results into a consistent format."""
        records = []
        
        for match in results['matches']:
            metadata = match['metadata']
            
            # Extract supplier record from metadata
            supplier_record = {}
            if "supplier_record" in metadata:
                try:
                    supplier_record = json.loads(metadata["supplier_record"])
                except json.JSONDecodeError:
                    supplier_record = {"error": "Could not parse supplier record"}
            
            # Build the record
            record = {
                "id": match['id'],
                "score": match['score'],
                "metadata": {k: v for k, v in metadata.items() if k != "supplier_record"},
                "content": supplier_record.get("content", ""),
                "supplier_context": supplier_record.get("supplier_context", {}),
                "category": metadata.get("Data_Category", "General")
            }
            
            # Include supplier info for cross-supplier searches
            if include_supplier_info:
                record["supplier_id"] = metadata.get("supplier_id", "")
                record["supplier_name"] = metadata.get("supplier_name", "")
            
            records.append(record)
        
        return records

    # ========================================
    # EXISTING PATIENT RECORD METHODS (UNCHANGED)
    # ========================================
    
    async def retrieve_record_by_id_async(self, case_id: str, index_name: str = PINECONE_INDEX_NAME) -> Optional[Dict]:
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
        index_name: str = PINECONE_INDEX_NAME
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
        index_name: str = PINECONE_INDEX_NAME
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
        index_name: str = PINECONE_INDEX_NAME
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
        index_name: str = PINECONE_INDEX_NAME
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
        index_name: str = PINECONE_INDEX_NAME
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
    
    # ========================================
    # UNIFIED RETRIEVAL METHOD (UPDATED)
    # ========================================
    
    async def retrieve_records_async(
        self, 
        search_type: str, 
        query: str, 
        practice_id: Optional[str] = None,
        top_k: int = 10,
        index_name: str = PINECONE_INDEX_NAME,
        supplier_id: Optional[str] = None  # NEW: For supplier searches
    ) -> Any:
        """
        Unified async method for all record retrieval operations.
        Now supports supplier searches.
        """
        
        # Force supplier index for supplier searches
        if search_type in ["supplier_products", "supplier_category", "supplier_overview", "all_suppliers"]:
            index_name = SUPPLIER_INDEX_NAME
        
        # Print the search type and query
        print("--------------------------------")
        print(f"Index name: {index_name}", flush=True)
        print(f"Trying to retrieve records with search_type: {search_type}, query: {query}", flush=True)
        if supplier_id:
            print(f"Supplier ID: {supplier_id}", flush=True)
        sys.stderr.write(f"Trying to retrieve records with search_type: {search_type}, query: {query}\n")
        sys.stderr.flush()
        print("--------------------------------")
        
        logger.info(f"Trying to retrieve records with search_type: {search_type}, query: {query}")
        logger.debug(f"Trying to retrieve records with search_type: {search_type}, query: {query}")
        
        try:
            # NEW: Supplier-specific search types
            if search_type == "supplier_products":
                if not supplier_id:
                    logger.error("supplier_id required for supplier_products search")
                    return []
                return await self.search_supplier_products_async(supplier_id, query, top_k, index_name)
                
            elif search_type == "supplier_category":
                if not supplier_id:
                    logger.error("supplier_id required for supplier_category search")
                    return []
                return await self.search_supplier_by_category_async(supplier_id, query, top_k, index_name)
                
            elif search_type == "supplier_overview":
                if not supplier_id:
                    logger.error("supplier_id required for supplier_overview search")
                    return []
                result = await self.get_supplier_overview_async(supplier_id, index_name)
                return [result] if result else []
                
            elif search_type == "all_suppliers":
                return await self.search_all_suppliers_async(query, top_k, index_name)
            
            # EXISTING: Patient record search types
            elif search_type == "id":
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