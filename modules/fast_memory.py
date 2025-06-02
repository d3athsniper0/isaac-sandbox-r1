# modules/fast_memory.py

import os
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio
import httpx
from pinecone import Pinecone
from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import redis
import ssl

from modules.fast_pinecone_retrieval import FastPineconeRetrieval #type: ignore

logger = logging.getLogger(__name__)

class FastMemoryManager:
    """Optimized version of MemoryManager with async capabilities"""
    
    # Modify the __init__ method to use a shared HTTP client
    def __init__(
        self, 
        openai_client: Optional[AsyncOpenAI] = None,
        pinecone_client: Optional[Pinecone] = None,
        index_name: str = None,
        http_client: Optional[httpx.AsyncClient] = None
    ):
        # Use provided clients or create new ones
        self.openai_client = openai_client or AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = pinecone_client or Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Use shared HTTP client with better connection pooling
        self.http_client = http_client or httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        # Initialize OpenAI embeddings for LangChain compatibility
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Attach an instance of FastPineconeRetrieval
        self.fast_pinecone_retrieval = FastPineconeRetrieval(pinecone_client)
        
        # Set the index name for patient data
        self.index_name = index_name or os.getenv("PINECONE_INDEX", "trust")
        
        # Get the Pinecone index
        pinecone_index = self.pc.Index(self.index_name)

        # Initialize the vector store
        self.vectorstore = PineconeVectorStore(
            index=pinecone_index,
            embedding=self.embeddings,
            text_key="text"
        )
        
        # Initialize Redis for conversation history
        try:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                self.redis_client = redis.from_url(
                    redis_url, 
                    ssl_cert_reqs=ssl.CERT_NONE,
                    decode_responses=True
                )
            else:
                self.redis_client = None
                logger.warning("Redis URL not found. Conversation history will not persist.")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis_client = None
    
        # Create a semaphore to limit concurrent operations
        self.semaphore = asyncio.Semaphore(5)
    
    async def create_embeddings_async(self, text_content: str) -> List[float]:
        """Async wrapper for generating embeddings."""
        async with self.semaphore:
            response = await self.openai_client.embeddings.create(
                input=text_content,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
    
    async def store_patient_data_async(self, patient_id: str, data: Dict[str, Any]) -> str:
        """Async version of store_patient_data."""
        try:
            # Convert data to string if needed
            if isinstance(data, dict):
                data_str = json.dumps(data)
            else:
                data_str = str(data)
            
            logger.info(f"Storing data for patient {patient_id}, data length: {len(data_str)}")
            
            # Split text into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            texts = text_splitter.split_text(data_str)
            logger.info(f"Split into {len(texts)} chunks")
            
            # Ensure we're matching the metadata format
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "Patient_Case_ID": patient_id,
                        "Timestamp": datetime.now().isoformat(),
                        "File_ID": f"{patient_id}-{i}",
                        "Content_Type": "Patient_Info"
                    }
                ) for i, chunk in enumerate(texts)
            ]
            
            # Add documents to vector store (this is a synchronous operation)
            # Use asyncio.to_thread to avoid blocking the event loop
            ids = await asyncio.to_thread(
                self.vectorstore.add_documents,
                documents
            )
            
            logger.info(f"Added {len(ids)} documents to vector store")
            
            return f"Added {len(ids)} chunks to Pinecone for patient {patient_id}"
        except Exception as e:
            logger.error(f"Error storing patient data: {str(e)}")
            return f"Error: {str(e)}"
    
    async def retrieve_patient_data_async(self, patient_id: str, query: str, k: int = 5) -> List[Document]:
        """Async version of retrieve_patient_data."""
        try:
            # Search by metadata filter + similarity (this is a synchronous operation)
            # Use asyncio.to_thread to avoid blocking the event loop
            results = await asyncio.to_thread(
                self.vectorstore.similarity_search,
                query=query,
                k=k,
                filter={"Patient_Case_ID": patient_id}
            )
            
            logger.info(f"Retrieved {len(results)} documents for patient {patient_id}")
            
            # Process documents to ensure they have text key
            processed_docs = []
            for doc in results:
                # If document doesn't have text attribute, add it
                if not hasattr(doc, 'text') or not doc.text:
                    # Assuming page_content contains the document text
                    setattr(doc, 'text', doc.page_content)
                processed_docs.append(doc)
                
            return processed_docs
        except Exception as e:
            logger.error(f"Error retrieving patient data: {str(e)}")
            return []
    
    async def search_all_patients_async(self, query: str, k: int = 5) -> List[Document]:
        """Async version of search_all_patients."""
        # Search without metadata filter
        results = await asyncio.to_thread(
            self.vectorstore.similarity_search,
            query=query,
            k=k
        )
        
        return results
    
    async def store_conversation_async(self, user_id: str, messages: List[Dict[str, str]]) -> bool:
        """Async version of store_conversation."""
        if self.redis_client is None:
            return False
        
        try:
            key = f"conversation:{user_id}"
            
            # Use asyncio.to_thread for Redis operations
            await asyncio.to_thread(
                self.redis_client.set,
                key, 
                json.dumps(messages)
            )
            
            # Set a TTL of 30 days
            await asyncio.to_thread(
                self.redis_client.expire,
                key, 
                60 * 60 * 24 * 30
            )
            
            return True
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            return False
    
    async def get_conversation_async(self, user_id: str) -> List[Dict[str, str]]:
        """Async version of get_conversation."""
        if self.redis_client is None:
            return []
        
        try:
            key = f"conversation:{user_id}"
            
            # Use asyncio.to_thread for Redis operations
            data = await asyncio.to_thread(
                self.redis_client.get,
                key
            )
            
            if data:
                return json.loads(data)
            return []
        except Exception as e:
            logger.error(f"Error retrieving conversation: {e}")
            return []
    
    # Maintain compatibility with original synchronous methods
    def store_patient_data(self, patient_id: str, data: Dict[str, Any]) -> str:
        """Synchronous wrapper for backward compatibility."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.store_patient_data_async(patient_id, data))
    
    def retrieve_patient_data(self, patient_id: str, query: str, k: int = 5) -> List[Document]:
        """Synchronous wrapper for backward compatibility."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.retrieve_patient_data_async(patient_id, query, k))
    
    def search_all_patients(self, query: str, k: int = 5) -> List[Document]:
        """Synchronous wrapper for backward compatibility."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.search_all_patients_async(query, k))
    
    # Deprecated: Do not use in an async context!
    def _store_conversation(self, user_id: str, messages: List[Dict[str, str]]) -> bool:
        """Synchronous wrapper for backward compatibility."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.store_conversation_async(user_id, messages))
    
    async def store_conversation(self, user_id: str, messages: List[Dict[str, str]]) -> bool:
        """
        Async alias to store conversation.
        This allows legacy code that calls store_conversation to work 
        by simply invoking store_conversation_async.
        """
        return await self.store_conversation_async(user_id, messages)
    
    # Corrected implementation
    def get_conversation(self, user_id: str) -> List[Dict[str, str]]:
        """Synchronous wrapper for backward compatibility."""
        try:
            # Try using asyncio.run (for sync context)
            return asyncio.run(self.get_conversation_async(user_id))
        except RuntimeError:
            try:
                # Already in event loop, use nest_asyncio
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.get_conversation_async(user_id))
            except Exception as e:
                logger.error(f"Failed to get conversation: {e}")
                return []  # Fallback to empty list on any error