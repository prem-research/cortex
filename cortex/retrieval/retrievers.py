from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import pickle
from nltk.tokenize import word_tokenize
import json
import logging

logger = logging.getLogger(__name__)

class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""
    def __init__(self, collection_name: str = "memories"):
        """Initialize ChromaDB retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        try:
            self.client = chromadb.Client(Settings(allow_reset=True))
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            logger.error(f"Error initializing ChromaDB collection '{collection_name}': {e}")

            
        # Map of doc_id -> content hash for tracking changes
        self.content_hashes = {}
        
    def _hash_content(self, document: str) -> str:
        """Generate a simple hash for content to detect changes."""
        import hashlib
        return hashlib.md5(document.encode()).hexdigest()
        
    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB with upsert logic.
        
        Args:
            document: Text content to add
            metadata: Dictionary of metadata
            doc_id: Unique identifier for the document
        """
        if not document or not doc_id:
            logger.warning(f"Skipping add for empty document or missing ID")
            return
            
        content_hash = self._hash_content(document + str(metadata))
        
        existing_hash = self.content_hashes.get(doc_id)
        if existing_hash == content_hash:
            # Same ID and same content - no need to update
            logger.debug(f"Skipping add for unchanged document: {doc_id}")
            return
            
        # Different content or new document - process metadata
        processed_metadata = {}
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue
                
            # Ensure links dictionary is also dumped as JSON string
            if key == 'links' and isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)
        
        # If ID exists but content changed, delete first
        if doc_id in self.content_hashes:
            try:
                self.collection.delete(ids=[doc_id])
                logger.debug(f"Deleted existing document to update: {doc_id}")
            except Exception as e:
                logger.error(f"Error deleting document before update: {e}")
                
        try:
            self.collection.add(
                documents=[document],
                metadatas=[processed_metadata],
                ids=[doc_id]
            )
            # Update hash cache
            self.content_hashes[doc_id] = content_hash
            logger.debug(f"Added/updated document: {doc_id}")
        except Exception as e:
            logger.error(f"Error adding document to '{self.collection_name}': {e}")
        
    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.
        
        Args:
            doc_id: ID of document to delete
        """
        if not doc_id:
            return
            
        try:
            self.collection.delete(ids=[doc_id])
            # Remove from hash cache
            if doc_id in self.content_hashes:
                del self.content_hashes[doc_id]
        except Exception as e:
            logger.error(f"Error deleting document from '{self.collection_name}': {e}")
    
    def save(self, file_path, embeddings_path=None):
        """Save the current collection state for later restoration."""
        # Save content hashes along with other data
        data = {
            'content_hashes': self.content_hashes
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, file_path, embeddings_path=None):
        """Load a previously saved collection state."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if 'content_hashes' in data:
                self.content_hashes = data['content_hashes']
        return self

    def search(self, query: str, k: int = 5, where_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents, optionally filtering by metadata.
        
        Args:
            query: Query text
            k: Number of results to return
            where_filter: Optional dictionary for metadata filtering (e.g., {"tag": "important"})
            
        Returns:
            List[Dict]: List of results, each containing id, metadata, distance, and document.
        """
            
        try:
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.warning(f"Collection '{self.collection_name}' is empty")
                return []
                
            if k > collection_count:
                logger.warning(f"Number of requested results {k} is greater than number of elements in index {collection_count}, updating n_results = {collection_count}")
                k = collection_count
            
            # a direct lookup (no semantic search)
            if not query and where_filter:
                results = self.collection.query(
                    query_texts=[""],
                    n_results=k,
                    where=where_filter,
                    include=['metadatas', 'documents', 'distances']
                )
            else:
                # semantic search
                results = self.collection.query(
                    query_texts=[query or ""],  # Handle None query
                    n_results=k,
                    where=where_filter,
                    include=['metadatas', 'documents', 'distances']
                )
        except Exception as e:
            logger.error(f"ChromaDB query failed in '{self.collection_name}': {e}")
            return []

        processed_results = []
        if results and all(key in results for key in ['ids', 'metadatas', 'distances', 'documents']) and results['ids']:
            if not results['ids'][0]:
                logger.warning(f"No results found in collection '{self.collection_name}' for query: '{query}' with filter: {where_filter}")
                return []
                
            # Assuming a single query, so we access the first list in each result key
            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            documents = results['documents'][0]
            
            for i in range(len(ids)):
                # Process metadata (convert JSON strings back)
                metadata = metadatas[i] if i < len(metadatas) else {}
                processed_metadata = {}
                
                if metadata:
                    for key, value in metadata.items():
                        try:
                            # Try to parse JSON for lists, dicts (including links)
                            if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                try:
                                    parsed_value = json.loads(value)
                                    processed_metadata[key] = parsed_value
                                except json.JSONDecodeError:
                                    processed_metadata[key] = value
                            # Convert numeric strings back to numbers
                            elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                if '.' in value:
                                    processed_metadata[key] = float(value)
                                else:
                                    processed_metadata[key] = int(value)
                            else:
                                processed_metadata[key] = value
                        except (json.JSONDecodeError, ValueError):
                            processed_metadata[key] = value
                
                document = documents[i] if i < len(documents) else ""
                distance = distances[i] if i < len(distances) else 1.0
                
                processed_results.append({
                    'id': ids[i],
                    'metadata': processed_metadata,
                    'distance': distance,
                    'document': document
                })
        else:
            logger.warning(f"Invalid or empty results from ChromaDB for query: '{query}' in collection '{self.collection_name}'")
                        
        return processed_results

    def load_from_local_memory(self, memories, model_name=None):
        """Initialize collection from memory objects dictionary.
        
        Args:
            memories: Dictionary of memory objects {id: memory_obj}
            model_name: Optional embedding model name
            
        Returns:
            self: For chaining
        """
        # Clear existing content hashes if any
        self.content_hashes = {}
        
        # Add each memory to the collection with content hash tracking
        for memory_id, memory in memories.items():
            metadata = {
                "id": memory.id,
                "content": memory.content,
                "keywords": memory.keywords,
                "links": memory.links,
                "retrieval_count": memory.retrieval_count,
                "timestamp": memory.timestamp,
                "last_accessed": memory.last_accessed,
                "context": memory.context,
                "evolution_history": memory.evolution_history,
                "category": getattr(memory, "category", "Uncategorized"),
                "tags": getattr(memory, "tags", [])
            }
            
            self.add_document(memory.content, metadata, memory_id)
            
        return self
    
    def clear(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Clear memories for a specific user/session or all if none specified"""
        where_conditions = {}
        
        if user_id is not None:
            where_conditions["user_id"] = {"$eq": user_id}
        
        if session_id is not None:
            where_conditions["session_id"] = {"$eq": session_id}
        
        try:
            # If no conditions specified, get all IDs and delete them
            if not where_conditions:
                # Get all document IDs
                results = self.collection.get(include=['documents'])
                if results and 'ids' in results and results['ids']:
                    self.collection.delete(ids=results['ids'])
                    logger.debug(f"Cleared all documents from collection '{self.collection_name}'")
                self.content_hashes.clear()
            else:
                # Delete only matching documents
                results = self.collection.get(where=where_conditions, include=['documents'])
                if results and 'ids' in results and results['ids']:
                    self.collection.delete(ids=results['ids'])
                    # Clean up content_hashes for deleted IDs
                    for doc_id in results['ids']:
                        if doc_id in self.content_hashes:
                            del self.content_hashes[doc_id]
                    logger.debug(f"Cleared documents with filter from '{self.collection_name}': {where_conditions}")
        except Exception as e:
            logger.error(f"Error clearing collection '{self.collection_name}': {e}")