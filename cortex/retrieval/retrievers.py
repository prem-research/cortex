from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import pickle
import json
import logging
import os
from cortex.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_CHROMA_URI
from cortex.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""
    def __init__(self, collection_name: str = "memories", embedding_model: str = DEFAULT_EMBEDDING_MODEL, 
                 chroma_uri: str = DEFAULT_CHROMA_URI):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.chroma_uri = chroma_uri
        

        self.embedding_manager = EmbeddingManager(embedding_model)
        
        try:

            self.client = chromadb.HttpClient(host=chroma_uri)
            logger.info(f"Connected to ChromaDB server at {chroma_uri}")
            

            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.debug(f"Initialized persistent collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB server at {chroma_uri}. "
                        f"Please ensure ChromaDB server is running. Error: {e}")
            logger.info("To start ChromaDB server locally, run: chroma run --host localhost --port 8000")
            raise RuntimeError(f"ChromaDB server connection failed: {e}")

            

        self.content_hashes = {}
        
    def _hash_content(self, document: str) -> str:
        import hashlib
        return hashlib.md5(document.encode()).hexdigest()
        
    def add_document(self, document: str, metadata: Dict, doc_id: str):
        if not document or not doc_id:
            logger.warning(f"Skipping add for empty document or missing ID")
            return
            
        content_hash = self._hash_content(document + str(metadata))
        
        existing_hash = self.content_hashes.get(doc_id)
        if existing_hash == content_hash:
            logger.debug(f"Skipping add for unchanged document: {doc_id}")
            return
        processed_metadata = {}
        # Derive a numeric timestamp for efficient range queries, store as timestamp_epoch
        if 'timestamp' in metadata and metadata.get('timestamp'):
            try:
                from datetime import datetime
                ts = str(metadata['timestamp']).replace('Z', '+00:00')
                processed_metadata['timestamp_epoch'] = datetime.fromisoformat(ts).timestamp()
            except Exception:
                pass
        processed_metadata["doc_id"] = doc_id
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (int, float, bool)):
                processed_metadata[key] = value
                continue
            if key == 'links' and isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)
        
        if doc_id in self.content_hashes:
            try:
                self.collection.delete(ids=[doc_id])
                logger.debug(f"Deleted existing document to update: {doc_id}")
            except Exception as e:
                logger.error(f"Error deleting document before update: {e}")
                
        try:
            embedding = self.embedding_manager.get_embedding(document)
            
            self.collection.add(
                documents=[document],
                metadatas=[processed_metadata],
                ids=[doc_id],
                embeddings=[embedding]
            )
            self.content_hashes[doc_id] = content_hash
            logger.debug(f"Added/updated document: {doc_id}")
        except Exception as e:
            logger.error(f"Error adding document to '{self.collection_name}': {e}")
        
    def delete_document(self, doc_id: str):
        if not doc_id:
            return
            
        try:
            self.collection.delete(ids=[doc_id])
            if doc_id in self.content_hashes:
                del self.content_hashes[doc_id]
        except Exception as e:
            logger.error(f"Error deleting document from '{self.collection_name}': {e}")
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if not doc_id:
            return None
            
        try:
            results = self.collection.get(where={"doc_id": {"$eq": doc_id}}, include=['documents', 'metadatas'])
            if results and results.get('metadatas'):
                metas = results.get('metadatas') or []
                docs = results.get('documents') or []
                if metas:
                    metadata = metas[0]
                    document = docs[0] if docs else ""
                    return {
                        'id': doc_id,
                        'metadata': metadata,
                        'document': document,
                        'distance': 0.0
                    }
            # fallback to direct ID lookup
            try:
                results = self.collection.get(ids=[doc_id], include=['documents', 'metadatas'])
                if results and results.get('ids'):
                    ids = results.get('ids') or []
                    docs = results.get('documents') or []
                    metas = results.get('metadatas') or []
                    if ids:
                        metadata = metas[0] if metas else {}
                        document = docs[0] if docs else ""
                        return {
                            'id': doc_id,
                            'metadata': metadata,
                            'document': document,
                            'distance': 0.0
                        }
            except Exception as inner:
                logger.debug(f"Chroma get(ids=...) failed for {doc_id} in '{self.collection_name}': {inner}")
                
        except Exception as e:
            logger.debug(f"Error getting document {doc_id} from '{self.collection_name}': {e}")
            
        return None
    
    def save(self, file_path, embeddings_path=None):
        data = {
            'content_hashes': self.content_hashes
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, file_path, embeddings_path=None):
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
            
            if not query and where_filter:
                results = self.collection.get(
                    where=where_filter,
                    include=['metadatas', 'documents']
                )
                if results and results.get('ids'):
                    ids = results.get('ids', [])[:k]
                    metadatas = results.get('metadatas', [])[:k] if results.get('metadatas') else []
                    documents = results.get('documents', [])[:k] if results.get('documents') else []
                    results = {
                        'ids': [ids],
                        'metadatas': [metadatas],
                        'documents': [documents],
                        'distances': [[0.0] * len(ids)]
                    }
                else:
                    results = {'ids': [[]], 'metadatas': [[]], 'documents': [[]], 'distances': [[]]}
            else:
                query_embedding = self.embedding_manager.get_embedding(query or "")
                results = self.collection.query(
                    query_embeddings=[query_embedding],
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
                
            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            documents = results['documents'][0]
            
            for i in range(len(ids)):
                metadata = metadatas[i] if i < len(metadatas) else {}
                processed_metadata = {}
                
                if metadata:
                    for key, value in metadata.items():
                        try:
                            if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                try:
                                    parsed_value = json.loads(value)
                                    processed_metadata[key] = parsed_value
                                except json.JSONDecodeError:
                                    processed_metadata[key] = value
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
        self.content_hashes = {}
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
        where_conditions = {}
        
        if user_id is not None:
            where_conditions["user_id"] = {"$eq": user_id}
        
        if session_id is not None:
            where_conditions["session_id"] = {"$eq": session_id}
        
        try:
            if not where_conditions:
                results = self.collection.get(include=['documents'])
                if results and 'ids' in results and results['ids']:
                    self.collection.delete(ids=results['ids'])
                    logger.debug(f"Cleared all documents from collection '{self.collection_name}'")
                self.content_hashes.clear()
            else:
                results = self.collection.get(where=where_conditions, include=['documents'])
                if results and 'ids' in results and results['ids']:
                    self.collection.delete(ids=results['ids'])
                    for doc_id in results['ids']:
                        if doc_id in self.content_hashes:
                            del self.content_hashes[doc_id]
                    logger.debug(f"Cleared documents with filter from '{self.collection_name}': {where_conditions}")
        except Exception as e:
            logger.error(f"Error clearing collection '{self.collection_name}': {e}")