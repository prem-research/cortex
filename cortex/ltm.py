from typing import List, Dict, Optional, Any
import threading
import json
import logging
from datetime import datetime
from cortex.stm import MemoryTier
from cortex.retrieval.retrievers import ChromaRetriever
from cortex.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_CHROMA_URI

logger = logging.getLogger(__name__)

class LongTermMemory(MemoryTier):
    
    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL, chroma_uri: str = DEFAULT_CHROMA_URI):
        super().__init__("long_term_memory")
        self.embedding_model = embedding_model
        self.chroma_uri = chroma_uri
        self.collections: Dict[str, ChromaRetriever] = {}
        self.lock = threading.RLock()
    
    def _get_collection_name(self, user_id: Optional[str], session_id: Optional[str]) -> str:
        base = "memories"
        if user_id and session_id:
            return f"{base}_{user_id}_{session_id}"
        elif user_id:
            return f"{base}_{user_id}"
        elif session_id:
            return f"{base}_{session_id}"
        return base
    
    def _get_collection(self, user_id: Optional[str], session_id: Optional[str]):
        """get or create a collection for a user/session"""
        
        collection_name = self._get_collection_name(user_id, session_id)
        
        with self.lock:
            if collection_name not in self.collections:
                retriever = ChromaRetriever(
                    collection_name=collection_name, 
                    embedding_model=self.embedding_model,
                    chroma_uri=self.chroma_uri
                )
                self.collections[collection_name] = retriever
            
            return self.collections[collection_name]
    
    def add(self, memory_id: str, content: str, metadata: Dict[str, Any], user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """ add memory to long-term storage"""
        collection = self._get_collection(user_id, session_id)
        
        if user_id is not None and "user_id" not in metadata:
            metadata["user_id"] = user_id
        if session_id is not None and "session_id" not in metadata:
            metadata["session_id"] = session_id
            
        collection.add_document(document=content, metadata=metadata, doc_id=memory_id)
        return memory_id
    
    def get(self, memory_id: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Optional[Dict]:
        """ get a memory by ID using direct retrieval (not search)"""
        collection = self._get_collection(user_id, session_id)
        
        try:
            result = collection.get_document(memory_id)
            
            if result:
                return {
                    "id": memory_id,
                    "content": result.get("document", ""),
                    **{k: v for k, v in result.get("metadata", {}).items() if k != "id"}
                }
            
            if user_id is not None or session_id is not None:
                default_collection = self._get_collection(None, None)
                if default_collection != collection:
                    result = default_collection.get_document(memory_id)
                    if result:
                        return {
                            "id": memory_id,
                            "content": result.get("document", ""),
                            **{k: v for k, v in result.get("metadata", {}).items() if k != "id"}
                        }
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id} from LTM: {e}")
        
        return None
    
    def search(self, query: str, limit: int, where_filter: Optional[Dict] = None, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict]:
        """ searchmemories in long-term storage"""
        collection = self._get_collection(user_id, session_id)
        
        try:
            where_conditions = []
            
            if where_filter:
                where_conditions.append(where_filter)
            
            if user_id is not None:
                where_conditions.append({"user_id": {"$eq": user_id}})
            
            if session_id is not None:
                where_conditions.append({"session_id": {"$eq": session_id}})
            
            chroma_where = {"$and": where_conditions} if len(where_conditions) > 1 else (where_conditions[0] if where_conditions else None)
            
            results = collection.search(query, k=limit, where_filter=chroma_where)
            
            if user_id is not None and session_id is None and (not results or len(results) == 0):
                logger.info(f"No results found in default collection for user {user_id}, searching across all collections")
                all_results = []
                
                user_collections = [name for name in self.collections.keys() if user_id in name]
                
                for coll_name in user_collections:
                    try:
                        coll = self.collections[coll_name]
                        combined_filter = {"user_id": {"$eq": user_id}}
                        if where_filter:
                            if isinstance(where_filter, dict) and "$and" in where_filter and isinstance(where_filter["$and"], list):
                                combined_filter = {"$and": where_filter["$and"] + [{"user_id": {"$eq": user_id}}]}
                            else:
                                combined_filter = {"$and": [where_filter, {"user_id": {"$eq": user_id}}]}

                        user_results = coll.search(
                            query,
                            k=limit,
                            where_filter=combined_filter,
                        )
                        all_results.extend(user_results)
                    except Exception as e:
                        logger.error(f"Error searching collection {coll_name}: {e}")
                
                all_results.sort(key=lambda x: x.get("distance", 1.0))
                results = all_results[:limit]
            
            formatted_results = []
            for result in results:
                # Convert cosine distance to similarity score
                distance = result.get("distance") or 0
                # With cosine distance: distance = 1 - cosine_similarity
                # So: cosine_similarity = 1 - distance
                # Clamp to [0, 1] range for safety
                similarity_score = max(0.0, min(1.0, 1.0 - distance))
                
                formatted_result = {
                    "id": result.get("id", ""),
                    "content": result.get("document", ""),
                    "distance": distance,
                    "score": similarity_score,
                }
                
                metadata = result.get("metadata", {})
                for key, value in metadata.items():
                    # Handle links specially - ensure they're converted from string to dict
                    if key == "links" and isinstance(value, str):
                        try:
                            formatted_result[key] = json.loads(value)
                        except json.JSONDecodeError:
                            formatted_result[key] = {}
                    else:
                        formatted_result[key] = value
                
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in LTM: {e}")
            return []
    
    def delete(self, memory_id: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> bool:
        """ delete a memory from long-term storage"""
        collection = self._get_collection(user_id, session_id)
        
        try:
            result = self.get(memory_id, user_id, session_id)
            if result:
                collection.delete_document(memory_id)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting from LTM: {e}")
            return False
    
    def clear(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """ clear memories for a specific user/session or all if none specified"""
        try:
            collection_name = self._get_collection_name(user_id, session_id)
            with self.lock:
                if collection_name in self.collections:
                    self.collections[collection_name].clear()
        except Exception as e:
            logger.error(f"Error clearing collection: {e}") 