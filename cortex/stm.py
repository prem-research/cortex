from typing import List, Dict, Optional, Any, Tuple
from collections import OrderedDict
import threading
import numpy as np
from cortex.embedding_manager import EmbeddingManager
from cortex.constants import DEFAULT_STM_CAPACITY, DEFAULT_EMBEDDING_MODEL

class MemoryTier:
    
    def __init__(self, name: str):
        self.name = name
        
    def add(self, memory_id: str, content: str, metadata: Dict[str, Any], user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        
        raise NotImplementedError("Subclasses must implement add()")
        
    def search(self, query: str, limit: int, where_filter: Optional[Dict] = None) -> List[Dict]:
        
        raise NotImplementedError("Subclasses must implement search()")
        
    def get(self, memory_id: str) -> Optional[Dict]:
        
        raise NotImplementedError("Subclasses must implement get()")
        
    def delete(self, memory_id: str) -> bool:
        
        raise NotImplementedError("Subclasses must implement delete()")


class ShortTermMemory(MemoryTier):
    
    
    def __init__(self, capacity: int = DEFAULT_STM_CAPACITY, model_name: str = DEFAULT_EMBEDDING_MODEL):
        super().__init__("short_term_memory")
        self.capacity = capacity
        self.user_memories: Dict[Tuple[Optional[str], Optional[str]], OrderedDict] = {}
        self.user_embeddings: Dict[Tuple[Optional[str], Optional[str]], Dict[str, List[float]]] = {}
        self.lock = threading.RLock()
        self.embedding_manager = EmbeddingManager(model_name)
        
    def _get_user_key(self, user_id: Optional[str], session_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        
        return (user_id, session_id)
        
    def _get_user_memory_store(self, user_id: Optional[str], session_id: Optional[str]) -> OrderedDict:
        
        key = self._get_user_key(user_id, session_id)
        with self.lock:
            if key not in self.user_memories:
                self.user_memories[key] = OrderedDict()
                self.user_embeddings[key] = {}
            return self.user_memories[key]
    
    def add(self, memory_id: str, content: str, metadata: Dict[str, Any], user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        
        store = self._get_user_memory_store(user_id, session_id)
        embedding_store = self.user_embeddings[self._get_user_key(user_id, session_id)]
        
        with self.lock:
            if len(store) >= self.capacity:
                oldest_id = next(iter(store))
                del store[oldest_id]
                if oldest_id in embedding_store:
                    del embedding_store[oldest_id]
            
            memory_data = {
                "id": memory_id,
                "content": content,
                **metadata
            }
            store[memory_id] = memory_data
            
            if "embedding" in metadata:
                embedding_store[memory_id] = metadata["embedding"]
            
            return memory_id
    
    def get(self, memory_id: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Optional[Dict]:
        
        store = self._get_user_memory_store(user_id, session_id)
        with self.lock:
            memory = store.get(memory_id)
            if memory and memory_id in store:
                store.move_to_end(memory_id)
            return memory
    
    def delete(self, memory_id: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> bool:
        
        store = self._get_user_memory_store(user_id, session_id)
        embedding_store = self.user_embeddings[self._get_user_key(user_id, session_id)]
        
        with self.lock:
            if memory_id in store:
                del store[memory_id]
                if memory_id in embedding_store:
                    del embedding_store[memory_id]
                return True
            return False
    
    def search(self, query: str, limit: int, where_filter: Optional[Dict] = None, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict]:
        """ memories using approximate cosine similarity"""
        store = self._get_user_memory_store(user_id, session_id)
        embedding_store = self.user_embeddings[self._get_user_key(user_id, session_id)]
        
        if not store:
            return []
        
        query_embedding = self.embedding_manager.get_embedding(query)
        
        with self.lock:
            results = []
            
            for memory_id, memory in store.items():
                if memory_id in embedding_store:
                    memory_embedding = embedding_store[memory_id]
                else:
                    memory_embedding = self.embedding_manager.get_embedding(memory["content"])
                    embedding_store[memory_id] = memory_embedding
                
                similarity = np.dot(query_embedding, memory_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
                )
                
                if where_filter:
                    match = True
                    for field, condition in where_filter.items():
                        if field in memory:
                            if isinstance(condition, dict):
                                for op, value in condition.items():
                                    if op == "$eq" and memory[field] != value:
                                        match = False
                                    elif op == "$contains" and isinstance(memory[field], list) and value not in memory[field]:
                                        match = False
                            else:
                                if memory[field] != condition:
                                    match = False
                    if not match:
                        continue
                
                results.append({
                    **memory,
                    "score": float(similarity),
                    "distance": 1.0 - float(similarity)
                })
            
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
    
    def clear(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """ clear memories for a specific user/session or all if none specified"""
        if user_id is None and session_id is None:
            with self.lock:
                self.user_memories.clear()
                self.user_embeddings.clear()
        else:
            key = self._get_user_key(user_id, session_id)
            with self.lock:
                if key in self.user_memories:
                    del self.user_memories[key]
                if key in self.user_embeddings:
                    del self.user_embeddings[key] 