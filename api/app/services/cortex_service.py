import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from cortex.memory_system import AgenticMemorySystem

class CortexService:
    _instance = None
    _memory_system = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CortexService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._memory_system is None:
            self._initialize_cortex()
    
    def _initialize_cortex(self):
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        chroma_uri = os.getenv("CHROMA_URI", "http://localhost:7003")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        try:
            self._memory_system = AgenticMemorySystem(
                api_key=api_key,
                chroma_uri=chroma_uri,
                enable_smart_collections=True,
                enable_background_processing=True
            )
            print(f"Cortex memory system initialized successfully with ChromaDB at {chroma_uri}")
        except Exception as e:
            print(f"Failed to initialize Cortex: {e}")
            raise
    
    async def add_memory_note(
        self, 
        content: str, 
        user_id: str,
        session_id: Optional[str] = '', 
        time: Optional[str] = None, 
        context: Optional[str] = None, 
        tags: Optional[List[str]] = None,
        **metadata
    ) -> str:
        """Add a memory note to Cortex"""
        try:
            memory_id = self._memory_system.add_note(
                content=content,
                user_id=user_id,
                session_id=session_id,
                time=time,
                context=context,
                tags=tags,
                **metadata
            )
            return memory_id
        except Exception as e:
            print(f"Error adding memory note: {e}")
            raise
    
    async def search_memories(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = '',
        memory_source: str = "ltm",
        temporal_weight: float = 0.0,
        date_range: Optional[str] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search memories in Cortex"""
        try:
            results = self._memory_system.search(
                query=query,
                user_id=user_id,
                session_id=session_id,
                memory_source=memory_source,
                temporal_weight=temporal_weight,
                date_range=date_range,
                where_filter=where_filter,
                limit=limit
            )
            return results
        except Exception as e:
            print(f"Error searching memories: {e}")
            raise
    
    async def get_memory(self, memory_id: str, user_id: str, session_id: Optional[str] = '') -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID"""
        try:
            memory = self._memory_system.ltm.get(memory_id, user_id, session_id)
            return memory
        except Exception as e:
            print(f"Error getting memory {memory_id}: {e}")
            return None
    
    async def get_linked_memories(
        self, 
        memory_ids: List[str], 
        user_id: str, 
        session_id: Optional[str] = '',
        limit: int = 4
    ) -> List[Dict[str, Any]]:
        """Get linked memories from a list of memory IDs"""
        try:
            all_linked_memories = []
            
            for memory_id in memory_ids:
                if len(all_linked_memories) >= limit:
                    break
                    
                # Get the memory first
                memory = await self.get_memory(memory_id, user_id, session_id)
                if not memory:
                    continue
                
                # Extract links from the memory
                links = memory.get('links', [])
                if not links or not isinstance(links, list):
                    continue
                
                # Get linked memories
                for link_id in links:
                    if len(all_linked_memories) >= limit:
                        break
                    linked_memory = await self.get_memory(link_id, user_id, session_id)
                    if linked_memory:
                        all_linked_memories.append(linked_memory)
            
            return all_linked_memories
        except Exception as e:
            print(f"Error getting linked memories: {e}")
            return []
    
    async def get_memories_with_linked(
        self,
        memory_ids: List[str],
        user_id: str,
        session_id: Optional[str] = '',
        linked_limit: int = 4
    ) -> Dict[str, Any]:
        """Get memories and their linked memories together"""
        try:
            memories = []
            for memory_id in memory_ids:
                memory = await self.get_memory(memory_id, user_id, session_id)
                if memory:
                    memories.append(memory)
            
            # Get linked memories from top 2 memories (like in the reference)
            top_memories = memories
            linked_memories = await self.get_linked_memories(
                [mem.get('id') for mem in top_memories if mem.get('id')],
                user_id,
                session_id,
                linked_limit
            )
            
            return {
                "memories": memories,
                "linked_memories": linked_memories
            }
        except Exception as e:
            print(f"Error getting memories with linked: {e}")
            return {"memories": [], "linked_memories": []}
    
    async def clear_memories(self, user_id: str, session_id: Optional[str] = '') -> bool:
        """Clear memories from Cortex"""
        try:
            self._memory_system.clear_ltm(user_id=user_id, session_id=session_id)
            if session_id:
                print(f"[CORTEX] Cleared chat conversations for user {user_id}")
            else:
                print(f"[CORTEX] Cleared all storage for user {user_id}")
            return True
        except Exception as e:
            print(f"Error clearing memories: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if Cortex is healthy by performing a simple operation"""
        try:
            # Try a simple search operation
            await self.search_memories(
                query="health_check",
                user_id="health_check_user",
                limit=1
            )
            return True
        except Exception as e:
            print(f"Cortex health check failed: {e}")
            return False

# Global instance
cortex_service = CortexService()