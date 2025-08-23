"""Cortex service layer for managing memory operations"""

import sys
import os
from typing import List, Dict, Any, Optional
import structlog

# Add parent directory to path to import cortex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cortex.memory_system import AgenticMemorySystem
from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class CortexService:
    """Service layer for Cortex memory operations"""
    
    def __init__(self):
        """Initialize the Cortex memory system"""
        self.memory_system = AgenticMemorySystem(
            model_name=settings.cortex_model_name,
            llm_backend=settings.cortex_llm_backend,
            llm_model=settings.cortex_llm_model,
            stm_capacity=settings.cortex_stm_capacity,
            api_key=settings.openai_api_key,
            enable_smart_collections=settings.cortex_enable_smart_collections,
            enable_background_processing=settings.cortex_enable_background_processing,
            chroma_uri=settings.chroma_uri
        )
        logger.info("Cortex memory system initialized", 
                   model=settings.cortex_model_name,
                   backend=settings.cortex_llm_backend)
    
    def store_memory(self, content: str, context: Optional[str] = None,
                    tags: Optional[List[str]] = None, timestamp: Optional[str] = None,
                    user_id: Optional[str] = None, session_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a new memory"""
        kwargs = {}
        if context:
            kwargs["context"] = context
        if tags:
            kwargs["tags"] = tags
        if metadata:
            kwargs.update(metadata)
        
        memory_id = self.memory_system.add_note(
            content=content,
            time=timestamp,
            user_id=user_id,
            session_id=session_id,
            **kwargs
        )
        
        logger.info("Memory stored", 
                   memory_id=memory_id, 
                   user_id=user_id, 
                   session_id=session_id)
        return memory_id
    
    def search_memories(self, query: str, limit: int = 10,
                       memory_source: str = "all", context: Optional[str] = None,
                       tags: Optional[List[str]] = None, exclude_content: bool = False,
                       include_links: bool = True, apply_postprocessing: bool = True,
                       user_id: Optional[str] = None, session_id: Optional[str] = None,
                       temporal_weight: Optional[float] = None,
                       date_range: Optional[str] = None) -> Dict[str, Any]:
        """Search for memories"""
        
        # Build where filter for tags
        where_filter = {}
        if tags:
            where_filter["tags"] = {"$contains": tags}
        
        # Parse date range if provided
        parsed_date_range = None
        if date_range:
            parsed_date_range = self.memory_system._parse_date_range(date_range)
        
        # Auto-detect temporal queries if temporal_weight not specified
        if temporal_weight is None:
            temporal_keywords = ["last", "recent", "latest", "yesterday", "today", "this week", "past", "ago"]
            if any(keyword in query.lower() for keyword in temporal_keywords) or date_range:
                temporal_weight = 0.7
            else:
                temporal_weight = 0.0
        
        # Search memories
        results = self.memory_system.search(
            query=query,
            limit=limit,
            memory_source=memory_source,
            where_filter=where_filter if where_filter else None,
            apply_postprocessing=apply_postprocessing,
            context=context,
            user_id=user_id,
            session_id=session_id,
            temporal_weight=temporal_weight,
            date_range=parsed_date_range
        )
        
        # Format results
        memories = []
        seen_ids = set()
        
        for result in results:
            memory_id = result.get("id", "")
            if not memory_id or memory_id in seen_ids:
                continue
            
            seen_ids.add(memory_id)
            
            memory_output = {
                "id": memory_id,
                "context": result.get("context", ""),
                "tags": result.get("tags", []),
                "keywords": result.get("keywords", []),
                "timestamp": str(result.get("timestamp", "")),
                "score": result.get("score"),
                "is_linked": False,
                "memory_tier": result.get("memory_tier", "unknown"),
                "collection_name": result.get("collection_name", ""),
                "category": result.get("category", ""),
                "composite_score": result.get("composite_score")
            }
            
            if not exclude_content:
                memory_output["content"] = result.get("content", "")
            
            memories.append(memory_output)
        
        # Process linked memories if requested
        if include_links:
            linked_memories = self._process_linked_memories(
                results, seen_ids, exclude_content, user_id, session_id
            )
            memories.extend(linked_memories)
        
        logger.info("Memory search completed", 
                   query=query, 
                   count=len(memories),
                   user_id=user_id)
        
        return {
            "memories": memories,
            "count": len(memories)
        }
    
    def _process_linked_memories(self, results: List[Dict[str, Any]], 
                                seen_ids: set, exclude_content: bool,
                                user_id: Optional[str] = None,
                                session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process linked memories from search results"""
        linked_memories = []
        
        for result in results:
            memory_id = result.get("id", "")
            if not memory_id:
                continue
            
            links = result.get("links", {})
            
            # Handle both dict and list formats for links
            link_ids = []
            if isinstance(links, dict):
                link_ids = list(links.keys())
            elif isinstance(links, list):
                link_ids = links
            
            # Process each linked memory
            for link_id in link_ids:
                if link_id in seen_ids:
                    continue
                
                seen_ids.add(link_id)
                
                linked_memory = self.memory_system.read(link_id, user_id, session_id)
                if not linked_memory:
                    continue
                
                link_output = {
                    "id": link_id,
                    "context": getattr(linked_memory, "context", ""),
                    "tags": getattr(linked_memory, "tags", []),
                    "keywords": getattr(linked_memory, "keywords", []),
                    "timestamp": str(getattr(linked_memory, "timestamp", "")),
                    "score": None,
                    "is_linked": True,
                    "memory_tier": "ltm",
                    "category": getattr(linked_memory, "category", "")
                }
                
                if not exclude_content:
                    link_output["content"] = getattr(linked_memory, "content", "")
                
                # Get relationship metadata if available
                if isinstance(links, dict) and link_id in links:
                    link_data = links[link_id]
                    if isinstance(link_data, dict):
                        link_output["relationship_type"] = link_data.get("type")
                        link_output["relationship_strength"] = link_data.get("strength")
                        link_output["relationship_reason"] = link_data.get("reason")
                
                linked_memories.append(link_output)
        
        return linked_memories
    
    def get_memory(self, memory_id: str, user_id: Optional[str] = None,
                  session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a memory by ID"""
        memory = self.memory_system.read(memory_id, user_id, session_id)
        
        if memory:
            return {
                "id": memory_id,
                "content": getattr(memory, "content", ""),
                "context": getattr(memory, "context", ""),
                "tags": getattr(memory, "tags", []),
                "keywords": getattr(memory, "keywords", []),
                "timestamp": str(getattr(memory, "timestamp", "")),
                "memory_tier": "ltm",
                "category": getattr(memory, "category", "")
            }
        return None
    
    def update_memory(self, memory_id: str, content: Optional[str] = None,
                     context: Optional[str] = None, tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a memory"""
        kwargs = {}
        if content:
            kwargs["content"] = content
        if context:
            kwargs["context"] = context
        if tags:
            kwargs["tags"] = tags
        if metadata:
            kwargs.update(metadata)
        
        success = self.memory_system.update(memory_id, **kwargs)
        
        if success:
            logger.info("Memory updated", memory_id=memory_id)
        else:
            logger.warning("Memory not found for update", memory_id=memory_id)
        
        return success
    
    def delete_memory(self, memory_id: str, user_id: Optional[str] = None,
                     session_id: Optional[str] = None) -> bool:
        """Delete a memory"""
        success = self.memory_system.delete(memory_id)
        
        if success:
            logger.info("Memory deleted", memory_id=memory_id)
        else:
            logger.warning("Memory not found for deletion", memory_id=memory_id)
        
        return success
    
    def clear_memories(self, memory_source: str = "all",
                      user_id: Optional[str] = None,
                      session_id: Optional[str] = None):
        """Clear memories from STM, LTM, or both"""
        if memory_source in ["stm", "all"]:
            self.memory_system.clear_stm(user_id, session_id)
            logger.info("STM cleared", user_id=user_id, session_id=session_id)
        
        if memory_source in ["ltm", "all"]:
            self.memory_system.clear_ltm(user_id, session_id)
            logger.info("LTM cleared", user_id=user_id, session_id=session_id)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "total_memories": len(self.memory_system.memories),
            "stm_capacity": settings.cortex_stm_capacity,
            "smart_collections_enabled": settings.cortex_enable_smart_collections,
            "background_processing_enabled": settings.cortex_enable_background_processing
        }
        
        if hasattr(self.memory_system, 'collection_manager') and self.memory_system.collection_manager:
            stats["collections"] = {
                "total": len(self.memory_system.collection_manager.collections),
                "category_patterns": len(self.memory_system.collection_manager.category_counts)
            }
        
        return stats