from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from cortex.llm_controllers.llm_controller import LLMController
from cortex.retrieval.retrievers import ChromaRetriever
import json
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)

class MemoryNote:
    """A memory note that represents a single unit of information in the memory system.
    
    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Relationship data (links to other memories)
    - Usage statistics (retrieval count)
    - Evolution tracking (history of changes)
    """
    
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 user_id: Optional[str] = None,
                 session_id: Optional[str] = None):
        """Initialize a new memory note with its associated metadata."""
        # Core content and ID
        self.content = content
        self.id = id or str(uuid.uuid4())
        
        # Semantic metadata
        self.keywords = keywords or []
        self.links = links or {}
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        # Temporal information
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Usage and evolution data
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []

        # User and session data
        self.user_id = user_id
        self.session_id = session_id

class MemoryTier:
    """Base class for memory tiers (STM/LTM)"""
    def __init__(self, name: str):
        self.name = name
        
    def add(self, memory_id: str, content: str, metadata: Dict[str, Any], user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Add a memory to this tier"""
        raise NotImplementedError("Subclasses must implement add()")
        
    def search(self, query: str, limit: int, where_filter: Optional[Dict] = None) -> List[Dict]:
        """Search memories in this tier"""
        raise NotImplementedError("Subclasses must implement search()")
        
    def get(self, memory_id: str) -> Optional[Dict]:
        """Get a specific memory by ID"""
        raise NotImplementedError("Subclasses must implement get()")
        
    def delete(self, memory_id: str) -> bool:
        """Delete a memory from this tier"""
        raise NotImplementedError("Subclasses must implement delete()")


class ShortTermMemory(MemoryTier):
    """In-memory short-term memory store with limited capacity"""
    
    def __init__(self, capacity: int = 100, embedding_dimension: int = 384):
        super().__init__("short_term_memory")
        self.capacity = capacity
        self.embedding_dimension = embedding_dimension
        self.user_memories: Dict[Tuple[Optional[str], Optional[str]], OrderedDict] = {}
        self.user_embeddings: Dict[Tuple[Optional[str], Optional[str]], Dict[str, List[float]]] = {}
        self.lock = threading.RLock()
        
    def _get_user_key(self, user_id: Optional[str], session_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Create a key tuple from user_id and session_id"""
        return (user_id, session_id)
        
    def _get_user_memory_store(self, user_id: Optional[str], session_id: Optional[str]) -> OrderedDict:
        """Get or create a memory store for a specific user/session"""
        key = self._get_user_key(user_id, session_id)
        with self.lock:
            if key not in self.user_memories:
                self.user_memories[key] = OrderedDict()
                self.user_embeddings[key] = {}
            return self.user_memories[key]
    
    def add(self, memory_id: str, content: str, metadata: Dict[str, Any], user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Add a memory to short-term memory"""
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
        """Get a memory by ID"""
        store = self._get_user_memory_store(user_id, session_id)
        with self.lock:
            memory = store.get(memory_id)
            if memory and memory_id in store:
                store.move_to_end(memory_id)
            return memory
    
    def delete(self, memory_id: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> bool:
        """Delete a memory"""
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
        """Search memories using approximate cosine similarity"""
        store = self._get_user_memory_store(user_id, session_id)
        embedding_store = self.user_embeddings[self._get_user_key(user_id, session_id)]
        
        if not store:
            return []
        
        if hasattr(self, '_model'):
            model = self._model
        else:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            self._model = model
        
        query_embedding = model.encode(query)
        
        with self.lock:
            results = []
            
            for memory_id, memory in store.items():
                if memory_id in embedding_store:
                    memory_embedding = embedding_store[memory_id]
                else:
                    memory_embedding = model.encode(memory["content"])
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
        """Clear memories for a specific user/session or all if none specified"""
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


class LongTermMemory(MemoryTier):
    """Persistent long-term memory using ChromaDB with enhanced metadata"""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        super().__init__("long_term_memory")
        self.embedding_model = embedding_model
        self.collections: Dict[str, ChromaRetriever] = {}
        self.lock = threading.RLock()
    
    def _get_collection_name(self, user_id: Optional[str], session_id: Optional[str]) -> str:
        """Get a unique collection name for a user/session combination"""
        base = "memories"
        if user_id and session_id:
            return f"{base}_{user_id}_{session_id}"
        elif user_id:
            return f"{base}_{user_id}"
        elif session_id:
            return f"{base}_{session_id}"
        return base
    
    def _get_collection(self, user_id: Optional[str], session_id: Optional[str]):
        """Get or create a ChromaDB collection for a user/session"""
        
        collection_name = self._get_collection_name(user_id, session_id)
        
        with self.lock:
            if collection_name not in self.collections:
                retriever = ChromaRetriever(collection_name=collection_name)
                self.collections[collection_name] = retriever
            
            return self.collections[collection_name]
    
    def add(self, memory_id: str, content: str, metadata: Dict[str, Any], user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Add a memory to long-term storage"""
        collection = self._get_collection(user_id, session_id)
        
        # Ensure user_id and session_id are in metadata
        if user_id is not None and "user_id" not in metadata:
            metadata["user_id"] = user_id
        if session_id is not None and "session_id" not in metadata:
            metadata["session_id"] = session_id
            
        collection.add_document(document=content, metadata=metadata, doc_id=memory_id)
        return memory_id
    
    def get(self, memory_id: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Optional[Dict]:
        """Get a memory by ID"""
        collection = self._get_collection(user_id, session_id)
        
        try:
            # First try to get directly by ID - most efficient approach
            results = collection.search(query="", k=1, where_filter={"id": {"$eq": memory_id}})
            
            if not results or len(results) == 0:
                # If not found, try other collections if user_id/session_id were specified
                if user_id is not None or session_id is not None:
                    # Try default collection as fallback
                    default_collection = self._get_collection(None, None)
                    if default_collection != collection:
                        results = default_collection.search(query="", k=1, where_filter={"id": {"$eq": memory_id}})
            
            if results and len(results) > 0:
                result = results[0]
                return {
                    "id": memory_id,
                    "content": result.get("document", ""),
                    **{k: v for k, v in result.get("metadata", {}).items() if k != "id"}
                }
        except Exception as e:
            logger.error(f"Error retrieving memory from LTM: {e}")
        
        return None
    
    def search(self, query: str, limit: int, where_filter: Optional[Dict] = None, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict]:
        """Search memories in long-term storage"""
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
            
            # If no results and we have a user_id but no session_id, try searching across all collections for this user
            if user_id is not None and session_id is None and (not results or len(results) == 0):
                logger.info(f"No results found in default collection for user {user_id}, searching across all collections")
                all_results = []
                
                # Get all collection names that could contain this user's data
                user_collections = [name for name in self.collections.keys() if user_id in name]
                
                for coll_name in user_collections:
                    try:
                        coll = self.collections[coll_name]
                        user_results = coll.search(
                            query, 
                            k=limit, 
                            where_filter={"user_id": {"$eq": user_id}}
                        )
                        all_results.extend(user_results)
                    except Exception as e:
                        logger.error(f"Error searching collection {coll_name}: {e}")
                
                # Sort by relevance (distance)
                all_results.sort(key=lambda x: x.get("distance", 1.0))
                results = all_results[:limit]
            
            formatted_results = []
            for result in results:
                formatted_result = {
                    "id": result.get("id", ""),
                    "content": result.get("document", ""),
                    "distance": result.get("distance"),
                    "score": 1.0 - (result.get("distance") or 0),
                }
                
                # Process metadata separately to handle special fields
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
        """Delete a memory from long-term storage"""
        collection = self._get_collection(user_id, session_id)
        
        try:
            # Try to find the memory first to make sure we're deleting from the right collection
            result = self.get(memory_id, user_id, session_id)
            if result:
                collection.delete_document(memory_id)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting from LTM: {e}")
            return False
    
    def clear(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Clear memories for a specific user/session or all if none specified"""
        try:
            collection_name = self._get_collection_name(user_id, session_id)
            with self.lock:
                if collection_name in self.collections:
                    self.collections[collection_name].clear()
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

class LightPreprocessor:
    """Lightweight preprocessing for STM"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
    
    def process(self, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Quickly extract basic metadata for STM"""
        result = metadata.copy() if metadata else {}
        
        embedding = self.memory_system._get_embedding(content)
        result["embedding"] = embedding
        
        if "keywords" not in result:
            words = content.lower().split()
            common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
            keywords = [w for w in words if w not in common_words and len(w) > 3]
            result["keywords"] = keywords[:5] if keywords else []
        
        if "context" not in result:
            result["context"] = "General"
        
        return result


class DeepPreprocessor:
    """Deep preprocessing for LTM"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.llm_controller = memory_system.llm_controller
    
    def process(self, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Thoroughly extract metadata for LTM storage using LLM for deeper understanding"""
        result = metadata.copy() if metadata else {}
        
        if "keywords" in result and len(result["keywords"]) > 0 and result.get("context") not in ["General", None]:
            return result
        
        try:
            llm_metadata = self.memory_system.analyze_content(content)
            if llm_metadata:
                result["keywords"] = llm_metadata.get("keywords", result.get("keywords", []))
                result["context"] = llm_metadata.get("context", result.get("context", "General"))
                result["tags"] = llm_metadata.get("tags", result.get("tags", []))
        except Exception as e:
            logger.warning(f"Error in deep preprocessing: {e}")
        
        return result


class RetrievalProcessor:
    """Post-processes retrieved memories from LTM"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
    
    def process(self, results: List[Dict], context: Optional[str] = None) -> List[Dict]:
        """Process and enhance retrieved results"""
        if not results:
            return []
        
        if not context:
            return results
        
        if context:
            context_embedding = self.memory_system._get_embedding(context)
            
            for result in results:
                content = result.get("content", "")
                content_embedding = self.memory_system._get_embedding(content)
                
                import numpy as np
                context_similarity = np.dot(context_embedding, content_embedding) / (
                    np.linalg.norm(context_embedding) * np.linalg.norm(content_embedding)
                )
                
                original_score = result.get("score", 0.5)
                result["context_score"] = float(context_similarity)
                result["score"] = 0.3 * original_score + 0.7 * context_similarity
        
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return results

class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution."""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 stm_capacity: int = 100,
                 api_key: Optional[str] = None):  
        """Initialize the memory system."""
        self.memories = {}
        
        self.llm_controller = LLMController(llm_backend, llm_model, api_key)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        
        self.stm = ShortTermMemory(capacity=stm_capacity)
        self.ltm = LongTermMemory(embedding_model=model_name)
        
        self.light_processor = LightPreprocessor(self)
        self.deep_processor = DeepPreprocessor(self)
        self.retrieval_processor = RetrievalProcessor(self)
        
        self._model = None
        self._model_name = model_name
        
        # * TODO: Temporal continuity (sequential memories of same event) - to be implemented later
        self._evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Your goal is to build a rich, interconnected network of memories.
                                Analyze the new memory note (content, context, keywords) and its relationship with the nearest neighbor memories provided.

                                New memory note:
                                Content: {content}
                                Context: {context}
                                Keywords: {keywords}

                                Nearest neighbors memories (up to {neighbor_number}):
                                {nearest_neighbors_memories}

                                Based on this analysis, make decisions about memory evolution:

                                1. Determine if evolution is beneficial (`should_evolve`). Evolution is beneficial if:
                                   - Meaningful connections can be established between the new note and neighbors (`strengthen` action is possible).
                                   OR
                                   - The context or tags of neighboring memories can be meaningfully updated or refined based on the new note (`update_neighbor` action is possible).
                                   OR
                                   - The new memory and an existing memory contain highly similar or complementary information that should be merged (`merge_memories` action is possible).
                                   Set `should_evolve` to `True` if any of these conditions is met. Otherwise, set it to `False`.
                                   ::IMPORTANT:: Set `should_evolve` to `True` only if a memory should really be evolved. Do not set it to true most of the time unless there's a very good and extremely confident reason. ::IMPORTANT::

                                2. If `should_evolve` is `True`, specify the `actions`:
                                   - **strengthen**: If connections should be made. Provide a list of connection objects. Each object must specify the 'id' of the neighbor memory (use the 'memory id' field from the neighbor description or the memory index value, not a made-up ID), the relationship 'type' (e.g., 'similar', 'causal', 'example', 'contradicts', 'extends', 'supports', 'refutes', 'prerequisite', 'related'), and a confidence 'strength' score (0.0 to 1.0). 
                                     IMPORTANT: Only suggest connections with strength >= 0.65 and only if a genuine, meaningful relationship exists. Do not force connections - it's better to have fewer high-quality links than many weak links. For each connection, include a brief 'reason' explaining why this connection is meaningful.
                                     CRITICAL: When specifying the 'id' field, do NOT generate new random UUIDs. You MUST use either:
                                     1. The memory index number (0, 1, 2, etc.) from the memory listing, which is the index prefixed to each memory in the neighbors
                                     2. The exact memory id value that appears in the "memory id:" field of each neighbor memory
                                     Never make up new memory IDs - only use IDs that are explicitly provided to you.
                                     Also, provide the potentially updated 'tags' for the *current* new memory note in `tags_to_update`.
                                   
                                   - **update_neighbor**: If neighbors should be updated. Provide the updated 'context' (max 2 lines, while keeping essence of original) strings in `new_context_neighborhood` and updated 'tags' lists in `new_tags_neighborhood` for the neighbors, in the same order they were presented. If a neighbor's context or tags are not updated, repeat the original values.

                                   - **merge_memories**: If the new memory should be merged with one or more existing memories if it's valid for either of the following conditions:
                                     * Information is complementary or continuation/extension and would form a more coherent whole when combined
                                     * Same core subject/topic/event with distributed details across memories
                                     * Content has high semantic overlap
                                                                          
                                     For each merge candidate, specify:
                                     * The 'id' of the memory to merge with (from 'memory id' field)
                                     * The 'merge_strategy': "combine" (take best from both), "augment" (keep primary but add details), "replace" (new memory supplants old)
                                     * A 'reasoning' explaining why these memories should be merged

                                Return your decision strictly in the following JSON format:
                                {{
                                    "should_evolve": true | false,
                                    "actions": ["strengthen" | "update_neighbor" | "merge_memories", ...],
                                    "suggested_connections": [ {{ "id": "neighbor_memory_id", "type": "relationship_type", "strength": 0.65-1.0, "reason": "brief explanation of connection" }}, ... ], 
                                    "tags_to_update": ["tag1", "tag2", ...],
                                    "new_context_neighborhood": ["neighbor1 new context", "neighbor2 new context", ...],
                                    "new_tags_neighborhood": [ ["tag1", ...], ["tagA", ...], ... ],
                                    "merge_candidates": [ {{ 
                                        "id": "memory_id_to_merge_with", 
                                        "merge_strategy": "combine|augment|replace",
                                        "reasoning": "explanation of why these should be merged" 
                                    }}, ... ]
                                }}
                                Ensure the output is a single, valid JSON object.
                                '''
        
    def analyze_content(self, content: str) -> Dict:            
        """Analyze content using LLM to extract semantic metadata."""
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }})
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, user_id: Optional[str] = None, 
                session_id: Optional[str] = None, **kwargs) -> str:
        """Add a new memory note with support for user/session and memory tiers"""
        if time is not None:
            kwargs['timestamp'] = time
        
        if user_id is not None:
            kwargs['user_id'] = user_id
        if session_id is not None:
            kwargs['session_id'] = session_id
        
        note = MemoryNote(content=content, **kwargs)
        
        self._process_through_tiers(note, user_id, session_id)
        
        evo_label, note = self.process_memory(note)
        print("\nevo_label: ", evo_label, "\n")
        
        if hasattr(note, 'merged') and note.merged:
            print(f"\nNote was merged into existing memory: {note.merged_into}\n")
            return note.merged_into
        
        self.memories[note.id] = note
        
        # Ensure the note is saved with any updated links
        if evo_label == True:
            # Re-save to LTM with updated links
            metadata = {
                "id": note.id,
                "content": note.content,
                "keywords": note.keywords,
                "links": note.links,
                "retrieval_count": note.retrieval_count,
                "timestamp": note.timestamp,
                "last_accessed": note.last_accessed,
                "context": note.context,
                "evolution_history": note.evolution_history,
                "category": note.category,
                "tags": note.tags,
                "user_id": user_id,
                "session_id": session_id
            }
            self.ltm._get_collection(user_id, session_id).add_document(
                document=note.content, 
                metadata=metadata, 
                doc_id=note.id
            )
            
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        
        return note.id
    
    def _process_through_tiers(self, note: MemoryNote, user_id: Optional[str] = None, 
                              session_id: Optional[str] = None):
        """Process a memory through both STM and LTM tiers"""
        base_metadata = {
            "keywords": note.keywords,
            "tags": note.tags,
            "context": note.context,
            "category": note.category,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "evolution_history": note.evolution_history,
            "user_id": user_id,
            "session_id": session_id
        }
        
        stm_metadata = self.light_processor.process(note.content, base_metadata)
        self.stm.add(note.id, note.content, stm_metadata, user_id, session_id)
        
        ltm_metadata = self.deep_processor.process(note.content, stm_metadata)
        self.ltm.add(note.id, note.content, ltm_metadata, user_id, session_id)
        
        if ltm_metadata != stm_metadata:
            self.stm.add(note.id, note.content, ltm_metadata, user_id, session_id)
    
    def read(self, memory_id: str, user_id: Optional[str] = None, 
            session_id: Optional[str] = None) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID."""
        stm_result = self.stm.get(memory_id, user_id, session_id)
        if stm_result:
            return self._dict_to_memory_note(stm_result)
        
        ltm_result = self.ltm.get(memory_id, user_id, session_id)
        if ltm_result:
            return self._dict_to_memory_note(ltm_result)
        
        return self.memories.get(memory_id)
    
    def _dict_to_memory_note(self, memory_dict: Dict) -> MemoryNote:
        """Convert a dictionary to a MemoryNote object"""
        note = MemoryNote(
            content=memory_dict.get("content", ""),
            id=memory_dict.get("id", None),
            keywords=memory_dict.get("keywords", []),
            links=memory_dict.get("links", {}),
            retrieval_count=memory_dict.get("retrieval_count", 0),
            timestamp=memory_dict.get("timestamp", None),
            last_accessed=memory_dict.get("last_accessed", None),
            context=memory_dict.get("context", None),
            evolution_history=memory_dict.get("evolution_history", []),
            category=memory_dict.get("category", None),
            tags=memory_dict.get("tags", []),
            user_id=memory_dict.get("user_id", None),
            session_id=memory_dict.get("session_id", None)
        )
        return note
    
    def search_memory(self, 
                     query: str, 
                     limit: int = 5, 
                     memory_source: str = "all",
                     where_filter: Optional[Dict] = None,
                     apply_postprocessing: bool = True,
                     context: Optional[str] = None,
                     user_id: Optional[str] = None,
                     session_id: Optional[str] = None) -> List[Dict]:
        """
        Search memories in specified tiers.
        
        Args:
            query: The search query text
            limit: Maximum number of results
            memory_source: Which memory tier to search ("stm", "ltm", "all")
            where_filter: Optional metadata filter
            apply_postprocessing: Whether to apply post-processing to results
            context: Optional conversation context for improved relevance
            user_id: Optional user ID to search within
            session_id: Optional session ID to search within
            
        Returns:
            List of matching memories with relevance scores
        """
        results = []
        
        if memory_source in ["stm", "all"]:
            stm_results = self.stm.search(
                query, 
                limit, 
                where_filter=where_filter,
                user_id=user_id,
                session_id=session_id
            )
            for result in stm_results:
                result["memory_tier"] = "stm"
            results.extend(stm_results)
        
        if memory_source in ["ltm", "all"]:
            ltm_results = self.ltm.search(
                query, 
                limit, 
                where_filter=where_filter,
                user_id=user_id,
                session_id=session_id
            )
            for result in ltm_results:
                result["memory_tier"] = "ltm"
                
            if apply_postprocessing and context:
                ltm_results = self.retrieval_processor.process(ltm_results, context)
                
            results.extend(ltm_results)
        
        # Check if we have results
        if not results and user_id:
            logger.warning(f"No results found for user {user_id} - trying search without user filtering")
            
            # Try a general search without user filtering as a fallback
            general_results = []
            
            if memory_source in ["ltm", "all"]:
                ltm_general_results = self.ltm.search(
                    query, 
                    limit, 
                    where_filter=where_filter,
                    user_id=None,  # No user filtering
                    session_id=None
                )
                for result in ltm_general_results:
                    result["memory_tier"] = "ltm"
                    
                if apply_postprocessing and context:
                    ltm_general_results = self.retrieval_processor.process(ltm_general_results, context)
                    
                general_results.extend(ltm_general_results)
            
            # Only use these results if we found nothing with the user filter
            if general_results:
                logger.info(f"Found {len(general_results)} results in general search without user filtering")
                results = general_results
        
        # Deduplicate by memory ID, prioritizing STM results
        seen_ids = set()
        unique_results = []
        
        # First add STM results
        for result in results:
            if result.get("memory_tier") == "stm":
                memory_id = result.get("id")
                if memory_id and memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    unique_results.append(result)
        
        # Then add LTM results if not already seen
        for result in results:
            if result.get("memory_tier") == "ltm":
                memory_id = result.get("id")
                if memory_id and memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    unique_results.append(result)
        
        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return unique_results[:limit]
    
    def search_agentic(self, query: str, limit: int = 5, user_id: Optional[str] = None, 
                      session_id: Optional[str] = None) -> List[Dict]:
        """
        Search for memories using the new tiered architecture.
        Provides backward compatibility with the original search_agentic method.
        """
        return self.search_memory(
            query=query,
            limit=limit,
            memory_source="all",
            apply_postprocessing=True,
            user_id=user_id,
            session_id=session_id
        )
    
    def search_filtered(self, query: str, limit: int = 5, where_filter: Optional[Dict] = None,
                       user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict]:
        """
        Search with filters using the new tiered architecture.
        Provides backward compatibility with original search_filtered method.
        """
        return self.search_memory(
            query=query,
            limit=limit,
            memory_source="all",
            where_filter=where_filter,
            apply_postprocessing=True,
            user_id=user_id,
            session_id=session_id
        )
    
    def clear_stm(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Clear short-term memory for a specific user/session or all if none specified"""
        self.stm.clear(user_id, session_id)
    
    def clear_ltm(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Clear long-term memory for a specific user/session or all if none specified"""
        self.ltm.clear(user_id, session_id)
    
    def consolidate_memories(self):
        """Consolidate memories: re-add all memories to ensure they are in correct collections"""
        # Group memories by user_id and session_id
        memory_groups = {}
        
        for memory in self.memories.values():
            user_id = getattr(memory, 'user_id', None)
            session_id = getattr(memory, 'session_id', None)
            key = (user_id, session_id)
            
            if key not in memory_groups:
                memory_groups[key] = []
            
            memory_groups[key].append(memory)
        
        # Process each group separately
        for (user_id, session_id), memories in memory_groups.items():
            collection = self.ltm._get_collection(user_id, session_id)
            
            # Don't clear - just update existing memories
            for memory in memories:
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
                    "category": memory.category,
                    "tags": memory.tags,
                    "user_id": user_id,
                    "session_id": session_id
                }
                collection.add_document(memory.content, metadata, memory.id)
    
    def find_related_memories(self, query: str, k: int = 5, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Tuple[str, List[str]]:
        """Find related memories using ChromaDB retrieval
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            user_id (Optional[str]): User ID to filter memories by
            session_id (Optional[str]): Session ID to filter memories by
            
        Returns:
            Tuple[str, List[str]]: A tuple containing (formatted_memory_string, memory_ids)
        """
        if not self.memories:
            return "", []
            
        try:
            # Use the proper LTM collection based on user_id and session_id
            ltm_collection = self.ltm._get_collection(user_id, session_id)
            
            where_conditions = []
            
            if user_id is not None:
                where_conditions.append({"user_id": {"$eq": user_id}})
            
            if session_id is not None:
                where_conditions.append({"session_id": {"$eq": session_id}})
            
            where_filter = {"$and": where_conditions} if len(where_conditions) > 1 else (where_conditions[0] if where_conditions else None)
            
            results = ltm_collection.search(query, k=k, where_filter=where_filter)
            
            memory_str = ""
            memory_ids = []
            
            for i, result in enumerate(results):
                doc_id = result.get('id')
                if not doc_id:
                    continue
                    
                metadata = result.get('metadata', {})
                
                memory_str += f"memory index:{i}\tmemory id:{doc_id}\ttimestamp:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                memory_ids.append(doc_id)
            
            return memory_str, memory_ids
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", []
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note.
        
        Args:
            memory_id: ID of memory to update
            **kwargs: Fields to update
            
        Returns:
            bool: True if update successful
        """
        if memory_id not in self.memories:
            return False
            
        note = self.memories[memory_id]
        
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags,
            "user_id": getattr(note, 'user_id', None),
            "session_id": getattr(note, 'session_id', None)
        }
        
        # Update in the correct LTM collection
        user_id = getattr(note, 'user_id', None)
        session_id = getattr(note, 'session_id', None)
        ltm_collection = self.ltm._get_collection(user_id, session_id)
        ltm_collection.add_document(document=note.content, metadata=metadata, doc_id=memory_id)
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted, False if not found
        """
        if memory_id in self.memories:
            note = self.memories[memory_id]
            user_id = getattr(note, 'user_id', None)
            session_id = getattr(note, 'session_id', None)
            
            # Delete from LTM using the correct collection
            self.ltm._get_collection(user_id, session_id).delete_document(memory_id)
            
            # Also delete from STM
            self.stm.delete(memory_id, user_id, session_id)
            
            # Delete from memory dictionary
            del self.memories[memory_id]
            return True
        return False

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note and determine if it should evolve.
        
        Args:
            note: The memory note to process
            
        Returns:
            Tuple[bool, MemoryNote]: (should_evolve, processed_note)
        """
        if not self.memories:
            return False, note
            
        # Skip evolution for very large content (likely batched data)
        # TODO: Implement evolution for large batched content
        if len(note.content) > 10000:
            logger.info("Skipping evolution for large batched content")
            return False, note
            
        try:
            user_id = getattr(note, 'user_id', None)
            session_id = getattr(note, 'session_id', None)
            
            logger.info(f"Processing memory for user: {user_id}, session: {session_id}")
            
            neighbors_text, neighbor_memory_ids = self.find_related_memories(note.content, k=4, user_id=user_id, session_id=session_id)
            
            if not neighbors_text or not neighbor_memory_ids:
                return False, note
               
            logger.info(f"Processing memory with content: {note.content[:100]}...")
            logger.info(f"Found {len(neighbor_memory_ids)} neighbors for potential evolution")
            
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(neighbor_memory_ids)
            )
            
            try:
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean"
                                },
                                "actions": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "type": {"type": "string"},
                                            "strength": {"type": "number"},
                                            "reason": {"type": "string"}
                                        },
                                        "required": ["id", "type", "strength", "reason"],
                                        "additionalProperties": False
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": { "type": "string" }
                                    }
                                },
                                "merge_candidates": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "merge_strategy": {"type": "string", "enum": ["combine", "augment", "replace"]},
                                            "reasoning": {"type": "string"}
                                        },
                                        "required": ["id", "merge_strategy", "reasoning"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["should_evolve", "actions", "suggested_connections", 
                                      "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood", "merge_candidates"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
                )
                
                response_json = json.loads(response)
                should_evolve = response_json.get("should_evolve", False)
                
                actions = response_json.get("actions", [])
                logger.info(f"Evolution decision - should_evolve: {should_evolve}, actions: {actions}")
                
                if "merge_memories" in actions:
                    merge_candidates = response_json.get("merge_candidates", [])
                    logger.info(f"Merge candidates detected: {len(merge_candidates)}")
                    for candidate in merge_candidates:
                        logger.info(f"Merge candidate: id={candidate.get('id')}, strategy={candidate.get('merge_strategy')}")
                
                if should_evolve:
                    if not isinstance(note.links, dict):
                        if isinstance(note.links, list):
                            logger.warning(f"Converting old list-based links for note {note.id} to dict format.")
                            old_links = note.links
                            note.links = {}
                            for link_id in old_links:
                                if self._is_valid_uuid(link_id):
                                     note.links[link_id] = {}  
                        else:
                           note.links = {}

                    for action in actions:
                        if action == "strengthen":
                            suggested_connections = response_json.get("suggested_connections", [])
                            new_tags = response_json.get("tags_to_update", note.tags)
                            
                            note.tags = new_tags
                            
                            current_time_iso = datetime.now().isoformat()
                            for conn_obj in suggested_connections:
                                conn_id_str = conn_obj.get("id")
                                conn_type = conn_obj.get("type", "related")
                                conn_strength = conn_obj.get("strength", 0)
                                conn_reason = conn_obj.get("reason", "")
                                
                                target_memory_id = None
                                
                                if isinstance(conn_id_str, str):
                                    if conn_id_str.isdigit():
                                        idx = int(conn_id_str)
                                        if 0 <= idx < len(neighbor_memory_ids):
                                            target_memory_id = neighbor_memory_ids[idx]
                                    elif self._is_valid_uuid(conn_id_str):
                                        # Only check in neighbor_memory_ids to avoid connecting to memories from different users/sessions
                                        if conn_id_str in neighbor_memory_ids:
                                            target_memory_id = conn_id_str
                                        elif conn_id_str in self.memories:
                                            # Additional check: only connect if user_id and session_id match
                                            target_memory = self.memories[conn_id_str]
                                            target_user_id = getattr(target_memory, 'user_id', None)
                                            target_session_id = getattr(target_memory, 'session_id', None)
                                            note_user_id = getattr(note, 'user_id', None)
                                            note_session_id = getattr(note, 'session_id', None)
                                            
                                            if target_user_id == note_user_id and target_session_id == note_session_id:
                                                target_memory_id = conn_id_str
                                            else:
                                                logger.warning(f"Skipping connection to memory {conn_id_str} due to user/session mismatch")
                                        else:
                                            logger.warning(f"LLM suggested connection to non-existent memory ID: {conn_id_str}")

                                if target_memory_id and target_memory_id != note.id:
                                    note.links[target_memory_id] = {
                                        'type': conn_type, 
                                        'strength': conn_strength, 
                                        'timestamp': current_time_iso,
                                        'reason': conn_reason
                                    }
                                    
                                    logger.info(f"Added link from {note.id} to {target_memory_id}: {note.links[target_memory_id]}")
                                    
                                    neighbor_note = self.memories.get(target_memory_id)
                                    if neighbor_note:
                                        if not isinstance(neighbor_note.links, dict):
                                             neighbor_note.links = {} 
                                        
                                        reciprocal_type = self._get_reciprocal_relationship(conn_type)
                                        reciprocal_strength = conn_strength * 0.9
                                             
                                        neighbor_note.links[note.id] = {
                                            'type': reciprocal_type,
                                            'strength': max(0.65, reciprocal_strength),
                                            'timestamp': current_time_iso,
                                            'reason': f"Reciprocal of: {conn_reason}"
                                        }
                                        
                                        logger.info(f"Added reciprocal link from {target_memory_id} to {note.id}: {neighbor_note.links[note.id]}")
                                        
                                        # Important: Update the neighbor in its correct collection
                                        neighbor_user_id = getattr(neighbor_note, 'user_id', None)
                                        neighbor_session_id = getattr(neighbor_note, 'session_id', None)
                                        
                                        neighbor_metadata = {
                                            "id": neighbor_note.id,
                                            "content": neighbor_note.content,
                                            "keywords": neighbor_note.keywords,
                                            "links": neighbor_note.links,
                                            "retrieval_count": neighbor_note.retrieval_count,
                                            "timestamp": neighbor_note.timestamp,
                                            "last_accessed": neighbor_note.last_accessed,
                                            "context": neighbor_note.context,
                                            "evolution_history": neighbor_note.evolution_history,
                                            "category": neighbor_note.category,
                                            "tags": neighbor_note.tags,
                                            "user_id": neighbor_user_id,
                                            "session_id": neighbor_session_id
                                        }
                                        
                                        # Update the neighbor in the LTM
                                        self.ltm._get_collection(neighbor_user_id, neighbor_session_id).add_document(
                                            document=neighbor_note.content,
                                            metadata=neighbor_metadata,
                                            doc_id=target_memory_id
                                        )
                                        
                                        # Also update in the memory dictionary
                                        self.memories[target_memory_id] = neighbor_note
                                    else:
                                         logger.warning(f"Could not find neighbor note {target_memory_id} to add backward link.")
                            
                        elif action == "update_neighbor":
                            new_context_neighborhood = response_json.get("new_context_neighborhood", [])
                            new_tags_neighborhood = response_json.get("new_tags_neighborhood", [])
                            
                            for i in range(min(len(neighbor_memory_ids), len(new_tags_neighborhood))):
                                if i >= len(neighbor_memory_ids): continue

                                memory_id = neighbor_memory_ids[i]
                                memory = self.memories.get(memory_id)
                                if not memory: continue

                                if i < len(new_tags_neighborhood) and memory.tags != new_tags_neighborhood[i]:
                                    memory.tags = new_tags_neighborhood[i]
                                
                                if i < len(new_context_neighborhood) and memory.context != new_context_neighborhood[i]:
                                    memory.context = new_context_neighborhood[i]
                                
                                self.memories[memory_id] = memory
                        
                        elif action == "merge_memories":
                            merge_candidates = response_json.get("merge_candidates", [])
                            if merge_candidates:
                                logger.info(f"Processing {len(merge_candidates)} merge candidates")
                                for merge_candidate in merge_candidates:
                                    target_id_str = merge_candidate.get("id")
                                    target_memory_id = self._resolve_memory_id(target_id_str, neighbor_memory_ids)
                                    
                                    if target_memory_id and target_memory_id in self.memories:
                                        logger.info(f"Attempting to merge note {note.id} with {target_memory_id}")
                                        self._merge_memories(
                                            note, 
                                            target_memory_id,
                                            merge_candidate.get("merge_strategy", "combine"),
                                            merge_candidate.get("reasoning", "Memories contain similar or complementary information")
                                        )
                                    else:
                                        logger.warning(f"Cannot merge with non-existent memory: {target_id_str}")
                                
                return should_evolve, note
                
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.error(f"Error processing LLM evolution response: {str(e)}")
                logger.error(f"Response was: {response}")
                return False, note
                
        except Exception as e:
            logger.error(f"Error in process_memory outer block: {str(e)}")
            return False, note

    def _resolve_memory_id(self, id_str: str, neighbor_memory_ids: List[str]) -> Optional[str]:
        """Resolve a potential memory ID from various formats.
        
        Args:
            id_str: The ID string to resolve (could be index, path, or UUID)
            neighbor_memory_ids: List of neighbor memory IDs for index-based resolution
            
        Returns:
            Resolved memory ID or None if invalid
        """
        if not isinstance(id_str, str):
            return None
            
        if id_str.isdigit():
            idx = int(id_str)
            if 0 <= idx < len(neighbor_memory_ids):
                return neighbor_memory_ids[idx]
                
        elif self._is_valid_uuid(id_str) and id_str in self.memories:
            return id_str
            
        return None

    def _merge_memories(self, new_note: MemoryNote, target_id: str, 
                        strategy: str, reasoning: str) -> None:
        """Merge two memory notes based on the specified strategy.
        
        Args:
            new_note: The new memory note
            target_id: ID of the existing memory to merge with
            strategy: Merge strategy ('combine', 'augment', or 'replace')
            reasoning: Explanation for why memories should be merged
        """
        target_note = self.memories.get(target_id)
        if not target_note:
            logger.error(f"Cannot merge: target memory {target_id} not found")
            return
            
        merge_time = datetime.now().isoformat()
        user_id = getattr(target_note, 'user_id', None)
        session_id = getattr(target_note, 'session_id', None)
        
        if strategy == "replace":
            merged_content = new_note.content
            
            merge_record = {
                "type": "merge_replace",
                "timestamp": merge_time,
                "merged_from": new_note.id,
                "previous_content": target_note.content,
                "reasoning": reasoning
            }
            
            if not isinstance(target_note.evolution_history, list):
                target_note.evolution_history = []
            target_note.evolution_history.append(merge_record)
            
            target_note.content = merged_content
            
            self._combine_metadata(target_note, new_note)
            
            new_note.evolution_history.append({
                "type": "merged_into",
                "timestamp": merge_time,
                "target": target_id,
                "reasoning": reasoning
            })
            
            new_note.merged = True
            new_note.merged_into = target_id
            
        elif strategy == "augment" or strategy == "combine":
            merged_content = f"{target_note.content}\n\n\n{new_note.content}"
            
            merge_record = {
                "type": "merge_augment",
                "timestamp": merge_time,
                "augmented_with": new_note.id,
                "previous_content": target_note.content,
                "reasoning": reasoning
            }
            
            if not isinstance(target_note.evolution_history, list):
                target_note.evolution_history = []
            target_note.evolution_history.append(merge_record)
            
            target_note.content = merged_content
            
            self._combine_metadata(target_note, new_note)
            
            new_note.evolution_history.append({
                "type": "merged_into",
                "timestamp": merge_time,
                "target": target_id,
                "reasoning": reasoning
            })
            new_note.merged = True
            new_note.merged_into = target_id
        
        # Merge links between the memories
        self._merge_links(target_note, new_note)
        
        metadata = {
            "id": target_note.id,
            "content": target_note.content,
            "keywords": target_note.keywords,
            "links": target_note.links,
            "retrieval_count": target_note.retrieval_count,
            "timestamp": target_note.timestamp,
            "last_accessed": target_note.last_accessed,
            "context": target_note.context,
            "evolution_history": target_note.evolution_history,
            "category": target_note.category,
            "tags": target_note.tags,
            "user_id": user_id,
            "session_id": session_id
        }
        
        self.ltm._get_collection(user_id, session_id).add_document(document=target_note.content, metadata=metadata, doc_id=target_id)
        
        logger.info(f"Merged memory {new_note.id} into {target_id} using strategy '{strategy}'")
        logger.info(f"Merged content reasoning: {reasoning}")

    def _combine_metadata(self, target_note: MemoryNote, source_note: MemoryNote) -> None:
        """Combine metadata from two memory notes.
        
        Args:
            target_note: The target memory note to update
            source_note: The source memory note to take metadata from
        """
        combined_keywords = list(set(target_note.keywords + source_note.keywords))
        target_note.keywords = combined_keywords
        
        combined_tags = list(set(target_note.tags + source_note.tags))
        target_note.tags = combined_tags
        
        try:
            def parse_timestamp(ts):
                formats = [
                    "%Y%m%d%H%M",               # Basic format (YYYYMMDDhhmm)
                    "%Y%m%d%H%M.%f",            # With microseconds
                    "%Y%m%d%H%M%S.%f",          # With seconds and microseconds
                    "%Y-%m-%dT%H:%M:%S.%f",     # ISO format
                    "%Y-%m-%dT%H:%M:%S.%fZ"     # ISO format with Z
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(ts, fmt)
                    except ValueError:
                        continue
                
                logger.warning(f"Could not parse timestamp: {ts}")
                return datetime(2000, 1, 1)
            
            target_time = parse_timestamp(target_note.timestamp)
            source_time = parse_timestamp(source_note.timestamp)
            
            if source_time > target_time:
                target_note.timestamp = source_note.timestamp
        except Exception as e:
            logger.warning(f"Error comparing timestamps: {e}. Using target timestamp.")
        
        target_note.retrieval_count = max(target_note.retrieval_count, source_note.retrieval_count)
        
        if source_note.context != "General" and source_note.context != "Uncategorized":
            target_note.context = source_note.context
            
        if source_note.category != "Uncategorized" and source_note.category != target_note.category:
            target_note.category = f"{target_note.category}/{source_note.category}"

    def _merge_links(self, target_note: MemoryNote, source_note: MemoryNote) -> None:
        """Merge links from source note into target note.
        
        Args:
            target_note: The target memory note to update
            source_note: The source memory note to take links from
        """
        # Ensure target links is a dictionary
        if not isinstance(target_note.links, dict):
            if isinstance(target_note.links, str):
                try:
                    target_note.links = json.loads(target_note.links)
                except json.JSONDecodeError:
                    target_note.links = {}
            else:
                target_note.links = {}
        
        # Ensure source links is a dictionary
        source_links = source_note.links
        if isinstance(source_links, str):
            try:
                source_links = json.loads(source_links)
            except json.JSONDecodeError:
                source_links = {}
        
        if not isinstance(source_links, dict):
            return
            
        for link_id, link_data in source_links.items():
            if link_id == target_note.id or link_id == source_note.id:
                continue
                
            if link_id in target_note.links:
                existing_strength = target_note.links[link_id].get('strength', 0)
                new_strength = link_data.get('strength', 0)
                
                if new_strength > existing_strength:
                    target_note.links[link_id] = link_data
            else:
                target_note.links[link_id] = link_data
                
        current_time_iso = datetime.now().isoformat()
        
        # Update all memories that link to the source note to now link to the target note
        for link_id in target_note.links:
            linked_note = self.memories.get(link_id)
            if linked_note and isinstance(linked_note.links, dict):
                # If linked note references the source note, update it to point to target
                if source_note.id in linked_note.links and target_note.id not in linked_note.links:
                    linked_note.links[target_note.id] = linked_note.links[source_note.id].copy()
                    if isinstance(linked_note.links[target_note.id], dict):
                        linked_note.links[target_note.id]['timestamp'] = current_time_iso
                        linked_note.links[target_note.id]['note'] = f"Link transferred from merged memory {source_note.id}"
                    
                    del linked_note.links[source_note.id]
                    
                    # Update the linked note in LTM storage
                    user_id = getattr(linked_note, 'user_id', None)
                    session_id = getattr(linked_note, 'session_id', None)
                    metadata = {
                        "id": linked_note.id,
                        "content": linked_note.content,
                        "keywords": linked_note.keywords,
                        "links": linked_note.links,
                        "retrieval_count": linked_note.retrieval_count,
                        "timestamp": linked_note.timestamp,
                        "last_accessed": linked_note.last_accessed,
                        "context": linked_note.context,
                        "evolution_history": linked_note.evolution_history,
                        "category": linked_note.category,
                        "tags": linked_note.tags,
                        "user_id": user_id,
                        "session_id": session_id
                    }
                    
                    # Update in the correct collection
                    self.ltm._get_collection(user_id, session_id).add_document(
                        document=linked_note.content, 
                        metadata=metadata, 
                        doc_id=link_id
                    )
                    
                    # Also update in STM if present
                    self.stm.add(
                        link_id, 
                        linked_note.content, 
                        metadata, 
                        user_id, 
                        session_id
                    )
                    
                    # Update in memory dictionary
                    self.memories[link_id] = linked_note

    def _is_valid_uuid(self, val: str) -> bool:
        """Check if a string is a valid UUID
        
        Args:
            val: String to check
            
        Returns:
            bool: True if valid UUID format
        """
        if not isinstance(val, str):
            return False
            
        try:
            uuid_obj = uuid.UUID(val)
            return str(uuid_obj) == val
        except (ValueError, AttributeError):
            return False

    def _get_reciprocal_relationship(self, relationship_type: str) -> str:
        """Get the reciprocal relationship type.
        
        Args:
            relationship_type: The original relationship type
            
        Returns:
            str: The reciprocal relationship type
        """
        reciprocals = {
            'supports': 'supported_by',
            'supported_by': 'supports',
            'contradicts': 'contradicted_by',
            'contradicted_by': 'contradicts',
            'refutes': 'refuted_by',
            'refuted_by': 'refutes',
            'extends': 'extended_by',
            'extended_by': 'extends',
            'prerequisite': 'depends_on',
            'depends_on': 'prerequisite',
            'example': 'exemplified_by',
            'exemplified_by': 'example',
            'causal': 'effect_of',
            'effect_of': 'causal'
        }
        
        return reciprocals.get(relationship_type, relationship_type)

    def _get_embedding(self, content: str) -> List[float]:
        """Generate embeddings for text content using sentence transformer model"""
        from sentence_transformers import SentenceTransformer
        
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
            
        embedding = self._model.encode(content)
        return embedding.tolist()
