from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime, timedelta
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from cortex.memory_note import MemoryNote
from cortex.stm import ShortTermMemory
from cortex.ltm import LongTermMemory
from cortex.processors import LightPreprocessor, DeepPreprocessor, RetrievalProcessor
from cortex.llm_controllers.llm_controller import LLMController
from cortex.embedding_manager import EmbeddingManager
from .collection_manager import CollectionManager
from cortex.constants import (
    LARGE_CONTENT_THRESHOLD, MIN_CONNECTION_STRENGTH, DEFAULT_LTM_WORKERS,
    DEFAULT_STM_CAPACITY, DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, 
    DEFAULT_LLM_BACKEND, DEFAULT_SEARCH_LIMIT, DEFAULT_RELATED_MEMORIES_COUNT,
    COLLECTION_SIMILARITY_WEIGHT, ITEM_SIMILARITY_WEIGHT, DEFAULT_CHROMA_URI
)

logger = logging.getLogger(__name__)

class AgenticMemorySystem:
    
    def __init__(self, 
                 model_name: str = DEFAULT_EMBEDDING_MODEL,
                 llm_backend: str = DEFAULT_LLM_BACKEND,
                 llm_model: str = DEFAULT_LLM_MODEL,
                 stm_capacity: int = DEFAULT_STM_CAPACITY,
                 api_key: Optional[str] = None,
                 enable_smart_collections: bool = False,
                 enable_background_processing: bool = True,
                 chroma_uri: str = DEFAULT_CHROMA_URI):  
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.enable_smart_collections = enable_smart_collections
        self.memories = {}
        
        self.llm_controller = LLMController(llm_backend, llm_model, api_key)
        
        self.stm = ShortTermMemory(capacity=stm_capacity, model_name=model_name)
        self.ltm = LongTermMemory(embedding_model=model_name, chroma_uri=chroma_uri)
        
        self.light_processor = LightPreprocessor(self)
        self.deep_processor = DeepPreprocessor(self)
        self.retrieval_processor = RetrievalProcessor(self)
        
        self.embedding_manager = EmbeddingManager(model_name)
        
        if enable_background_processing:
            self._ltm_executor = ThreadPoolExecutor(max_workers=DEFAULT_LTM_WORKERS, thread_name_prefix="ltm_processor")
            logger.info("Background processing enabled for LTM operations")
        else:
            self._ltm_executor = None
            logger.info("Synchronous processing enabled - all operations will complete before returning")
        
        if self.enable_smart_collections:
            self.collection_manager = CollectionManager(self)
        

        self._evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Your goal is to build a rich, interconnected network of memories.
                                Analyze the new memory note (content, context, keywords) and its relationship with the nearest neighbor memories provided.

                                New memory note:
                                Content: {content}
                                Context: {context}
                                Keywords: {keywords}

                                Nearest neighbors memories found (up to {neighbor_number}):
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
                                     Never make up new memory IDs - ONLY use IDs that are explicitly provided to you (BUT ONLY IF THEY ARE ACTUALLY RELATED TO THE CURRENT MEMORY)
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
                                
                                ::VERY IMPORTANT::
                                - Set should_evolve to true only if there's a VALID and somewhat CONFIDENT REASON to do so.
                                - You do not have to make connections with all the neighbors, only make connections with memory where you ACTUALLY see some sort of relationship in between. Do not FORCE connections.
                                - It's better to have fewer high-quality links than many weak links.

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
        if self.enable_smart_collections:
            existing_categories_context = ""
            if hasattr(self, 'collection_manager') and self.collection_manager:
                existing_categories_context = self.collection_manager.get_existing_categories_context()
                logger.debug(f"Using existing categories context: {len(self.collection_manager.category_counts)} patterns available")
            
            prompt = f"""Generate a structured analysis of the following content by:
                1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
                2. Extracting core themes and contextual elements  
                3. Creating relevant categorical tags
                4. Generating a hierarchical category using dot notation, be highly specific to the content

                **Category Examples (Content → Category (These are quite small but in real content will mostly be much longer)):**
                
                Content: "Fixed Django ORM bug with nested queries causing slow performance"
                Category: "work.programming.python"
                
                Content: "Team standup meeting discussed sprint goals and blockers"  
                Category: "work.meetings.standup"
                
                Content: "Started intermittent fasting 16:8 schedule, feeling more energetic"
                Category: "personal.health.fitness"
                
                Content: "Set up automatic savings transfer of $500/month to emergency fund"
                Category: "personal.finance.budgeting"
                
                Content: "Learned Spanish past tense conjugations - ser vs estar usage"
                Category: "education.languages.spanish"
                
                Content: "Implemented React hooks for API state management in the frontend"
                Category: "projects.app.frontend"

                **Existing Categories Context**
                {existing_categories_context}

                **Category Rules (CRITICAL FOR CONSISTENCY):**
                - Use 2-4 levels: domain.subdomain.specific  
                - Keep consistent naming (lowercase, no spaces)
                - STRONGLY PREFER EXISTING SUBSTANTIAL PATTERNS: If content relates to existing categories with N+ memories, USE THEM
                - AVOID UNNECESSARY FRAGMENTATION: Don't create work.programming.python.django if work.programming.python fits well
                - CREATE NEW ONLY FOR DISTINCT DOMAINS: Only create new patterns for genuinely different domains/topics
                - EXTEND THOUGHTFULLY: Add hierarchy levels only when content is significantly different from existing patterns
                - GROW COLLECTIONS NATURALLY: Let successful categories accumulate related memories rather than fragmenting them

                Format the response as a JSON object (all fields required, category must be specific and non-generic):
                {{
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
                    ],
                    "category":
                        // hierarchical category following the examples above
                        // Examples: "work.programming.python", "personal.finance.budgeting", "education.science.physics"
                        // Start with broad domain, then subdomain, then specific area
                        // Use 2-4 levels maximum
                        // Never use "Uncategorized", "General" or any generic bucket
                }}

                Content for analysis:
                {content}"""
        else:
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
            response_schema = {"type": "json_schema", "json_schema": {
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
                            },
                            "required": ["keywords", "context", "tags"]
                            }
                    }}
            if self.enable_smart_collections:
                response_schema["json_schema"]["schema"]["properties"]["category"] = {
                    "type": "string"
                        }
                response_schema["json_schema"]["schema"]["required"].append("category")
            response = self.llm_controller.llm.get_completion(prompt, response_format=response_schema)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, user_id: Optional[str] = None, 
                session_id: Optional[str] = None, **kwargs) -> str:
        """add a new memory note: STM immediate, LTM background processing"""
        if time is not None:
            kwargs['timestamp'] = time
        
        if user_id is not None:
            kwargs['user_id'] = user_id
        if session_id is not None:
            kwargs['session_id'] = session_id
        
        allowed_fields = {
            "id", "keywords", "links", "retrieval_count", "timestamp",
            "last_accessed", "context", "evolution_history", "category",
            "tags", "user_id", "session_id"
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_fields}
        note = MemoryNote(content=content, **filtered_kwargs)
        self._process_through_tiers(note, user_id, session_id)
        self.memories[note.id] = note
        
        return note.id
    
    def shutdown(self):
        """Shutdown background processing threads"""
        if hasattr(self, '_ltm_executor') and self._ltm_executor is not None:
            self._ltm_executor.shutdown(wait=True)
    
    def _process_through_tiers(self, note: MemoryNote, user_id: Optional[str] = None, 
                              session_id: Optional[str] = None):
        """Process a memory through STM (immediate) and LTM (background or synchronous)"""
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
        
        if self._ltm_executor is not None:
            self._ltm_executor.submit(self._process_ltm_background, note, base_metadata, user_id, session_id)
        else:
            self._process_ltm_background(note, base_metadata, user_id, session_id)
    
    def _process_ltm_background(self, note: MemoryNote, base_metadata: Dict[str, Any], 
                               user_id: Optional[str], session_id: Optional[str]):
        """Background processing for LTM with deep analysis and evolution"""
        try:
            ltm_metadata = self.deep_processor.process(note.content, base_metadata)
            self.ltm.add(note.id, note.content, ltm_metadata, user_id, session_id)
            
            # Update category counts for smart collections
            if self.enable_smart_collections and hasattr(self, 'collection_manager'):
                category = ltm_metadata.get('category') or getattr(note, 'category', '')
                if category:
                    note.category = category
                    self.collection_manager.update_category_counts(category)
                    self.collection_manager.update_collection_metadata(category)
            
            evo_label, evolved_note = self.process_memory(note)
            
            if hasattr(evolved_note, 'merged') and evolved_note.merged:
                logger.info(f"Memory {note.id} was merged into {evolved_note.merged_into} during background processing")
                return
                
            self.memories[evolved_note.id] = evolved_note
            if evo_label:
                updated_metadata = {
                    "id": evolved_note.id,
                    "content": evolved_note.content,
                    "keywords": evolved_note.keywords,
                    "links": evolved_note.links,
                    "retrieval_count": evolved_note.retrieval_count,
                    "timestamp": evolved_note.timestamp,
                    "last_accessed": evolved_note.last_accessed,
                    "context": evolved_note.context,
                    "evolution_history": evolved_note.evolution_history,
                    "category": evolved_note.category,
                    "tags": evolved_note.tags,
                    "user_id": user_id,
                    "session_id": session_id
                }
                self.ltm._get_collection(user_id, session_id).add_document(
                    document=evolved_note.content, 
                    metadata=updated_metadata, 
                    doc_id=evolved_note.id
                )
                    
        except Exception as e:
            logger.error(f"Error in background LTM processing for {note.id}: {e}")
    
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
    
    def search(self, 
               query: str, 
               limit: int = DEFAULT_SEARCH_LIMIT, 
               memory_source: str = "all",
               where_filter: Optional[Dict] = None,
               apply_postprocessing: bool = True,
               context: Optional[str] = None,
               user_id: Optional[str] = None,
               session_id: Optional[str] = None,
               temporal_weight: float = 0.0,
                date_range: Optional[Dict] = None) -> List[Dict]:
        """
        Unified search interface for memories across STM and LTM.
        
        Args:
            query: The search query text
            limit: Maximum number of results
            memory_source: Which memory tier to search ("stm", "ltm", "all")
            where_filter: Optional metadata filter
            apply_postprocessing: Whether to apply post-processing to results
            context: Optional conversation context for improved relevance
            user_id: Optional user ID to search within
            session_id: Optional session ID to search within
            temporal_weight: Weight for temporal scoring (0.0=semantic only, 1.0=recency only)
            date_range: Optional temporal filter {"start": "YYYYMMDDHHMM", "end": "YYYYMMDDHHMM"}
            
        Returns:
            List of matching memories with relevance scores
        """
                # Perform global search (existing logic with relationships, evolution, etc.)
        # Accept date_range as string (natural language or yyyy-mm) and parse here if needed
        parsed_date_range = date_range
        if isinstance(date_range, str):
            parsed_date_range = self._parse_date_range(date_range)
        global_results = self._global_search(
            query, limit, memory_source, where_filter,
            apply_postprocessing, context, user_id, session_id, temporal_weight, parsed_date_range
        )
        
        # Enhance with collection-aware search if enabled
        if (self.enable_smart_collections and hasattr(self, 'collection_manager') 
            and self.collection_manager.collections):
            collection_results = self._collection_aware_search(query, limit, memory_source, 
                                                             where_filter, user_id, session_id, context)
            if apply_postprocessing and context:
                collection_results = self.retrieval_processor.process(collection_results, context)
                
            return self._merge_hybrid_results(global_results, collection_results, limit, temporal_weight)
        
        # Apply temporal weighting to global-only results if needed
        if temporal_weight > 0.0:
            global_results = self._apply_temporal_weighting(global_results, temporal_weight)
            global_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return global_results
    
    def _global_search(self, query: str, limit: int, memory_source: str, where_filter: Optional[Dict],
                      apply_postprocessing: bool, context: Optional[str], user_id: Optional[str], 
                      session_id: Optional[str], temporal_weight: float = 0.0, date_range: Optional[Dict] = None) -> List[Dict]:
        """Global search with relationships and evolution features"""
        results = []
        
        temporal_filter = self._build_temporal_filter(date_range, where_filter)
        
        if memory_source in ["stm", "all"] and not date_range:
            stm_results = self.stm.search(
                query, 
                limit, 
                where_filter=temporal_filter,
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
                where_filter=temporal_filter,
                user_id=user_id,
                session_id=session_id
            )
            for result in ltm_results:
                result["memory_tier"] = "ltm"
                
            if apply_postprocessing and context:
                ltm_results = self.retrieval_processor.process(ltm_results, context)
                
            results.extend(ltm_results)
        
        if not results and user_id:
            logger.warning(f"No results found for user {user_id} - trying search without user filtering")
            general_results = []
            
            if memory_source in ["ltm", "all"]:
                ltm_general_results = self.ltm.search(
                    query, 
                    limit, 
                    where_filter=temporal_filter,
                    user_id=None,
                    session_id=None
                )
                for result in ltm_general_results:
                    result["memory_tier"] = "ltm"
                    
                if apply_postprocessing and context:
                    ltm_general_results = self.retrieval_processor.process(ltm_general_results, context)
                    
                general_results.extend(ltm_general_results)
            
            if general_results:
                logger.info(f"Found {len(general_results)} results in general search without user filtering")
                results = general_results
        
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.get("memory_tier") == "stm":
                memory_id = result.get("id")
                if memory_id and memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    unique_results.append(result)
        
        for result in results:
            if result.get("memory_tier") == "ltm":
                memory_id = result.get("id")
                if memory_id and memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    unique_results.append(result)
        
        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return unique_results[:limit]
    
    def _build_temporal_filter(self, date_range: Optional[Dict], base_filter: Optional[Dict] = None) -> Optional[Dict]:
        """Build vectordb where filter for temporal constraints"""
        if not date_range:
            return base_filter
            
        and_clauses = []
        if "start" in date_range:
            try:
                start_ts = datetime.fromisoformat(date_range["start"].replace('Z', '+00:00')).timestamp()
                and_clauses.append({"timestamp_epoch": {"$gte": start_ts}})
            except Exception:
                and_clauses.append({"timestamp_epoch": {"$gte": date_range["start"]}})
        if "end" in date_range:
            try:
                end_ts = datetime.fromisoformat(date_range["end"].replace('Z', '+00:00')).timestamp()
                and_clauses.append({"timestamp_epoch": {"$lte": end_ts}})
            except Exception:
                and_clauses.append({"timestamp_epoch": {"$lte": date_range["end"]}})

        if base_filter:
            if "$and" in base_filter and isinstance(base_filter["$and"], list):
                merged = {"$and": base_filter["$and"] + and_clauses}
            else:
                merged = {"$and": and_clauses + [base_filter]}
            return merged
        else:
            if len(and_clauses) == 1:
                return and_clauses[0]
            return {"$and": and_clauses}

    def _parse_date_range(self, date_str: str) -> Optional[Dict[str, str]]:
        """Lightweight natural language/RFC3339 date range parser for core API."""
        import re
        if not date_str:
            return None
        s_original = date_str.strip()
        s_lower = date_str.lower().strip()
        now = datetime.now().astimezone()

        # RFC3339 single timestamp passthrough
        if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', s_original):
            return {"start": s_original, "end": s_original}

        if "yesterday" in s_lower:
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(microseconds=1)
            return {"start": start.isoformat(), "end": end.isoformat()}

        if "last week" in s_lower:
            # Last 7 full days excluding today
            end = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
            start = (end - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
            return {"start": start.isoformat(), "end": end.isoformat()}

        if "last month" in s_lower:
            # Last 30 full days excluding today
            end = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
            start = (end - timedelta(days=29)).replace(hour=0, minute=0, second=0, microsecond=0)
            return {"start": start.isoformat(), "end": end.isoformat()}

        if re.match(r'^\d{4}-\d{2}$', s_lower):
            year, month = s_lower.split('-')
            start = datetime(int(year), int(month), 1).astimezone()
            next_month = start + timedelta(days=32)
            end = next_month.replace(day=1) - timedelta(microseconds=1)
            return {"start": start.isoformat(), "end": end.isoformat()}

        if re.match(r'^\d{4}$', s_lower):
            start = datetime(int(s_lower), 1, 1).astimezone()
            end = datetime(int(s_lower), 12, 31, 23, 59, 59, 999999).astimezone()
            return {"start": start.isoformat(), "end": end.isoformat()}

        logger.warning(f"Could not parse date range: {date_str}")
        return None
    
    def _apply_temporal_weighting(self, results: List[Dict], temporal_weight: float) -> List[Dict]:
        """Apply temporal weighting to search results"""
        from datetime import datetime
        import time
        
        current_time = time.time()
        
        for result in results:
            semantic_score = result.get("score", 0.0)            
            recency_score = 0.0
            timestamp = result.get("timestamp")
            
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        # Try to parse as RFC3339 (ISO 8601)
                        parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        memory_time = parsed_time.timestamp()
                        
                        # Calculate days ago (more recent = higher score)
                        days_ago = (current_time - memory_time) / (24 * 3600)
                        
                        # Exponential decay: score = e^(-days/7) (half-life ~5 days)
                        recency_score = max(0.0, min(1.0, 2.71828 ** (-days_ago / 7.0)))
                        
                except (ValueError, TypeError):
                    # Fallback: if timestamp can't be parsed, assume medium recency
                    recency_score = 0.5
            
            # Combine semantic and temporal scores = form a composite score to rank results
            combined_score = (semantic_score * (1 - temporal_weight) + 
                            recency_score * temporal_weight)
            
            result["score"] = combined_score
            result["semantic_score"] = semantic_score
            result["recency_score"] = recency_score
            result["temporal_weighted"] = True
        
        return results
    
    def _collection_aware_search(self, query: str, limit: int, memory_source: str, 
                               where_filter: Optional[Dict], user_id: Optional[str], 
                               session_id: Optional[str], context: Optional[str] = None) -> List[Dict]:
        """Enhanced search using collection discovery and intelligent query transformation"""
        logger.info(f"Collection-aware search for: '{query}'")
        
        top_collections = self.collection_manager.discover_relevant_collections(query)
        
        if not top_collections:
            logger.info(f"No collections found for query: '{query}'")
            return []
        
        all_results = []
        relevant_collections_count = 0
        
        for collection_name, collection_similarity in top_collections:
            # Smart query transformation: check relevance first, then enhance if relevant
            is_relevant, transformed_query = self.collection_manager.transform_query_for_collection(
                query, collection_name, context or ""
            )
            
            # Always search the collection, but use enhanced query if relevant, original if not
            search_query = transformed_query if is_relevant else query
            relevant_collections_count += 1
            
            collection_results = self._search_collection_memories(
                search_query, collection_name, limit, memory_source, 
                where_filter, user_id, session_id
            )
            
            # Log search strategy
            if is_relevant:
                logger.info(f"Found {len(collection_results)} results in collection '{collection_name}' (enhanced query)")
            else:
                logger.info(f"Found {len(collection_results)} results in collection '{collection_name}' (original query)")
            
            # Add collection scores and metadata
            for result in collection_results:
                result["collection_similarity"] = collection_similarity
                result["collection_name"] = collection_name
                result["query_enhanced"] = is_relevant and (transformed_query != query)
                result["relevance_checked"] = True
                result["search_strategy"] = "enhanced" if is_relevant else "original"
                
            all_results.extend(collection_results)
        
        logger.info(f"Collection search summary: {len(all_results)} total results from {relevant_collections_count}/{len(top_collections)} relevant collections")
        return all_results
    
    def _search_collection_memories(self, query: str, collection_name: str, limit: int,
                                  memory_source: str, where_filter: Optional[Dict],
                                  user_id: Optional[str], session_id: Optional[str]) -> List[Dict]:
        """Search memories belonging to a specific collection"""
        results = []
        
        if memory_source in ["stm", "all"]:
            stm_results = self.stm.search(query, limit, where_filter=where_filter,
                                        user_id=user_id, session_id=session_id)
            # STM memories don't have categories (processed by LightPreprocessor)
            # Include all STM results in collection search since they're recent/relevant
            for result in stm_results:
                result["memory_tier"] = "stm"
                result["collection_source"] = "stm_included"  # Mark as STM inclusion
            results.extend(stm_results)
        
        if memory_source in ["ltm", "all"]:
            ltm_results = self.ltm.search(query, limit, where_filter=where_filter,
                                        user_id=user_id, session_id=session_id)
            # LTM memories have categories - filter by collection membership
            filtered_ltm = [r for r in ltm_results
                          if self.collection_manager.memory_belongs_to_collection(r, collection_name)]
            for result in filtered_ltm:
                result["memory_tier"] = "ltm"
                result["collection_source"] = "ltm_filtered"  # Mark as LTM filtering
            results.extend(filtered_ltm)
        
        return results
    
    def _merge_hybrid_results(self, global_results: List[Dict], collection_results: List[Dict], 
                            limit: int, temporal_weight: float = 0.0) -> List[Dict]:
        """Merge global and collection-aware results with composite scoring and temporal weighting"""
        # Add composite scores to collection results
        for result in collection_results:
            collection_sim = result.get("collection_similarity", 0)
            item_score = result.get("score", 0)
            result["composite_score"] = (COLLECTION_SIMILARITY_WEIGHT * collection_sim + 
                                       ITEM_SIMILARITY_WEIGHT * item_score)
        
        # Combine all results
        all_results = global_results + collection_results
        
        # Deduplicate by memory ID, preserving collection metadata
        seen_ids = {}
        for result in all_results:
            memory_id = result.get("id")
            if not memory_id:
                continue
                
            score = result.get("composite_score", result.get("score", 0))
            has_collection_info = result.get("collection_name") is not None
            
            if memory_id not in seen_ids:
                seen_ids[memory_id] = {
                    "result": result,
                    "score": score
                }
            else:
                existing = seen_ids[memory_id]
                existing_has_collection = existing["result"].get("collection_name") is not None
                
                # Prefer collection results over global results, or higher score if both have same type
                should_replace = False
                if has_collection_info and not existing_has_collection:
                    should_replace = True  # Always prefer collection results
                elif has_collection_info == existing_has_collection and score > existing["score"]:
                    should_replace = True  # Same type, prefer higher score
                
                if should_replace:
                    seen_ids[memory_id] = {
                        "result": result,
                        "score": score
                    }
        
        unique_results = [item["result"] for item in seen_ids.values()]
        
        # Apply temporal weighting if requested (affects both global and collection results)
        if temporal_weight > 0.0:
            unique_results = self._apply_temporal_weighting(unique_results, temporal_weight)
        
        # Sort by final scores (composite or temporal-weighted)
        unique_results.sort(key=lambda x: x.get("composite_score", x.get("score", 0)), reverse=True)
        
        return unique_results[:limit]
    
    def search_memory(self, 
                     query: str, 
                     limit: int = DEFAULT_SEARCH_LIMIT, 
                     memory_source: str = "all",
                     where_filter: Dict = None,
                     apply_postprocessing: bool = True,
                     context: str = None,
                     user_id: Optional[str] = None,
                     session_id: Optional[str] = None,
                     temporal_weight: float = 0.0,
                     date_range: Optional[Dict] = None) -> List[Dict[str, Any]]:
        return self.search(query=query, limit=limit, memory_source=memory_source,
                         where_filter=where_filter, apply_postprocessing=apply_postprocessing,
                         context=context, user_id=user_id, session_id=session_id,
                         temporal_weight=temporal_weight, date_range=date_range)
    
    def search_agentic(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT, user_id: Optional[str] = None, 
                      session_id: Optional[str] = None) -> List[Dict]:
        return self.search(query=query, limit=limit, user_id=user_id, session_id=session_id)
    
    def search_filtered(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT, where_filter: Optional[Dict] = None,
                       user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict]:
        return self.search(query=query, limit=limit, where_filter=where_filter, user_id=user_id, session_id=session_id)
    
    def clear_stm(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Clear short-term memory for a specific user/session or all if none specified"""
        self.stm.clear(user_id, session_id)
    
    def clear_ltm(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Clear long-term memory for a specific user/session or all if none specified"""
        self.ltm.clear(user_id, session_id)
    

    def find_related_memories(self, query: str, k: int = DEFAULT_RELATED_MEMORIES_COUNT, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Tuple[str, List[str]]:
        """Find related memories using vectorDB retrieval"""
        if not self.memories:
            return "", []
            
        try:
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
        """Update a memory note."""
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
        """Delete a memory note by its ID."""
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

    def _get_embedding(self, content: str) -> List[float]:
        """Generate embeddings for text content using shared embedding manager"""
        return self.embedding_manager.get_embedding(content)
    

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note and determine if it should evolve."""
        if not self.memories:
            return False, note
        if len(note.content) > LARGE_CONTENT_THRESHOLD:
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
                                                logger.debug(f"Skipping connection to memory {conn_id_str} due to user/session mismatch")
                                        else:
                                            logger.debug(f"LLM suggested connection to non-existent memory ID: {conn_id_str} - skipping")

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
                                            'strength': max(MIN_CONNECTION_STRENGTH, reciprocal_strength),
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
        """Resolve a potential memory ID from various formats."""
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
        """Merge two memory notes based on the specified strategy."""
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
            
        elif strategy in ["augment", "combine"]:
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
        """Combine metadata from two memory notes."""
        combined_keywords = list(set(target_note.keywords + source_note.keywords))
        target_note.keywords = combined_keywords
        
        combined_tags = list(set(target_note.tags + source_note.tags))
        target_note.tags = combined_tags
        
        try:
            def parse_timestamp(ts):
                """Parse RFC3339 timestamp with fallback for legacy formats"""
                try:
                    # Primary: RFC3339/ISO 8601 format
                    return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    # Fallback: try legacy formats for backward compatibility
                    legacy_formats = [
                        "%Y%m%d%H%M",               # Legacy format (YYYYMMDDhhmm)
                        "%Y-%m-%dT%H:%M:%S.%f",     # ISO format with microseconds
                        "%Y-%m-%dT%H:%M:%S"         # ISO format without microseconds
                    ]
                    
                    for fmt in legacy_formats:
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
        """Merge links from source note into target note."""
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
                    
                    self.ltm._get_collection(user_id, session_id).add_document(
                        document=linked_note.content, 
                        metadata=metadata, 
                        doc_id=link_id
                    )
                    
                    self.stm.add(
                        link_id, 
                        linked_note.content, 
                        metadata, 
                        user_id, 
                        session_id
                    )
                    
                    self.memories[link_id] = linked_note

    def _is_valid_uuid(self, val: str) -> bool:
        """Check if a string is a valid UUID"""
        if not isinstance(val, str):
            return False
            
        try:
            uuid_obj = uuid.UUID(val)
            return str(uuid_obj) == val
        except (ValueError, AttributeError):
            return False

    def _get_reciprocal_relationship(self, relationship_type: str) -> str:
        """Get the reciprocal relationship type."""
        # TODO: add/reduce relationship types (hardcoded for now)
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
