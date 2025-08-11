import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from cortex.constants import (
    COLLECTION_CREATION_THRESHOLD, TOP_COLLECTIONS_FOR_SEARCH,
    COLLECTION_SIMILARITY_WEIGHT, ITEM_SIMILARITY_WEIGHT,
    MIN_UPDATE_INTERVAL, FORCE_UPDATE_INTERVAL,
    SIGNIFICANT_GROWTH_RATIO, SIGNIFICANT_GROWTH_ABSOLUTE, METADATA_UPDATE_THRESHOLD,
    COLLECTION_METADATA_SAMPLE_SIZE, COLLECTION_CONTENT_SAMPLE_SIZE,
    MIN_MEMORIES_FOR_COLLECTION_CREATION
)

logger = logging.getLogger(__name__)

class CollectionManager:
    """Manages smart collections for domain-specific memory organization
    
    Design Note: collection_name IS the category_pattern.
    A collection named 'work.programming' contains all memories with categories 
    that start with 'work.programming' (e.g., 'work.programming.python').
    """
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.collections = {}
        self.category_counts = {}
        self.collection_embeddings = {}
        self.last_update_times = {}  # Track last update time per collection
    
    def update_category_counts(self, category: str):
        """Update category counts and trigger collection creation if needed"""
        if not category:
            return
            
        patterns = self._extract_category_patterns(category)
        for pattern in patterns:
            old_count = self.category_counts.get(pattern, 0)
            new_count = old_count + 1
            self.category_counts[pattern] = new_count
            
            logger.debug(f"Category count updated: '{pattern}' ({old_count} → {new_count})")
            
            if new_count == COLLECTION_CREATION_THRESHOLD - 2:
                logger.info(f"Category '{pattern}' approaching collection threshold ({new_count}/{COLLECTION_CREATION_THRESHOLD})")
            elif new_count == COLLECTION_CREATION_THRESHOLD:
                logger.info(f"Category '{pattern}' reached collection creation threshold ({new_count})")
                
            if self._should_create_collection(pattern):
                self._create_collection(pattern)
    
    def get_existing_categories_context(self) -> str:
        """Return formatted context of existing categories for LLM prompts"""
        if not self.category_counts:
            return "No existing categories yet."
            
        domain_groups = {}
        for category, count in self.category_counts.items():
            if '.' in category and len(category.split('.')) <= 4:
                domain = category.split('.')[0]
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append((category, count))
        
        if not domain_groups:
            return "No meaningful category patterns yet."
            
        context_lines = []
        for domain in sorted(domain_groups.keys()):
            categories = sorted(domain_groups[domain], key=lambda x: x[1], reverse=True)
            domain_categories = [f"{cat} ({count})" for cat, count in categories[:8]]
            context_lines.append(f"{domain}: {', '.join(domain_categories)}")
        
        substantial_patterns = sum(1 for counts in domain_groups.values() for _, count in counts if count >= COLLECTION_CREATION_THRESHOLD)
        created_collections = len(self.collections)
        
        header = "Existing category patterns IN MEMORY CURRENTLY by domain:\n" + '\n'.join(context_lines)
        
        if substantial_patterns > 0:
            header += f"\n\nNote: {substantial_patterns} patterns have 15+ memories (collection-ready), {created_collections} collections created."
            header += "\nFAVOR growing existing substantial patterns over creating narrow sub-categories."
        
        return header
    
    def discover_relevant_collections(self, query: str) -> List[Tuple[str, float]]:
        """Find top relevant collections for the query"""
        if not self.collection_embeddings:
            logger.debug(f"No collections available for query: '{query}'")
            return []
            
        logger.debug(f"Discovering relevant collections for query: '{query}'")
        
        query_embedding = self.memory_system._get_embedding(query)
        collection_scores = []
        
        for collection_name, collection_embedding in self.collection_embeddings.items():
            similarity = np.dot(query_embedding, collection_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(collection_embedding)
            )
            collection_scores.append((collection_name, float(similarity)))
        
        collection_scores.sort(key=lambda x: x[1], reverse=True)
        top_collections = collection_scores[:TOP_COLLECTIONS_FOR_SEARCH]
        
        if top_collections:
            logger.info(f"Top {len(top_collections)} relevant collections for '{query}':")
            for i, (collection_name, score) in enumerate(top_collections, 1):
                logger.info(f"{i}. '{collection_name}' (similarity: {score:.3f})")
        else:
            logger.info(f"No relevant collections found for query: '{query}'")
            
        return top_collections
    
    def transform_query_for_collection(self, original_query: str, collection_name: str, context: str = "") -> tuple:
        """Check query relevance and transform if relevant. Returns (is_relevant, enhanced_query)"""
        if collection_name not in self.collections:
            return False, original_query
            
        query_helper = self.collections[collection_name].get('query_helper', '')
        if not query_helper:
            return False, original_query
            
        context_part = f"\nContext: {context}" if context else ""
        
        prompt = f"""Using this collection guide: {query_helper}

Query: {original_query}\n{context_part}

Determine if this query is relevant to this collection and write an enhanced rephrased version ONLY if relevant.
The enhanced query will be used to search through the collection later.

**Relevance Decision Guidelines:**
- Query is RELEVANT if it asks about topics, problems, or concepts that this collection contains
- Query is NOT RELEVANT if it asks about completely different domains or unrelated topics
- Consider both the explicit query terms AND the context provided

**Enhancement Guidelines (only if relevant):**
- Add domain-specific terms, frameworks, tools or whatever context mentioned in the collection guide
- Keep the original intent and meaning intact
- Make it more specific to help find/match the right memories in this collection
- Don't change the core question being asked

Respond in valid JSON format:
{{
    "relevant": true/false,
    "enhanced_query": "enhanced version if relevant, otherwise original query",
    "reasoning": "brief explanation of relevance decision"
}}"""

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                prompt, 
                response_format={"type": "json_schema", "json_schema": {
                    "name": "query_transformation",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "relevant": {"type": "boolean"},
                            "enhanced_query": {"type": "string"},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["relevant", "enhanced_query", "reasoning"],
                        "additionalProperties": False
                    }
                }}
            )
            result = json.loads(response.strip())
            
            is_relevant = result.get('relevant', False)
            enhanced_query = result.get('enhanced_query', original_query)
            reasoning = result.get('reasoning', '')
            
            logger.info(f"Query transformation for '{collection_name}': {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")
            
            if reasoning:
                logger.info(f"Reasoning: {reasoning}")
                
            if is_relevant and enhanced_query != original_query:
                logger.info(f"Enhanced query: '{original_query}' → '{enhanced_query}'")
            elif is_relevant:
                logger.debug(f"Query kept unchanged: '{original_query}'")
            
            return is_relevant, enhanced_query if enhanced_query else original_query
            
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON response in query transformation: {e}")
            return False, original_query
        except Exception as e:
            logger.warning(f"Error in smart query transformation: {e}")
            return False, original_query
    
    def memory_belongs_to_collection(self, memory_result: Dict, collection_name: str) -> bool:
        """Check if a memory belongs to a specific collection based on category"""
        memory_category = memory_result.get("category", "")
        return memory_category and memory_category.startswith(collection_name)
    
    def update_collection_metadata(self, category: str):
        """Update existing collection metadata if this memory belongs to one (with smart conditions)"""
        import time
        
        for collection_name in self.collections:
            if category.startswith(collection_name):
                current_time = time.time()
                last_update = self.last_update_times.get(collection_name, 0)
                time_since_update = current_time - last_update
                
                should_check = (
                    time_since_update >= MIN_UPDATE_INTERVAL and
                    (self._has_significant_growth(collection_name) or 
                     time_since_update >= FORCE_UPDATE_INTERVAL)
                )
                
                if should_check:
                    logger.info(f"Checking metadata update for collection '{collection_name}' (last update: {time_since_update/3600:.1f}h ago)")
                    updated = self._update_collection_metadata(collection_name)
                    if updated:
                        self.last_update_times[collection_name] = current_time
                        logger.info(f"Collection '{collection_name}' metadata updated successfully")
                    else:
                        logger.debug(f"Collection '{collection_name}' metadata unchanged (insufficient growth)")
                else:
                    logger.debug(f"Collection '{collection_name}' metadata update skipped (conditions not met)")
                break
    
    def _extract_category_patterns(self, category: str) -> List[str]:
        """Extract hierarchical patterns from category"""
        if not category:
            return []
        
        parts = category.split('.')
        patterns = []
        
        if len(parts) >= 1:
            patterns.append(parts[0])
        
        for i in range(1, len(parts)):
            patterns.append('.'.join(parts[:i+1]))
            
        return patterns
    
    def _should_create_collection(self, category_pattern: str) -> bool:
        """Check if a collection should be created for this category pattern"""
        pattern_count = self.category_counts.get(category_pattern, 0)
        
        if pattern_count < COLLECTION_CREATION_THRESHOLD:
            return False
            
        if self._would_fragment_existing_collection(category_pattern):
            logger.info(f"Preventing fragmentation: '{category_pattern}' would fragment substantial parent collection")
            return False
            
        return True
    
    def _would_fragment_existing_collection(self, new_pattern: str) -> bool:
        """Check if creating this pattern would unnecessarily fragment an existing substantial collection"""
        parts = new_pattern.split('.')
        
        for i in range(len(parts) - 1, 0, -1):
            parent_pattern = '.'.join(parts[:i])
            parent_count = self.category_counts.get(parent_pattern, 0)
            
            if parent_count >= COLLECTION_CREATION_THRESHOLD * 3:
                new_count = self.category_counts.get(new_pattern, 0)
                
                if new_count < parent_count * 0.3:
                    logger.debug(f"Anti-fragmentation: '{new_pattern}' ({new_count}) would fragment '{parent_pattern}' ({parent_count})")
                    return True
                    
        return False
    
    def _create_collection(self, category_pattern: str):
        """Create a new collection for the given category pattern"""
        if category_pattern in self.collections:
            logger.debug(f"Collection '{category_pattern}' already exists, skipping creation")
            return
            
        logger.info(f"Creating new collection for pattern: '{category_pattern}'")
        
        matching_memories = self._find_memories_by_pattern(category_pattern)
        
        if len(matching_memories) < COLLECTION_CREATION_THRESHOLD:
            logger.warning(f"Not enough memories ({len(matching_memories)}) for collection '{category_pattern}' (need {COLLECTION_CREATION_THRESHOLD})")
            return
            
        contents_sample = [m.content for m in matching_memories[:COLLECTION_METADATA_SAMPLE_SIZE]]
        logger.info(f"Generating metadata for '{category_pattern}' using {len(contents_sample)} sample memories")
        
        description, query_helper = self._generate_collection_metadata(category_pattern, contents_sample)
        
        collection_info = {
            'name': category_pattern,
            'description': description,
            'query_helper': query_helper,
            'memory_count': len(matching_memories),
            'created_at': datetime.now().isoformat()
        }
        
        self.collections[category_pattern] = collection_info
        
        collection_text = f"{category_pattern} {description}"
        self.collection_embeddings[category_pattern] = self.memory_system._get_embedding(collection_text)
        
        logger.info(f"Created collection '{category_pattern}' with {len(matching_memories)} memories")
        logger.info(f"Description: {description[:100]}{'...' if len(description) > 100 else ''}")
        logger.debug(f"Query helper: {query_helper[:150]}{'...' if len(query_helper) > 150 else ''}")
    
    def _generate_collection_metadata(self, category_pattern: str, content_samples: List[str]) -> Tuple[str, str]:
        """Generate description and query_helper for a collection"""
        if not content_samples:
            return f"Collection for {category_pattern}", ""
            
        sample_text = "\n".join(content_samples[:COLLECTION_CONTENT_SAMPLE_SIZE])
        
        prompt = f"""Analyze this collection of memories with category pattern "{category_pattern}":

Sample contents:
{sample_text}

Generate collection metadata:
1. Description: A concise summary of what this collection contains (2-3 sentences)
2. Query Helper: A detailed description of what queries are relevant to this collection and how to enhance them

Format as JSON:
{{
    "description": "Brief description of collection contents and domain",
    "query_helper": "This collection contains memories about [domain]. A query is relevant if it asks about [topics]. Enhance by adding [key terms]."
}}

Examples of good query_helper:
- Programming: "This collection contains Python programming memories. A query is relevant if it asks about Python coding, web frameworks, or debugging. Enhance by adding Python, Django, Flask, debugging terms."
- Fitness: "This collection contains fitness and exercise memories. A query is relevant if it asks about workouts, nutrition, or training. Enhance by adding exercise, training, fitness terms."
"""

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                prompt,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "collection_metadata",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "query_helper": {"type": "string"}
                        },
                        "required": ["description", "query_helper"],
                        "additionalProperties": False
                    }
                }}
            )
            metadata = json.loads(response)
            return metadata.get('description', f"Collection for {category_pattern}"), metadata.get('query_helper', '')
        except Exception as e:
            logger.warning(f"Error generating collection metadata: {e}")
            return f"Collection for {category_pattern}", ""
    
    def _find_memories_by_pattern(self, pattern: str) -> List:
        """Find all memories that match a category pattern (used for both category patterns and collection names)"""
        matching_memories = []
        seen_ids = set()
        
        for memory in self.memory_system.memories.values():
            memory_category = getattr(memory, 'category', '')
            if memory_category and memory_category.startswith(pattern):
                memory_id = getattr(memory, 'id', None)
                if memory_id and memory_id not in seen_ids:
                    matching_memories.append(memory)
                    seen_ids.add(memory_id)
        
        pattern_count = self.category_counts.get(pattern, 0)
        
        if len(matching_memories) >= MIN_MEMORIES_FOR_COLLECTION_CREATION or pattern_count >= COLLECTION_CREATION_THRESHOLD:
            return matching_memories
        
        return matching_memories
    
    def _has_significant_growth(self, collection_name: str) -> bool:
        """Check if collection has grown significantly since last update"""
        if collection_name not in self.collections:
            return False
            
        current_count = self.collections[collection_name]['memory_count']
        pattern_count = self.category_counts.get(collection_name, 0)
        
        return (pattern_count >= current_count * SIGNIFICANT_GROWTH_RATIO) or (pattern_count >= current_count + SIGNIFICANT_GROWTH_ABSOLUTE)
    
    def _update_collection_metadata(self, collection_name: str) -> bool:
        """Update collection description and query helper as it grows. Returns True if updated."""
        if collection_name not in self.collections:
            logger.warning(f"Cannot update metadata: collection '{collection_name}' not found")
            return False
            
        matching_memories = self._find_memories_by_pattern(collection_name)
        
        current_count = self.collections[collection_name]['memory_count']
        new_count = len(matching_memories)
        growth_ratio = new_count / current_count if current_count > 0 else float('inf')
        
        logger.debug(f"Metadata update check for '{collection_name}': {current_count} → {new_count} memories ({growth_ratio:.2f}x growth)")
        
        if new_count >= current_count * METADATA_UPDATE_THRESHOLD:
            logger.info(f"Updating metadata for '{collection_name}' ({current_count} → {new_count} memories, {growth_ratio:.2f}x growth)")
            
            contents_sample = [m.content for m in matching_memories[-COLLECTION_METADATA_SAMPLE_SIZE:]]
            description, query_helper = self._generate_collection_metadata(collection_name, contents_sample)
            
            old_description = self.collections[collection_name]['description']
            
            self.collections[collection_name]['description'] = description
            self.collections[collection_name]['query_helper'] = query_helper
            self.collections[collection_name]['memory_count'] = new_count
            
            collection_text = f"{collection_name} {description}"
            self.collection_embeddings[collection_name] = self.memory_system._get_embedding(collection_text)
            
            logger.info(f"Updated metadata for '{collection_name}' with {new_count} memories")
            if description != old_description:
                logger.info(f"New description: {description[:100]}{'...' if len(description) > 100 else ''}")
            
            return True
        
        logger.debug(f"No metadata update needed for '{collection_name}' (growth {growth_ratio:.2f}x < {METADATA_UPDATE_THRESHOLD}x threshold)")
        return False 