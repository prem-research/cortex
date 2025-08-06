from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import logging
import os
import time
import argparse
from cortex.memory_system import AgenticMemorySystem
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv(".env")


logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for detailed logging
logger = logging.getLogger(__name__)

def _parse_date_range(date_str: str) -> Optional[Dict[str, str]]:
    """Parse natural language date range into RFC3339 timestamp format"""
    from datetime import datetime, timedelta
    import re
    
    if not date_str:
        return None
    
    original_str = date_str.strip()
    date_str_lower = date_str.lower().strip()
    current_time = datetime.now().astimezone()
    
    # RFC3339 format (direct passthrough) - check first before lowercasing
    if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', original_str):
        return {"start": original_str, "end": original_str}
    
    # Use lowercase for natural language patterns
    date_str = date_str_lower
    
    # Relative date patterns
    if "yesterday" in date_str:
        start = current_time - timedelta(days=1)
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1) - timedelta(microseconds=1)
        return {
            "start": start.isoformat(),
            "end": end.isoformat()
        }
    
    if "last week" in date_str:
        start = current_time - timedelta(weeks=1)
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = current_time
        return {
            "start": start.isoformat(),
            "end": end.isoformat()
        }
    
    if "last month" in date_str:
        start = current_time - timedelta(days=30)
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = current_time
        return {
            "start": start.isoformat(),
            "end": end.isoformat()
        }
    
    # Specific date formats
    # YYYY-MM format
    if re.match(r'^\d{4}-\d{2}$', date_str):
        year, month = date_str.split('-')
        start = datetime(int(year), int(month), 1).astimezone()
        # Calculate end of month
        next_month = start + timedelta(days=32)
        end = next_month.replace(day=1) - timedelta(microseconds=1)
        return {
            "start": start.isoformat(),
            "end": end.isoformat()
        }
    
    # YYYY format
    if re.match(r'^\d{4}$', date_str):
        start = datetime(int(date_str), 1, 1).astimezone()
        end = datetime(int(date_str), 12, 31, 23, 59, 59, 999999).astimezone()
        return {
            "start": start.isoformat(),
            "end": end.isoformat()
        }
    

    
    logger.warning(f"Could not parse date range: {date_str}")
    return None

memory_system = AgenticMemorySystem(
    stm_capacity=5,
    enable_smart_collections=True,  # Enable for better organization of diverse content
    enable_background_processing=False  # Synchronous processing to see collection creation before queries
)  

# for text splitting (before storing into memory)
CHUNK_SIZE = 6000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

class MemoryInput(BaseModel):
    """Input model for storing a memory"""
    content: str = Field(..., description="The content of the memory to store")
    context: Optional[str] = Field(None, description="Optional context for the memory")
    tags: Optional[List[str]] = Field(None, description="Optional tags for the memory")
    timestamp: Optional[str] = Field(None, description="Optional custom timestamp (RFC3339 format: 2023-01-01T12:00:00+00:00)")
    user_id: Optional[str] = Field(None, description="Optional user identifier for memory segregation")
    session_id: Optional[str] = Field(None, description="Optional session identifier for memory segregation")

class MemoryOutput(BaseModel):
    """Output model for a memory"""
    id: str
    content: str
    context: Optional[str] = None
    tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    timestamp: Optional[str] = None
    score: Optional[float] = None
    is_linked: bool = False
    memory_tier: Optional[str] = None
    collection_name: Optional[str] = None
    category: Optional[str] = None
    composite_score: Optional[float] = None
    relationship_type: Optional[str] = None
    relationship_strength: Optional[float] = None
    relationship_reason: Optional[str] = None

class StoreResponse(BaseModel):
    """Response model for the store endpoint"""
    id: str
    success: bool
    message: str

class RetrieveResponse(BaseModel):
    """Response model for the retrieve endpoint"""
    memories: List[MemoryOutput]
    count: int

def store_memory(memory: MemoryInput) -> StoreResponse:
    """
    Store a new memory in the system.
    
    The memory content is processed automatically to extract keywords,
    establish relationships with existing memories, and handle potential
    merging with similar content.
    
    Returns:
        StoreResponse: Contains the ID of the stored memory, success status, and message
    """
    # Extract data from the request
    content = memory.content
    context = memory.context
    tags = memory.tags or []
    timestamp = memory.timestamp
    user_id = memory.user_id
    session_id = memory.session_id
    
    # Prepare keyword arguments for memory storage
    # add_note automatically analyzes content through DeepPreprocessor
    kwargs = {}
    if context:
        kwargs["context"] = context
    if tags:
        kwargs["tags"] = tags
        
    # Store the memory and get its ID
    memory_id = memory_system.add_note(
        content, 
        time=timestamp, 
        user_id=user_id, 
        session_id=session_id, 
        **kwargs
    )
    
    return StoreResponse(
        id=memory_id,
        success=True,
        message="Memory stored successfully"
    )

def retrieve_memories(
    q: str,
    limit: int = 5,
    memory_source: str = "all",  # Options: "stm", "ltm", "all"
    context: Optional[str] = None,
    tags: Optional[str] = None,
    exclude_content: bool = False,
    include_links: bool = True,
    apply_postprocessing: bool = True,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    temporal_weight: Optional[float] = None,
    date_range: Optional[str] = None
) -> RetrieveResponse:
    """
    Retrieve memories related to the provided query.
    
    The system uses semantic search to find the most relevant memories.
    Optional filters can be applied for context and tags.
    
    Args:
        q: Search query to find related memories
        limit: Maximum number of memories to return
        memory_source: Which memory tiers to search ("stm", "ltm", "all")
        context: Optional context for filtering and relevance
        tags: Optional comma-separated tags for filtering
        exclude_content: Whether to exclude content in results
        include_links: Whether to include linked memories
        apply_postprocessing: Whether to apply post-retrieval processing
        user_id: Optional user identifier for memory segregation
        session_id: Optional session identifier for memory segregation
        temporal_weight: Optional temporal weighting (0.0=semantic only, 1.0=recency only, None=auto-detect)
        date_range: Optional date range filter ("last week", "2023-01", "yesterday", RFC3339 timestamps)
        
    Returns:
        RetrieveResponse: Contains list of memories ordered by relevance and count
    """
    # Parse tags if provided
    tag_list = _parse_tags(tags)
            
    # Create filter based on context and tags
    # where_filter = _create_filter(context, tag_list)
    where_filter = {}
    
    # Parse date range if provided
    parsed_date_range = None
    if date_range:
        parsed_date_range = _parse_date_range(date_range)
    
    # Auto-detect temporal queries if temporal_weight not specified
    if temporal_weight is None:
        temporal_keywords = ["last", "recent", "latest", "yesterday", "today", "this week", "past", "ago"]
        if any(keyword in q.lower() for keyword in temporal_keywords) or date_range:
            temporal_weight = 0.7  # Heavy temporal weighting for temporal queries
        else:
            temporal_weight = 0.0  # Pure semantic search for non-temporal queries
    
    # Search memories in specified tiers
    results = memory_system.search_memory(
        query=q,
        limit=limit,
        memory_source=memory_source,
        where_filter=where_filter,
        apply_postprocessing=apply_postprocessing,
        context=context,
        user_id=user_id,
        session_id=session_id,
        temporal_weight=temporal_weight,
        date_range=parsed_date_range
    )
    
    # Format the results
    memories = []
    seen_ids = set()
    
    # Process direct results
    for result in results:
        memory_id = result.get("id", "")
        if not memory_id or memory_id in seen_ids:
            continue
            
        seen_ids.add(memory_id)
        
        memory_output = {
            "id": memory_id,
            "context": result.get("context", ""),
            "tags": result.get("tags", []),
            "timestamp": str(result.get("timestamp", "")),  # Convert timestamp to string
            "score": result.get("score"),
            "is_linked": False,
            "memory_tier": result.get("memory_tier", "unknown"),
            "collection_name": result.get("collection_name", "No Collection"),
            "category": result.get("category", "Uncategorized"),
            "composite_score": result.get("composite_score")
        }
        
        if not exclude_content:
            memory_output["content"] = result.get("content", "")
            
        memories.append(memory_output)
    
    # Process linked memories if requested
    if include_links:
        linked_memories = _process_linked_memories(
            results, 
            seen_ids, 
            exclude_content,
            user_id,
            session_id
        )
        memories.extend(linked_memories)
    
    return RetrieveResponse(
        memories=memories,
        count=len(memories)
    )

def clear_short_term_memory(user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Clear the short-term memory for a user/session or all if not specified.
    
    Args:
        user_id: Optional user identifier
        session_id: Optional session identifier
        
    Returns:
        Dict with success status and message
    """
    memory_system.clear_stm(user_id, session_id)
    
    scope = "all memories"
    if user_id and session_id:
        scope = f"user '{user_id}' session '{session_id}'"
    elif user_id:
        scope = f"user '{user_id}'"
    elif session_id:
        scope = f"session '{session_id}'"
        
    return {
        "success": True,
        "message": f"Short-term memory cleared for {scope}"
    }

def _parse_tags(tags: Optional[str]) -> Optional[List[str]]:
    if not tags:
        return None
    try:
        return tags.split(",")
    except:
        return None

def _create_filter(context: Optional[str], tags: Optional[List[str]]) -> Dict[str, Any]:
    where_filter = {}
    if context:
        where_filter["context"] = {"$contains": context}
    if tags:
        where_filter["tags"] = {"$contains": tags}
    return where_filter

def _process_linked_memories(
    results: List[Dict[str, Any]], 
    seen_ids: set, 
    exclude_content: bool,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> List[Dict[str, Any]]:

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
            
            linked_memory = memory_system.read(link_id, user_id, session_id)
            if not linked_memory:
                continue
            
            link_output = _create_linked_memory_output(
                linked_memory, 
                link_id, 
                exclude_content, 
                links
            )
            linked_memories.append(link_output)
    
    return linked_memories

def _create_linked_memory_output(
    linked_memory: Any, 
    link_id: str, 
    exclude_content: bool, 
    links: Dict[str, Any]
) -> Dict[str, Any]:

    # Get memory attributes safely
    link_output = {
        "id": link_id,
        "context": getattr(linked_memory, "context", ""),
        "tags": getattr(linked_memory, "tags", []),
        "timestamp": str(getattr(linked_memory, "timestamp", "")),  # Convert timestamp to string
        "score": None,
        "is_linked": True,
        "memory_tier": "ltm"  # Linked memories typically come from LTM
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
    
    return link_output

def compare_memory_sources(query: str, user_id: Optional[str] = None, session_id: Optional[str] = None):
    """
    Compare retrieval results from different memory sources with detailed collection analytics
    """
    print(f"\nQuery: '{query}'")
    
    # Capture collection state before search
    collection_snapshot = capture_collection_snapshot()
    
    # Search in STM only
    stm_results = retrieve_memories(
        q=query,
        limit=3,
        user_id=user_id, 
        session_id=session_id,
        memory_source="stm"
    )
    
    # Search in LTM only
    ltm_results = retrieve_memories(
        q=query, 
        limit=3,
        user_id=user_id, 
        session_id=session_id,
        memory_source="ltm"
    )
    
    # Perform detailed "all" search to capture collection-aware analytics
    all_results_detailed = retrieve_memories_with_collection_analytics(
        q=query,
        limit=6,
        user_id=user_id,
        session_id=session_id,
        context=None  # Could be enhanced with context
    )
    
    # Combine STM/LTM for compatibility comparison
    combined_memories = []
    seen_ids = set()
    
    # Add STM memories first
    for memory in stm_results.memories:
        combined_memories.append(memory)
        seen_ids.add(memory.id)
    
    # Add LTM memories that aren't already in the combined list
    for memory in ltm_results.memories:
        if memory.id not in seen_ids:
            combined_memories.append(memory)
            seen_ids.add(memory.id)
    
    # Sort the combined memories by score in descending order (highest scores first)
    combined_memories.sort(key=lambda x: (x.score is not None, x.score or float('-inf')), reverse=True)
    
    # Create a RetrieveResponse with the combined results
    all_results = RetrieveResponse(
        memories=combined_memories,
        count=len(combined_memories)
    )
    
    # Print results with collection info
    print(f"STM results: {stm_results.count}")
    for memory in stm_results.memories:
        collection_info = memory.collection_name if hasattr(memory, 'collection_name') else 'No Collection'
        print(f"- {memory.content[:100]}... (Tier: {memory.memory_tier}, Score: {memory.score}, Collection: {collection_info})")
    
    print(f"\nLTM results: {ltm_results.count}")
    for memory in ltm_results.memories:
        collection_info = memory.collection_name if hasattr(memory, 'collection_name') else 'No Collection'
        print(f"- {memory.content[:100]}... (Tier: {memory.memory_tier}, Score: {memory.score}, Collection: {collection_info})")
    
    print(f"\nCollection-Aware results: {all_results_detailed['count']}")
    for memory in all_results_detailed['memories']:
        collection_info = memory.get('collection_name', 'No Collection')
        composite_score = memory.get('composite_score', memory.get('score', 0))
        score_display = f"{composite_score:.3f}" if composite_score is not None else "None"
        print(f"- {memory.get('content', '')[:100]}... (Tier: {memory.get('memory_tier', 'unknown')}, Score: {score_display}, Collection: {collection_info})")
    
    # Show collection analytics if available
    if 'collection_analytics' in all_results_detailed:
        analytics = all_results_detailed['collection_analytics']
        print(f"\nCollection Analytics:")
        print(f"   Collections searched: {analytics.get('collections_searched', 0)}")
        print(f"   Query transformations: {analytics.get('query_transformations', 0)}")
        print(f"   Relevant collections: {analytics.get('relevant_collections', 0)}")
        
        if analytics.get('collection_details'):
            print(f"   Collection breakdown:")
            for collection_name, details in analytics['collection_details'].items():
                transformed = "✓" if details.get('query_transformed') else "✗"
                print(f"     • {collection_name}: {details.get('results_count', 0)} results, enhanced={transformed}")
    
    print("\n--------------------------------\n")
    
    return {
        "stm": stm_results.memories,
        "ltm": ltm_results.memories,
        "all": all_results.memories,
        "collection_aware": all_results_detailed,
        "collection_snapshot": collection_snapshot
    }

def retrieve_memories_with_collection_analytics(
    q: str,
    limit: int = 5,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhanced retrieve that captures detailed collection analytics
    """
    # Direct call to memory system to get detailed results
    results = memory_system.search_memory(
        query=q,
        limit=limit,
        memory_source="all",
        user_id=user_id,
        session_id=session_id,
        context=context
    )
    
    # Extract collection analytics from results
    collection_analytics = extract_collection_analytics(results)
    
    # Format memories with full details
    formatted_memories = []
    for result in results:
        memory_data = {
            "id": result.get("id", ""),
            "content": result.get("content", ""),
            "context": result.get("context", ""),
            "tags": result.get("tags", []),
            "keywords": result.get("keywords", []),
            "timestamp": str(result.get("timestamp", "")),
            "score": result.get("score"),
            "composite_score": result.get("composite_score"),
            "memory_tier": result.get("memory_tier", "unknown"),
            "collection_name": result.get("collection_name", "No Collection"),
            "collection_similarity": result.get("collection_similarity"),
            "category": result.get("category", "Uncategorized")
        }
        formatted_memories.append(memory_data)
    
    return {
        "memories": formatted_memories,
        "count": len(formatted_memories),
        "collection_analytics": collection_analytics,
        "query": q,
        "context": context
    }

def extract_collection_analytics(results: List[Dict]) -> Dict[str, Any]:
    """Extract detailed analytics from search results"""
    collections_used = set()
    query_transformations = 0
    relevant_collections = 0
    collection_details = {}
    
    for result in results:
        collection_name = result.get("collection_name")
        if collection_name and collection_name != "No Collection":
            collections_used.add(collection_name)
            
            if collection_name not in collection_details:
                collection_details[collection_name] = {
                    "results_count": 0,
                    "query_transformed": bool(result.get("composite_score")),  # Has composite score = was transformed
                    "avg_collection_similarity": 0,
                    "collection_similarity_sum": 0
                }
            
            details = collection_details[collection_name]
            details["results_count"] += 1
            
            if result.get("collection_similarity"):
                details["collection_similarity_sum"] += result.get("collection_similarity", 0)
                details["avg_collection_similarity"] = details["collection_similarity_sum"] / details["results_count"]
            
            if result.get("composite_score"):
                relevant_collections += 1
    
    # Count unique transformations
    query_transformations = sum(1 for details in collection_details.values() if details["query_transformed"])
    
    return {
        "collections_searched": len(collections_used),
        "query_transformations": query_transformations,
        "relevant_collections": len(set(result.get("collection_name") for result in results 
                                      if result.get("composite_score"))),
        "collection_details": collection_details,
        "total_results": len(results)
    }

def capture_collection_snapshot() -> Dict[str, Any]:
    """Capture current state of collections for analytics"""
    if not hasattr(memory_system, 'collection_manager') or not memory_system.collection_manager:
        return {"collections_enabled": False}
    
    cm = memory_system.collection_manager
    
    snapshot = {
        "collections_enabled": True,
        "total_collections": len(cm.collections),
        "collections": {},
        "category_stats": dict(cm.category_counts),
        "total_patterns": len(cm.category_counts),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Capture each collection's metadata
    for collection_name, info in cm.collections.items():
        snapshot["collections"][collection_name] = {
            "memory_count": info.get("memory_count", 0),
            "created_at": info.get("created_at", ""),
            "description": info.get("description", "")[:200],  # Truncate for storage
            "query_helper": info.get("query_helper", "")[:200],  # Truncate for storage
            "last_updated": info.get("last_updated", "")
        }
    
    return snapshot

def extract_segments_from_file(file_path):
    """
    Extract meaningful text segments from a file using RecursiveCharacterTextSplitter.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of text segments
    """
    logger.info(f"Loading data from: {file_path}")
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"Input file not found: {file_path}")
            return []
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if not content.strip():
            logger.warning(f"File {file_path} is empty")
            return []
            
        # Define the splitter based on Langchain's RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n\n\n", "\n\n\n", "\n\n", "\n", " ", ""],
        )
        
        # Initial split by major separator (if desired, or just use the recursive splitter directly)
        initial_pages = content.split("\n\n\n\n") 
        final_chunks = []
        logger.info(f"Initial split resulted in {len(initial_pages)} major sections.")

        for i, page in enumerate(initial_pages):
            if len(page) > CHUNK_SIZE:
                # Recursively split oversized pages
                sub_chunks = splitter.split_text(page)
                final_chunks.extend(sub_chunks)
                logger.debug(f"Section {i+1} was larger than chunk size, split into {len(sub_chunks)} sub-chunks.")
            elif page.strip(): # Add non-empty pages smaller than chunk size directly
                final_chunks.append(page)
            else:
                 logger.debug(f"Skipping empty section {i+1}.")

        logger.info(f"Total chunks after recursive splitting: {len(final_chunks)}")
        return final_chunks
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []

def save_memories_to_json(memories, filename):
    """Save memories to a JSON file"""
    # Extract relevant data for each memory
    serialized_memories = []
    
    for memory in memories:
        # Try different ways to get keywords (depending on memory object type)
        if hasattr(memory, "keywords"):
            keywords = memory.keywords
        elif isinstance(memory, dict) and "keywords" in memory:
            keywords = memory["keywords"]
        else:
            # Try to get from original memory object
            memory_id = getattr(memory, "id", None)
            if memory_id and memory_id in memory_system.memories:
                original_memory = memory_system.memories[memory_id]
                keywords = getattr(original_memory, "keywords", [])
            else:
                keywords = []
        
        # Build dictionary with all relevant fields
        mem_dict = {
            "id": memory.id,
            "content": memory.content,
            "context": memory.context,
            "tags": memory.tags,
            "keywords": keywords,
            "timestamp": memory.timestamp,
            "score": memory.score,
            "memory_tier": getattr(memory, "memory_tier", None)
        }
        serialized_memories.append(mem_dict)
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serialized_memories, f, indent=2)
    
    print(f"Saved {len(serialized_memories)} memories to {filename}")

def get_stm_memories():
    """Get all memories from STM"""
    # Use the user_memories dict in STM to get all memories
    memories = []
    
    for user_key in memory_system.stm.user_memories:
        store = memory_system.stm.user_memories[user_key]
        for memory_id, memory_data in store.items():
            memory_data["id"] = memory_id
            memory_data["memory_tier"] = "stm"
            # Convert to MemoryOutput for consistent format
            memory_out = MemoryOutput(**memory_data)
            memories.append(memory_out)
    
    return memories

def display_collection_summary():
    """Display summary of created collections"""
    if not hasattr(memory_system, 'collection_manager') or not memory_system.collection_manager:
        print("No Smart Collections enabled")
        return
        
    collections = memory_system.collection_manager.collections
    if not collections:
        print("No collections created yet")
        return
        
    print(f"\n{'='*60}")
    print(f"SMART COLLECTIONS SUMMARY ({len(collections)} collections)")
    print(f"{'='*60}")
    
    for i, (collection_name, info) in enumerate(collections.items(), 1):
        created_at = info.get('created_at', 'Unknown')
        memory_count = info.get('memory_count', 0)
        description = info.get('description', 'No description')
        
        print(f"\n{i}. Collection: '{collection_name}'")
        print(f"   Memories: {memory_count}")
        print(f"   Created: {created_at[:19] if created_at != 'Unknown' else created_at}")
        print(f"   Description: {description[:120]}{'...' if len(description) > 120 else ''}")
        
    category_counts = memory_system.collection_manager.category_counts
    print(f"\nCATEGORY STATISTICS:")
    print(f"   Total patterns tracked: {len(category_counts)}")
    
    # Show top 10 patterns by count
    sorted_patterns = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if sorted_patterns:
        print(f"   Top patterns:")
        for pattern, count in sorted_patterns:
            print(f"     • {pattern}: {count} memories")
    
    print(f"{'='*60}\n")

def get_ltm_memories(user_id=None, session_id=None):
    """
    Get memories from LTM by searching with a broad query
    """
    # We'll use a very generic query to get as many memories as possible
    results = memory_system.search_memory(
        query="",  # Empty query returns everything
        limit=100,  # Large limit to get most memories
        memory_source="ltm",
        user_id=user_id,
        session_id=session_id
    )
    
    memories = []
    for result in results:
        result["timestamp"] = str(result["timestamp"])
        
        # Ensure keywords are included if available
        if "keywords" not in result and memory_system.memories.get(result["id"]):
            mem_note = memory_system.memories.get(result["id"])
            result["keywords"] = getattr(mem_note, "keywords", [])
            
        # Convert to MemoryOutput for consistent format
        memory_out = MemoryOutput(**result)
        memories.append(memory_out)
    
    return memories

def load_memories_from_json(stm_json_path=None, ltm_json_path=None):
    """
    Load memories from pre-stored JSON files into the memory system.
    
    Args:
        stm_json_path: Path to the STM memories JSON file
        ltm_json_path: Path to the LTM memories JSON file
        
    Returns:
        Tuple of (stm_memories, ltm_memories) as MemoryOutput objects
    """
    stm_memories = []
    ltm_memories = []
    
    # Load STM memories if provided
    if stm_json_path and os.path.exists(stm_json_path):
        logger.info(f"Loading STM memories from {stm_json_path}")
        try:
            with open(stm_json_path, 'r', encoding='utf-8') as f:
                stm_data = json.load(f)
                
            # Convert JSON data to MemoryOutput objects
            for mem_data in stm_data:
                memory_out = MemoryOutput(**mem_data)
                stm_memories.append(memory_out)
                
                # Prepare metadata for STM
                content = mem_data["content"]
                user_id = mem_data.get("user_id")
                session_id = mem_data.get("session_id")
                
                # Create metadata dict for ShortTermMemory.add method
                metadata = {
                    "context": mem_data.get("context", ""),
                    "tags": mem_data.get("tags", []),
                    "keywords": mem_data.get("keywords", []),
                    "timestamp": mem_data.get("timestamp"),
                    "category": mem_data.get("category", "Uncategorized"),
                    "links": mem_data.get("links", {}),
                    "retrieval_count": mem_data.get("retrieval_count", 0),
                    "last_accessed": mem_data.get("last_accessed"),
                    "evolution_history": mem_data.get("evolution_history", []),
                    "user_id": user_id,
                    "session_id": session_id
                }
                
                # Calculate embedding for content
                # We need to process it with the light processor
                enhanced_metadata = memory_system.light_processor.process(content, metadata)
                
                # Add to STM using the correct method
                memory_system.stm.add(
                    mem_data["id"],
                    content,
                    enhanced_metadata,
                    user_id,
                    session_id
                )
                
            logger.info(f"Loaded {len(stm_memories)} memories into STM")
        except Exception as e:
            logger.error(f"Error loading STM memories: {e}")
    
    # Load LTM memories if provided
    if ltm_json_path and os.path.exists(ltm_json_path):
        logger.info(f"Loading LTM memories from {ltm_json_path}")
        try:
            with open(ltm_json_path, 'r', encoding='utf-8') as f:
                ltm_data = json.load(f)
                
            # Convert JSON data to MemoryOutput objects
            for mem_data in ltm_data:
                memory_out = MemoryOutput(**mem_data)
                ltm_memories.append(memory_out)
                
                # Add to LTM
                content = mem_data["content"]
                user_id = mem_data.get("user_id")
                session_id = mem_data.get("session_id")
                
                # Create metadata dict for LongTermMemory.add method
                metadata = {
                    "context": mem_data.get("context", ""),
                    "tags": mem_data.get("tags", []),
                    "keywords": mem_data.get("keywords", []),
                    "timestamp": mem_data.get("timestamp"),
                    "category": mem_data.get("category", "Uncategorized"),
                    "links": mem_data.get("links", {}),
                    "retrieval_count": mem_data.get("retrieval_count", 0),
                    "last_accessed": mem_data.get("last_accessed"),
                    "evolution_history": mem_data.get("evolution_history", []),
                    "user_id": user_id,
                    "session_id": session_id,
                    "id": mem_data["id"]  # Add ID to metadata for LTM
                }
                
                # Process with deep processor to get proper embeddings
                enhanced_metadata = memory_system.deep_processor.process(content, metadata)
                
                # Add to LTM
                memory_system.ltm.add(
                    mem_data["id"],
                    content,
                    enhanced_metadata,
                    user_id,
                    session_id
                )
                
                # Also add to the memory system's dictionary for backward compatibility
                note = memory_system._dict_to_memory_note(mem_data)
                memory_system.memories[mem_data["id"]] = note
                
            logger.info(f"Loaded {len(ltm_memories)} memories into LTM")
        except Exception as e:
            logger.error(f"Error loading LTM memories: {e}")
    

    
    return stm_memories, ltm_memories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the memory system with Lyra content')
    parser.add_argument('--stm-json', type=str, help='Path to pre-stored STM memories JSON file')
    parser.add_argument('--ltm-json', type=str, help='Path to pre-stored LTM memories JSON file')
    parser.add_argument('--input-file', type=str, default="examples/lyra/lyra.txt", help='Path to the input text file (default: ../lyra.txt)')
    parser.add_argument('--queries-file', type=str, default="examples/lyra/queries.json", help='Path to the queries file (default: examples/lyra/queries.json)')
    parser.add_argument('--output-dir', type=str, default="examples/lyra", help='Path to the output directory (default: examples/lyra)')
    parser.add_argument('--skip-storage', action='store_true', help='Skip storing segments and go directly to queries')
    args = parser.parse_args()
    
    # Check if we should load from JSON files (not needed if --skip-storage is used, not required for first run)
    stm_memories = []
    ltm_memories = []
    if args.stm_json or args.ltm_json:
        stm_memories, ltm_memories = load_memories_from_json(args.stm_json, args.ltm_json)
        print(f"Loaded {len(stm_memories)} STM memories and {len(ltm_memories)} LTM memories from JSON files")
    
    # Process and store given corpus content if not skipping
    if not args.skip_storage and not (args.stm_json and args.ltm_json):
        
        lyra_segments = extract_segments_from_file(args.input_file)
        print(f"Extracted {len(lyra_segments)} segments from {args.input_file}")
        
        # Define user and session for Lyra data
        user = "testuser1"
        session = "testsession1"
        
        # Initialize collection tracking
        collection_evolution = []
        stored_count = 0
        last_collection_count = 0
        
        # Store segments as memories
        for segment in lyra_segments:
            try:
                # Capture collection state before storing
                pre_storage_snapshot = capture_collection_snapshot()
                
                print(f"Storing segment {stored_count+1}/{len(lyra_segments)}: {segment[:50]}...")
                store_memory(MemoryInput(
                    content=segment, 
                    user_id=user, 
                    session_id=session,
                ))
                stored_count += 1
                
                # Capture collection state after storing
                post_storage_snapshot = capture_collection_snapshot()
                
                # Check if new collections were created
                if hasattr(memory_system, 'collection_manager') and memory_system.collection_manager:
                    current_collection_count = len(memory_system.collection_manager.collections)
                    if current_collection_count > last_collection_count:
                        new_collections = current_collection_count - last_collection_count
                        print(f"  {new_collections} new collection(s) created! Total: {current_collection_count}")
                        
                        # Track collection evolution
                        collection_evolution.append({
                            "memory_index": stored_count,
                            "segment_preview": segment[:100],
                            "collections_before": last_collection_count,
                            "collections_after": current_collection_count,
                            "new_collections": new_collections,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "pre_snapshot": pre_storage_snapshot,
                            "post_snapshot": post_storage_snapshot
                        })
                        
                        last_collection_count = current_collection_count
                
                time.sleep(0.1)
            except Exception as e:
                print(f"Error storing segment: {e}")
        
        print(f"\n=== Successfully stored {stored_count} segments of Lyra content ===\n")
        
        # Save collection evolution data
        if collection_evolution:
            evolution_file = os.path.join(args.output_dir, "collection_evolution.json")
            with open(evolution_file, 'w', encoding='utf-8') as f:
                json.dump(collection_evolution, f, indent=2)
            print(f"Collection evolution saved to: {evolution_file}")
        
        # Display collection summary after storage
        display_collection_summary()
    elif args.skip_storage:
        print("=== Skipping storage phase as requested ===")
    
    # Example queries for Lyra content
    fetched_queries = json.load(open(args.queries_file))
    
    # Define user and session for queries
    user = "testuser1"
    session = "testsession1"
    
    print(f"=== Testing memory retrieval with {len(fetched_queries)} queries ===")
    
    # Run test queries
    all_results = {}
    for query in fetched_queries:
        results = compare_memory_sources(query, user, session)
        all_results[query] = results
    
    # Display final collection summary
    print("\n=== Final Smart Collections Summary ===")
    display_collection_summary()
    
    print("\n=== Saving memory contents to JSON files ===")
    
    # Get and save STM memories for rerun later if req (use --stm-json with correct path to previously saved stm_memories.json to skip)
    if not args.stm_json:
        stm_memories = get_stm_memories()
    save_memories_to_json(stm_memories, os.path.join(args.output_dir, "stm_memories.json"))
    
    # Get and save LTM memories for rerun later if req (use --ltm-json with correct path to previously saved ltm_memories.json to skip)
    if not args.ltm_json:
        ltm_memories = get_ltm_memories(user, session)
    save_memories_to_json(ltm_memories, os.path.join(args.output_dir, "ltm_memories.json"))
    
    # Save all query results
    print(f"Saving query results to {os.path.join(args.output_dir, 'full_query_results.json')}")
    query_results = {}
    
    # Create comprehensive analytics summary
    analytics_summary = {
        "total_queries": len(all_results),
        "collections_enabled": hasattr(memory_system, 'collection_manager') and memory_system.collection_manager is not None,
        "final_collection_snapshot": capture_collection_snapshot(),
        "query_analytics": {}
    }
    
    for query, results in all_results.items():
        # Enhanced memory results with collection info
        enhanced_results = {
            "stm": [],
            "ltm": [], 
            "all": [],
            "collection_aware": results.get("collection_aware", {}),
            "collection_snapshot": results.get("collection_snapshot", {}),
            "query_metadata": {
                "query": query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "collections_involved": []
            }
        }
        
        # Process STM results
        for m in results["stm"]:
            enhanced_results["stm"].append({
                "id": m.id, 
                "content": m.content, 
                "score": m.score,
                "memory_tier": m.memory_tier,
                "collection_name": m.collection_name or 'No Collection',
                "category": m.category or 'Uncategorized',
                "composite_score": m.composite_score,
                "timestamp": m.timestamp
            })
        
        # Process LTM results
        for m in results["ltm"]:
            enhanced_results["ltm"].append({
                "id": m.id, 
                "content": m.content, 
                "score": m.score,
                "memory_tier": m.memory_tier,
                "collection_name": m.collection_name or 'No Collection',
                "category": m.category or 'Uncategorized',
                "composite_score": m.composite_score,
                "timestamp": m.timestamp
            })
        
        # Process combined results
        for m in results["all"]:
            enhanced_results["all"].append({
                "id": m.id, 
                "content": m.content, 
                "score": m.score, 
                "tier": m.memory_tier,
                "collection_name": m.collection_name or 'No Collection',
                "category": m.category or 'Uncategorized',
                "composite_score": m.composite_score,
                "timestamp": m.timestamp
            })
        
        # Extract collections involved in this query
        if "collection_aware" in results and "collection_analytics" in results["collection_aware"]:
            collection_details = results["collection_aware"]["collection_analytics"].get("collection_details", {})
            enhanced_results["query_metadata"]["collections_involved"] = list(collection_details.keys())
            
            # Add to analytics summary
            analytics_summary["query_analytics"][query] = {
                "collections_searched": results["collection_aware"]["collection_analytics"].get("collections_searched", 0),
                "query_transformations": results["collection_aware"]["collection_analytics"].get("query_transformations", 0),
                "relevant_collections": results["collection_aware"]["collection_analytics"].get("relevant_collections", 0),
                "total_results": results["collection_aware"]["collection_analytics"].get("total_results", 0),
                "collection_breakdown": collection_details
            }
        
        query_results[query] = enhanced_results
    
    # Save main results file
    with open(os.path.join(args.output_dir, "full_query_results.json"), 'w', encoding='utf-8') as f:
        json.dump(query_results, f, indent=2)
    
    # Save separate analytics file for easier analysis
    with open(os.path.join(args.output_dir, "collection_analytics.json"), 'w', encoding='utf-8') as f:
        json.dump(analytics_summary, f, indent=2)
    
    print("Memory contents saved to:")
    print("- stm_memories.json")
    print("- ltm_memories.json")
    print("- full_query_results.json (enhanced with collection analytics)")
    print("- collection_analytics.json (focused collection insights)")
    
    # Check if collection evolution file was created
    evolution_file = os.path.join(args.output_dir, "collection_evolution.json")
    if os.path.exists(evolution_file):
        print("- collection_evolution.json (timeline of collection creation)")
    
    print("\nEnhanced Analytics Available:")
    print("   • Collection-aware search results with composite scoring")
    print("   • Query transformation analytics per collection")
    print("   • Collection evolution timeline during memory storage")
    print("   • Detailed collection snapshots at each query")
    print("   • Category distribution and pattern analysis")


