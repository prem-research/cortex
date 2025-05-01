#!/usr/bin/env python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import logging
import os
import time
import argparse
from cortex.memory_system import AgenticMemorySystem
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the memory system with a small STM capacity to ensure LTM usage
memory_system = AgenticMemorySystem(stm_capacity=5)  

# Define chunking parameters
CHUNK_SIZE = 5000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Define API models
class MemoryInput(BaseModel):
    """Input model for storing a memory"""
    content: str = Field(..., description="The content of the memory to store")
    context: Optional[str] = Field(None, description="Optional context for the memory")
    tags: Optional[List[str]] = Field(None, description="Optional tags for the memory")
    timestamp: Optional[str] = Field(None, description="Optional custom timestamp (format: YYYYMMDDHHMM)")
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
    
    # Analyze content to extract metadata
    metadata = memory_system.analyze_content(content)
    
    # Prepare keyword arguments for memory storage
    kwargs = {}
    if context:
        kwargs["context"] = context
    if tags:
        kwargs["tags"] = tags
    if metadata:
        kwargs["keywords"] = metadata.get("keywords", [])
        if not context:  # Only use metadata context if not explicitly provided
            kwargs["context"] = metadata.get("context", "")
        
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
    session_id: Optional[str] = None
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
        
    Returns:
        RetrieveResponse: Contains list of memories ordered by relevance and count
    """
    # Parse tags if provided
    tag_list = _parse_tags(tags)
            
    # Create filter based on context and tags
    where_filter = _create_filter(context, tag_list)
    
    # Search memories in specified tiers
    results = memory_system.search_memory(
        query=q,
        limit=limit,
        memory_source=memory_source,
        where_filter=where_filter,
        apply_postprocessing=apply_postprocessing,
        context=context,
        user_id=user_id,
        session_id=session_id
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
            "memory_tier": result.get("memory_tier", "unknown")
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
    Compare retrieval results from different memory sources
    """
    print(f"\nQuery: '{query}'")
    
    # Search in STM only
    stm_results = retrieve_memories(
        q=query, 
        user_id=user_id, 
        session_id=session_id,
        memory_source="stm"
    )
    
    # Search in LTM only
    ltm_results = retrieve_memories(
        q=query, 
        user_id=user_id, 
        session_id=session_id,
        memory_source="ltm"
    )
    
    # Instead of doing a separate search for "all", combine and sort the STM and LTM results
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
    # Handle None scores by treating them as lowest priority
    combined_memories.sort(key=lambda x: (x.score is not None, x.score or float('-inf')), reverse=True)
    
    # Create a RetrieveResponse with the combined results
    all_results = RetrieveResponse(
        memories=combined_memories,
        count=len(combined_memories)
    )
    
    # Print results
    print(f"STM results: {stm_results.count}")
    for memory in stm_results.memories:
        print(f"- {memory.content[:100]}... (Tier: {memory.memory_tier}, Score: {memory.score})")
    
    print(f"\nLTM results: {ltm_results.count}")
    for memory in ltm_results.memories:
        print(f"- {memory.content[:100]}... (Tier: {memory.memory_tier}, Score: {memory.score})")
    
    print(f"\nCombined results: {all_results.count}")
    for memory in all_results.memories:
        print(f"- {memory.content[:100]}... (Tier: {memory.memory_tier}, Score: {memory.score})")
    
    print("\n--------------------------------\n")
    
    return {
        "stm": stm_results.memories,
        "ltm": ltm_results.memories,
        "all": all_results.memories
    }

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

def get_ltm_memories():
    """
    Get memories from LTM by searching with a broad query
    """
    # We'll use a very generic query to get as many memories as possible
    results = memory_system.search_memory(
        query="",  # Empty query returns everything
        limit=100,  # Large limit to get most memories
        memory_source="ltm"
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
    
    # If we loaded memories, consolidate them to ensure proper indexing
    if stm_memories or ltm_memories:
        logger.info("Consolidating memories after loading from JSON")
        memory_system.consolidate_memories()
    
    return stm_memories, ltm_memories

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the memory system with Lyra content')
    parser.add_argument('--stm-json', type=str, help='Path to pre-stored STM memories JSON file')
    parser.add_argument('--ltm-json', type=str, help='Path to pre-stored LTM memories JSON file')
    # parser.add_argument('--input-file', type=str, default="../lyra.txt", help='Path to the input text file (default: ../lyra.txt)')
    parser.add_argument('--input-file', type=str, default="../manual.md", help='Path to the input text file (default: ../lyra.txt)')
    parser.add_argument('--skip-storage', action='store_true', help='Skip storing segments and go directly to queries')
    args = parser.parse_args()
    
    # Check if we should load from JSON files
    stm_memories = []
    ltm_memories = []
    if args.stm_json or args.ltm_json:
        stm_memories, ltm_memories = load_memories_from_json(args.stm_json, args.ltm_json)
        print(f"Loaded {len(stm_memories)} STM memories and {len(ltm_memories)} LTM memories from JSON files")
    
    # Process and store Lyra content if not skipping
    if not args.skip_storage and not (args.stm_json and args.ltm_json):
        print("=== Loading and storing Lyra content ===")
        
        # Extract segments from the Lyra text file
        lyra_segments = extract_segments_from_file(args.input_file)
        print(f"Extracted {len(lyra_segments)} segments from {args.input_file}")
        
        # Define user and session for Lyra data
        lyra_user = "lyra"
        lyra_session = "default"
        
        # Store segments as memories
        stored_count = 0
        for segment in lyra_segments:
            if len(segment.strip()) < 50:  # Skip very short segments
                continue
                
            try:
                print(f"Storing segment {stored_count+1}/{len(lyra_segments)}: {segment[:50]}...")
                store_memory(MemoryInput(
                    content=segment, 
                    user_id=lyra_user, 
                    session_id=lyra_session,
                ))
                stored_count += 1
                # Small delay to allow processing
                time.sleep(0.2)
            except Exception as e:
                print(f"Error storing segment: {e}")
        
        print(f"\n=== Successfully stored {stored_count} segments of Lyra content ===\n")
    elif args.skip_storage:
        print("=== Skipping storage phase as requested ===")
    
    # Example queries for Lyra content
    # lyra_queries = [
    #     "Who is Lyra Drake?",
    #     "What are Lyra's artistic abilities?",
    #     "What is Lyra's background?",
    #     "What books has Lyra read?",
    #     "What are Lyra's thoughts on faith?",
    #     "What does Lyra think about beauty?",
    #     "What are Lyra's quotes about pleasure?",
    #     "Desert experience with Mr. Maverick"
    # ]
    lyra_queries = [
        "what is the SMEM memory address of ICSSG1 core",
        "what is the specific mechanism for enabling IP Halt with corresponding CPU halt in the CRC_HALTEN register?",
        "how does the MSS_CTRL_MSS_CR5B1_AXI_WR_BUS_SAFETY_ERR register handle dual error detection in different segments of the Data Bus?",
        "what is the relationship between MPU_ADDR_INTR_ERRAGG3_MASK register and protection error propagation to R5SS1 CORE1?",
        "explain the difference between masking an error with 1'b1 versus 1'b0 in the MPU_PROT_INTR_ERRAGG3_MASK register",
        "what are the specific conditions required to inject a fault for request signals on Safe Interconnect using MSS_TPTC_A1_RD_BUS_SAFETY_FI_GLOBAL_SAFE_REQ?",
        "how does the HSM_TPTC_A0_WR_BUS_SAFETY_CTRL_TYPE field differ from HSM_TPTC_A0_WR_BUS_SAFETY_CTRL_ENABLE in terms of functionality?",
        "what is the significance of the address latching mechanism when a parity error occurs in the B0TCM of R5SS1 CORE0?",
        "how does the raw status reporting in MPU_ADDR_INTR_ERRAGG3_STATUS_RAW differ from the masked status reporting?",
        "what is the architectural significance of the MSS_CR5B1_AXI_WR_BUS_SAFETY_ERR_SEC field in detecting single errors in the Data port?",
        "explain the process of clearing an interrupt in the MSS_TPCC_A_INTAGG_STATUS register and its relationship to the MSS_TPCC_A_INTAGG_MASK"
    ]
    
    # Define user and session for queries
    lyra_user = "lyra"
    lyra_session = "default"
    
    print("=== Testing memory retrieval with Lyra-specific queries ===")
    
    # Run test queries
    all_results = {}
    for query in lyra_queries:
        results = compare_memory_sources(query, lyra_user, lyra_session)
        all_results[query] = results
    
    print("\n=== Saving memory contents to JSON files ===")
    
    # Get and save STM memories
    if not args.stm_json:  # Only get from system if not loaded from file
        stm_memories = get_stm_memories()
    save_memories_to_json(stm_memories, "stm_memories.json")
    
    # Get and save LTM memories
    if not args.ltm_json:  # Only get from system if not loaded from file
        ltm_memories = get_ltm_memories()
    save_memories_to_json(ltm_memories, "ltm_memories.json")
    
    # Save all query results
    print("Saving query results to queries_results.json")
    query_results = {}
    for query, results in all_results.items():
        query_results[query] = {
            "stm": [{"id": m.id, "content": m.content, "score": m.score} for m in results["stm"]],
            "ltm": [{"id": m.id, "content": m.content, "score": m.score} for m in results["ltm"]],
            "all": [{"id": m.id, "content": m.content, "score": m.score, "tier": m.memory_tier} for m in results["all"]]
        }
    
    with open("query_results.json", 'w', encoding='utf-8') as f:
        json.dump(query_results, f, indent=2)
    
    print("Done! Memory contents saved to:")
    print("- stm_memories.json")
    print("- ltm_memories.json")
    print("- query_results.json")


