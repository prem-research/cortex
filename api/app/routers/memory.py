from fastapi import APIRouter, Depends, HTTPException, status

from app.models import (
    MemoryCreateRequest, MemoryResponse, SearchRequest, SearchResponse,
    ClearRequest, ClearResponse, User, LinkedMemoriesRequest, MemoriesWithLinkedRequest
)
from app.auth import get_current_user
from app.services.cortex_service import cortex_service

router = APIRouter(prefix="/memory", tags=["memory"])

@router.post("/add", response_model=MemoryResponse,
    summary="Add Memory",
    description="""
Store a new memory with automatic content analysis.

Cortex automatically extracts keywords, context, and relationships.
> Note: session_id can also be a constant global id where you'd like to dump everything, or you can based on your application logic segregate memories using custom session_id
**Example Request:**
```json
{
  "content": "I had a productive meeting with the team about Q1 goals",
  "session_id": "work_session",
  "metadata": {"priority": "high"}
}
```
    """
)
async def add_memory(
    request: MemoryCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Add a new memory note with automatic analysis"""
    try:
        memory_id = await cortex_service.add_memory_note(
            content=request.content,
            user_id=str(current_user.id),
            session_id=request.session_id,
            time=request.time,
            context=request.context,
            tags=request.tags,
            **request.metadata
        )
        return MemoryResponse(
            memory_id=memory_id,
            message="Memory added successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding memory: {str(e)}"
        )

@router.post("/search", response_model=SearchResponse)
async def search_memories(
    request: SearchRequest,
    current_user: User = Depends(get_current_user)
):
    """Search memories"""
    try:
        results = await cortex_service.search_memories(
            query=request.query,
            user_id=str(current_user.id),
            session_id=request.session_id,
            memory_source=request.memory_source,
            temporal_weight=request.temporal_weight,
            date_range=request.date_range,
            where_filter=request.where_filter,
            limit=request.limit
        )
        return SearchResponse(
            results=results,
            count=len(results)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching memories: {str(e)}"
        )

@router.get("/get/{memory_id}")
async def get_memory(
    memory_id: str,
    session_id: str = '',
    current_user: User = Depends(get_current_user)
):
    """Get a specific memory by ID"""
    try:
        memory = await cortex_service.get_memory(
            memory_id=memory_id,
            user_id=str(current_user.id),
            session_id=session_id
        )
        if not memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory not found"
            )
        return memory
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving memory: {str(e)}"
        )

@router.post("/get-linked", 
    summary="Get Linked Memories",
    description="Retrieve memories linked to specific memory IDs based on semantic similarity and relationships."
)
async def get_linked_memories(
    request: LinkedMemoriesRequest,
    current_user: User = Depends(get_current_user)
):
    """Get linked memories from a list of memory IDs"""
    try:
        linked_memories = await cortex_service.get_linked_memories(
            memory_ids=request.memory_ids,
            user_id=str(current_user.id),
            session_id=request.session_id,
            limit=request.limit
        )
        return {
            "linked_memories": linked_memories,
            "count": len(linked_memories)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting linked memories: {str(e)}"
        )

@router.post("/get-with-linked",
    summary="Get Memories with Linked Context",
    description="Retrieve specific memories along with their related linked memories for context."
)
async def get_memories_with_linked(
    request: MemoriesWithLinkedRequest,
    current_user: User = Depends(get_current_user)
):
    """Get memories and their linked memories together"""
    try:
        result = await cortex_service.get_memories_with_linked(
            memory_ids=request.memory_ids,
            user_id=str(current_user.id),
            session_id=request.session_id,
            linked_limit=request.linked_limit
        )
        return {
            "memories": result["memories"],
            "linked_memories": result["linked_memories"],
            "memory_count": len(result["memories"]),
            "linked_count": len(result["linked_memories"])
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting memories with linked: {str(e)}"
        )

@router.post("/clear", response_model=ClearResponse)
async def clear_memories(
    request: ClearRequest,
    current_user: User = Depends(get_current_user)
):
    """Clear memories for user/session"""
    try:
        success = await cortex_service.clear_memories(
            user_id=str(current_user.id),
            session_id=request.session_id
        )
        return ClearResponse(
            success=success,
            message="Memories cleared successfully" if success else "Failed to clear memories"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing memories: {str(e)}"
        )