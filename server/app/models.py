"""Pydantic models for API request/response schemas"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MemorySource(str, Enum):
    """Memory source options"""
    STM = "stm"
    LTM = "ltm"
    ALL = "all"


class MemoryInput(BaseModel):
    """Input model for storing a memory"""
    content: str = Field(..., description="The content of the memory to store", min_length=1)
    context: Optional[str] = Field(None, description="Optional context for the memory")
    tags: Optional[List[str]] = Field(None, description="Optional tags for the memory")
    timestamp: Optional[str] = Field(None, description="Optional custom timestamp (RFC3339 format)")
    user_id: Optional[str] = Field(None, description="Optional user identifier for memory segregation")
    session_id: Optional[str] = Field(None, description="Optional session identifier for memory segregation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "User prefers TypeScript over JavaScript for new projects",
                "context": "programming preferences",
                "tags": ["typescript", "preferences", "development"],
                "user_id": "user_123",
                "session_id": "session_456"
            }
        }


class MemoryOutput(BaseModel):
    """Output model for a memory"""
    id: str = Field(..., description="Unique identifier of the memory")
    content: str = Field(..., description="Content of the memory")
    context: Optional[str] = Field(None, description="Context of the memory")
    tags: Optional[List[str]] = Field(None, description="Tags associated with the memory")
    keywords: Optional[List[str]] = Field(None, description="Extracted keywords")
    timestamp: Optional[str] = Field(None, description="Timestamp when memory was created")
    score: Optional[float] = Field(None, description="Relevance score (for search results)")
    is_linked: bool = Field(False, description="Whether this is a linked memory")
    memory_tier: Optional[str] = Field(None, description="Memory tier (STM/LTM)")
    collection_name: Optional[str] = Field(None, description="Collection this memory belongs to")
    category: Optional[str] = Field(None, description="Category classification")
    composite_score: Optional[float] = Field(None, description="Composite score including collection relevance")
    relationship_type: Optional[str] = Field(None, description="Type of relationship (for linked memories)")
    relationship_strength: Optional[float] = Field(None, description="Strength of relationship")
    relationship_reason: Optional[str] = Field(None, description="Reason for relationship")


class SearchQuery(BaseModel):
    """Search query parameters"""
    query: str = Field(..., description="Search query text", min_length=1)
    limit: int = Field(10, description="Maximum number of results", ge=1, le=100)
    memory_source: MemorySource = Field(MemorySource.ALL, description="Which memory tiers to search")
    context: Optional[str] = Field(None, description="Optional context for improved relevance")
    tags: Optional[List[str]] = Field(None, description="Optional tags for filtering")
    exclude_content: bool = Field(False, description="Whether to exclude content in results")
    include_links: bool = Field(True, description="Whether to include linked memories")
    apply_postprocessing: bool = Field(True, description="Whether to apply post-retrieval processing")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    temporal_weight: Optional[float] = Field(None, description="Temporal weighting (0.0-1.0)", ge=0.0, le=1.0)
    date_range: Optional[str] = Field(None, description="Date range filter (e.g., 'last week', '2023-01')")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "programming preferences",
                "limit": 5,
                "memory_source": "all",
                "temporal_weight": 0.3,
                "user_id": "user_123"
            }
        }


class StoreResponse(BaseModel):
    """Response model for the store endpoint"""
    id: str = Field(..., description="ID of the stored memory")
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RetrieveResponse(BaseModel):
    """Response model for the retrieve endpoint"""
    memories: List[MemoryOutput] = Field(..., description="List of retrieved memories")
    count: int = Field(..., description="Total count of memories returned")
    query_metadata: Optional[Dict[str, Any]] = Field(None, description="Query execution metadata")


class DeleteRequest(BaseModel):
    """Request model for deleting a memory"""
    memory_id: str = Field(..., description="ID of the memory to delete")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")


class UpdateRequest(BaseModel):
    """Request model for updating a memory"""
    memory_id: str = Field(..., description="ID of the memory to update")
    content: Optional[str] = Field(None, description="New content")
    context: Optional[str] = Field(None, description="New context")
    tags: Optional[List[str]] = Field(None, description="New tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata to update")


class ClearMemoryRequest(BaseModel):
    """Request model for clearing memory"""
    memory_source: MemorySource = Field(..., description="Which memory tier to clear")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")


class TokenRequest(BaseModel):
    """Request model for token generation"""
    api_key: str = Field(..., description="API key for authentication")
    expires_in: Optional[int] = Field(1440, description="Token expiration in minutes")


class TokenResponse(BaseModel):
    """Response model for token generation"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current timestamp")
    services: Dict[str, str] = Field(..., description="Status of dependent services")