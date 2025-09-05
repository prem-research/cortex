from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from typing import Optional, List, Dict, Any
from datetime import datetime

# SQLAlchemy models
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(128), nullable=False)
    api_key = Column(String(64), unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# Pydantic models for requests/responses

# Auth models
class UserCreate(BaseModel):
    username: str = Field(..., description="Unique username (3-50 characters)", example="john_doe")
    email: EmailStr = Field(..., description="Valid email address", example="john@example.com")
    password: str = Field(..., description="Password (min 8 characters)", example="secure_password123")

class UserLogin(BaseModel):
    username: str = Field(..., description="Username for login", example="john_doe")
    password: str = Field(..., description="User password", example="secure_password123")

class Token(BaseModel):
    access_token: str = Field(..., description="JWT access token (valid for 365 days)", example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    token_type: str = Field("bearer", description="Token type for Authorization header")
    user_id: str = Field(..., description="User ID", example="123")

class UserResponse(BaseModel):
    id: int = Field(..., description="Unique user ID", example=123)
    username: str = Field(..., description="Username", example="john_doe")
    email: str = Field(..., description="User email", example="john@example.com")
    is_active: bool = Field(..., description="Whether user account is active", example=True)

# Memory models
class MemoryCreateRequest(BaseModel):
    content: str = Field(..., description="Memory content to store", example="I had a productive meeting with the team about Q1 goals")
    session_id: Optional[str] = Field('', description="Session ID for memory isolation (optional)", example="work_session")
    time: Optional[str] = Field(None, description="RFC3339 timestamp (optional, defaults to now)", example="2024-01-15T10:30:00Z")
    context: Optional[str] = Field(None, description="Context about the memory being created (optional, preferred to not fill as it'll be auto-generated)")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization (optional, preferred to not fill as it'll be auto-generated)")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Additional metadata (optional)", example={"priority": "high", "department": "engineering"})

class MemoryResponse(BaseModel):
    memory_id: str = Field(..., description="Unique ID of the created memory", example="mem_12345abcd")
    message: str = Field(..., description="Success message", example="Memory added successfully")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text", example="team meeting goals")
    session_id: Optional[str] = Field('', description="Session ID to search within (optional)", example="work_session")
    memory_source: str = Field("ltm", description="Memory source: 'stm', 'ltm', or 'all'", example="ltm")
    temporal_weight: float = Field(0.0, description="Temporal weighting (0.0-1.0): higher values favor recent memories", example=0.3)
    date_range: Optional[str] = Field(None, description="Date filter: RFC3339 timestamp, date ranges (JSON string), or natural language ('yesterday', 'last week', '2024-01')", example='{"start": "2024-01-01T00:00:00Z", "end": "2024-01-31T23:59:59Z"}')
    where_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter conditions", example={"tags": {"$contains": "work"}})
    limit: int = Field(10, description="Maximum number of results (1-50)", example=5)

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="List of matching memories with metadata")
    count: int = Field(..., description="Number of results returned", example=5)

class ClearRequest(BaseModel):
    session_id: Optional[str] = Field('', description="Session ID to clear (empty = clear all user memories)", example="work_session")

class ClearResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful", example=True)
    message: str = Field(..., description="Operation result message", example="Memories cleared successfully")

# Health check models
class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status", example="healthy")
    service: str = Field(..., description="Service name", example="cortex-api")
    timestamp: Optional[datetime] = Field(None, description="Health check timestamp")

class CortexHealthResponse(BaseModel):
    status: str = Field(..., description="Overall Cortex health status", example="healthy")
    cortex: str = Field(..., description="Cortex connection status", example="connected")
    chromadb_connected: bool = Field(..., description="Whether ChromaDB is accessible", example=True)

# Linked memory request models
class LinkedMemoriesRequest(BaseModel):
    memory_ids: List[str] = Field(..., description="List of memory IDs", example=["id1", "id2"])
    session_id: Optional[str] = Field('', description="Session filter", example="work")
    limit: int = Field(4, description="Max results", example=4)

class MemoriesWithLinkedRequest(LinkedMemoriesRequest):
    linked_limit: int = Field(4, description="Max linked per memory", example=3)