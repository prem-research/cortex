"""Main FastAPI application with OpenAPI documentation and authentication"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import structlog
import uvicorn
import asyncio
from typing import Dict, Any

from app.config import get_settings
from app.models import (
    MemoryInput, MemoryOutput, SearchQuery, StoreResponse, RetrieveResponse,
    DeleteRequest, UpdateRequest, ClearMemoryRequest, TokenRequest, TokenResponse,
    HealthResponse
)
from app.auth import auth_handler, require_auth, require_write_permission
from app.cortex_service import CortexService
from app.middleware import setup_middleware

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
settings = get_settings()

# Initialize services
cortex_service = CortexService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Cortex API Server", 
               version=settings.app_version,
               debug=settings.debug)
    
    # Start background tasks if needed
    yield
    
    # Shutdown
    logger.info("Shutting down Cortex API Server")
    if hasattr(cortex_service.memory_system, 'shutdown'):
        cortex_service.memory_system.shutdown()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    ## Cortex Memory API Server
    
    A production-ready API server for the Cortex memory system, providing:
    
    - **Memory Storage**: Store and organize memories with automatic analysis
    - **Intelligent Search**: Semantic search with temporal awareness
    - **Smart Collections**: Automatic organization and categorization
    - **Multi-user Support**: Isolated memory spaces per user/session
    - **Authentication**: API key and JWT token authentication
    - **gRPC Support**: High-performance RPC interface
    
    ### Authentication
    
    All endpoints require authentication via Bearer token. Use one of:
    - Direct API key: `Authorization: Bearer your-api-key`
    - JWT token: `Authorization: Bearer jwt-token`
    
    Generate a JWT token using the `/auth/token` endpoint with your API key.
    """,
    docs_url=None,  # Disable default docs to use custom
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom OpenAPI documentation with integrated viewer
@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
async def custom_swagger_ui_html():
    """Serve custom Swagger UI with authentication support"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Interactive API Documentation",
        oauth2_redirect_url="/docs/oauth2-redirect",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )


# Health check endpoint (no auth required)
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check service health and dependencies"""
    services_status = {}
    
    # Check ChromaDB connection
    try:
        # Try to perform a simple operation
        cortex_service.get_system_stats()
        services_status["chromadb"] = "healthy"
    except Exception as e:
        services_status["chromadb"] = f"unhealthy: {str(e)}"
    
    # Check memory system
    services_status["memory_system"] = "healthy"
    
    return HealthResponse(
        status="healthy" if all(s == "healthy" for s in services_status.values()) else "degraded",
        version=settings.app_version,
        timestamp=datetime.utcnow(),
        services=services_status
    )


# Authentication endpoints
@app.post("/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def generate_token(request: TokenRequest):
    """Generate JWT token from API key"""
    if not auth_handler.verify_api_key(request.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Create token with user data
    token_data = {
        "sub": "api_user",
        "permissions": ["read", "write"],
        "api_key_used": request.api_key[:8] + "..."  # Store partial key for audit
    }
    
    expires_delta = timedelta(minutes=request.expires_in)
    access_token = auth_handler.create_access_token(token_data, expires_delta)
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=request.expires_in * 60  # Convert to seconds
    )


# Memory management endpoints
@app.post("/api/v1/memory", response_model=StoreResponse, tags=["Memory Operations"])
async def store_memory(
    memory: MemoryInput,
    auth_data: Dict[str, Any] = Depends(require_write_permission)
):
    """
    Store a new memory in the system.
    
    The memory will be automatically analyzed to extract:
    - Keywords and key concepts
    - Contextual information
    - Relationships with existing memories
    - Category classification
    """
    try:
        memory_id = cortex_service.store_memory(
            content=memory.content,
            context=memory.context,
            tags=memory.tags,
            timestamp=memory.timestamp,
            user_id=memory.user_id or auth_data.get("user_id"),
            session_id=memory.session_id,
            metadata=memory.metadata
        )
        
        return StoreResponse(
            id=memory_id,
            success=True,
            message="Memory stored successfully",
            metadata={"stored_by": auth_data.get("user_id")}
        )
    except Exception as e:
        logger.error("Error storing memory", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store memory: {str(e)}"
        )


@app.post("/api/v1/memory/search", response_model=RetrieveResponse, tags=["Memory Operations"])
async def search_memories(
    query: SearchQuery,
    auth_data: Dict[str, Any] = Depends(require_auth)
):
    """
    Search for memories using semantic search.
    
    Features:
    - Semantic similarity search
    - Temporal weighting for recency
    - Date range filtering
    - Collection-aware search with query enhancement
    - Linked memory traversal
    """
    try:
        results = cortex_service.search_memories(
            query=query.query,
            limit=query.limit,
            memory_source=query.memory_source.value,
            context=query.context,
            tags=query.tags,
            exclude_content=query.exclude_content,
            include_links=query.include_links,
            apply_postprocessing=query.apply_postprocessing,
            user_id=query.user_id or auth_data.get("user_id"),
            session_id=query.session_id,
            temporal_weight=query.temporal_weight,
            date_range=query.date_range
        )
        
        # Convert to response model
        memories = []
        for mem in results["memories"]:
            memories.append(MemoryOutput(**mem))
        
        return RetrieveResponse(
            memories=memories,
            count=results["count"],
            query_metadata={
                "searched_by": auth_data.get("user_id"),
                "query": query.query,
                "temporal_weight": query.temporal_weight
            }
        )
    except Exception as e:
        logger.error("Error searching memories", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search memories: {str(e)}"
        )


@app.get("/api/v1/memory/{memory_id}", response_model=MemoryOutput, tags=["Memory Operations"])
async def get_memory(
    memory_id: str,
    user_id: str = None,
    session_id: str = None,
    auth_data: Dict[str, Any] = Depends(require_auth)
):
    """Get a specific memory by ID"""
    try:
        memory = cortex_service.get_memory(
            memory_id=memory_id,
            user_id=user_id or auth_data.get("user_id"),
            session_id=session_id
        )
        
        if not memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory not found"
            )
        
        return MemoryOutput(**memory)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting memory", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory: {str(e)}"
        )


@app.put("/api/v1/memory", response_model=StoreResponse, tags=["Memory Operations"])
async def update_memory(
    request: UpdateRequest,
    auth_data: Dict[str, Any] = Depends(require_write_permission)
):
    """Update an existing memory"""
    try:
        success = cortex_service.update_memory(
            memory_id=request.memory_id,
            content=request.content,
            context=request.context,
            tags=request.tags,
            metadata=request.metadata
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory not found"
            )
        
        return StoreResponse(
            id=request.memory_id,
            success=True,
            message="Memory updated successfully",
            metadata={"updated_by": auth_data.get("user_id")}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating memory", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update memory: {str(e)}"
        )


@app.delete("/api/v1/memory", response_model=StoreResponse, tags=["Memory Operations"])
async def delete_memory(
    request: DeleteRequest,
    auth_data: Dict[str, Any] = Depends(require_write_permission)
):
    """Delete a memory"""
    try:
        success = cortex_service.delete_memory(
            memory_id=request.memory_id,
            user_id=request.user_id or auth_data.get("user_id"),
            session_id=request.session_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory not found"
            )
        
        return StoreResponse(
            id=request.memory_id,
            success=True,
            message="Memory deleted successfully",
            metadata={"deleted_by": auth_data.get("user_id")}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting memory", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete memory: {str(e)}"
        )


@app.post("/api/v1/memory/clear", response_model=StoreResponse, tags=["Memory Operations"])
async def clear_memories(
    request: ClearMemoryRequest,
    auth_data: Dict[str, Any] = Depends(require_write_permission)
):
    """Clear memories from STM, LTM, or both"""
    try:
        cortex_service.clear_memories(
            memory_source=request.memory_source.value,
            user_id=request.user_id or auth_data.get("user_id"),
            session_id=request.session_id
        )
        
        return StoreResponse(
            id="",
            success=True,
            message=f"Memories cleared from {request.memory_source.value}",
            metadata={"cleared_by": auth_data.get("user_id")}
        )
    except Exception as e:
        logger.error("Error clearing memories", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear memories: {str(e)}"
        )


@app.get("/api/v1/stats", tags=["System"])
async def get_stats(auth_data: Dict[str, Any] = Depends(require_auth)):
    """Get system statistics"""
    try:
        stats = cortex_service.get_system_stats()
        return stats
    except Exception as e:
        logger.error("Error getting stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """API root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "health": "/health"
    }


def start_server():
    """Start the FastAPI server"""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers if not settings.debug else 1,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    start_server()