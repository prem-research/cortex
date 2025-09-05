from fastapi import APIRouter, HTTPException, status
from datetime import datetime

from app.models import HealthResponse, CortexHealthResponse
from app.services.cortex_service import cortex_service

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check for the API service"""
    return HealthResponse(
        status="healthy",
        service="cortex-api",
        timestamp=datetime.utcnow()
    )

@router.get("/cortex", response_model=CortexHealthResponse)
async def cortex_health():
    """Health check for Cortex memory system and ChromaDB connectivity"""
    try:
        is_healthy = await cortex_service.health_check()
        
        if is_healthy:
            return CortexHealthResponse(
                status="healthy",
                cortex="connected",
                chromadb_connected=True
            )
        else:
            return CortexHealthResponse(
                status="unhealthy",
                cortex="connection_failed",
                chromadb_connected=False
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cortex health check failed: {str(e)}"
        )