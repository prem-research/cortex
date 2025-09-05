from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import settings
from app.database import init_db
from app.routers import auth, memory, health

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with detailed documentation
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="""
## Cortex Memory API

A powerful memory system for AI agents with cognitive architecture featuring:

- **Dual-tier Memory**: Fast STM (short-term) + persistent LTM (long-term) with ChromaDB
- **Smart Collections**: Context-aware categorization and domain organization  
- **Temporal Awareness**: Time-sensitive search with recency weighting
- **Memory Evolution**: Automatic relationship building and content consolidation
- **Multi-user Support**: Complete isolation by user_id and optional session_id

### Authentication
Two methods supported:
1. **JWT Tokens**: Register → Login → Use Bearer token
2. **API Keys**: Get from registration → Use X-API-Key header

### Memory Operations
- **Add**: Store content with auto-analysis for keywords, context, tags
- **Search**: Semantic search with temporal filters and metadata queries
- **Get**: Retrieve specific memories by ID
- **Clear**: Remove memories by user/session scope

### Quick Start
1. Register: `POST /auth/register`
2. Login: `POST /auth/login` (get JWT token)
3. Add memory: `POST /memory/add` with Bearer token
4. Search: `POST /memory/search` with your query

See endpoint documentation below for detailed examples.
    """,
    version=settings.PROJECT_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Cortex API Support",
        "url": "https://github.com/prem-research/cortex",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(memory.router, prefix=settings.API_V1_STR)
app.include_router(health.router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

@app.get("/", 
    summary="API Root",
    description="Welcome endpoint with basic API information and links to documentation."
)
async def root():
    """Root endpoint with API information"""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.PROJECT_VERSION,
        "docs_url": "/docs",
        "health_check": f"{settings.API_V1_STR}/health",
        "api_base": settings.API_V1_STR,
        "features": [
            "JWT & API Key Authentication",
            "Multi-user Memory Isolation", 
            "Semantic Search with Temporal Filtering",
            "Smart Collections & Memory Evolution",
            "Session-based Memory Organization"
        ]
    }

@app.get("/info",
    summary="API Information", 
    description="Detailed API information including all available endpoints and their purposes."
)
async def api_info():
    """API information and available endpoints"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "description": "Advanced memory system for AI agents with cognitive architecture",
        "endpoints": {
            "authentication": {
                "register": f"{settings.API_V1_STR}/auth/register",
                "login": f"{settings.API_V1_STR}/auth/login", 
                "me": f"{settings.API_V1_STR}/auth/me"
            },
            "memory": {
                "add": f"{settings.API_V1_STR}/memory/add",
                "search": f"{settings.API_V1_STR}/memory/search",
                "get": f"{settings.API_V1_STR}/memory/get/{{memory_id}}",
                "get_linked": f"{settings.API_V1_STR}/memory/get-linked", 
                "get_with_linked": f"{settings.API_V1_STR}/memory/get-with-linked",
                "clear": f"{settings.API_V1_STR}/memory/clear"
            },
            "health": {
                "api": f"{settings.API_V1_STR}/health",
                "cortex": f"{settings.API_V1_STR}/health/cortex"
            }
        }
    }