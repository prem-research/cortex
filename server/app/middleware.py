"""Middleware for rate limiting, logging, and monitoring"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
import time
import structlog
import redis
from typing import Optional
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

# Prometheus metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])


class RateLimiter:
    """Redis-based rate limiter"""
    
    def __init__(self, redis_url: str, requests_per_minute: int):
        self.redis_client = None
        self.requests_per_minute = requests_per_minute
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis rate limiter initialized")
        except Exception as e:
            logger.warning(f"Redis not available for rate limiting: {e}")
    
    async def check_rate_limit(self, key: str) -> bool:
        """Check if request is within rate limit"""
        if not self.redis_client:
            return True  # Allow if Redis is not available
        
        try:
            pipe = self.redis_client.pipeline()
            now = time.time()
            window = 60  # 1 minute window
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, now - window)
            # Add current request
            pipe.zadd(key, {str(now): now})
            # Count requests in window
            pipe.zcard(key)
            # Set expiry
            pipe.expire(key, window)
            
            results = pipe.execute()
            request_count = results[2]
            
            return request_count <= self.requests_per_minute
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error


# Initialize rate limiter
rate_limiter = RateLimiter(settings.redis_url, settings.rate_limit_per_minute)


async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Extract API key or user ID for rate limiting
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        key = f"rate_limit:{auth_header[7:20]}"  # Use first part of token
    else:
        key = f"rate_limit:{request.client.host}"
    
    # Check rate limit
    if not await rate_limiter.check_rate_limit(key):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    response = await call_next(request)
    return response


async def logging_middleware(request: Request, call_next):
    """Request logging middleware"""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown"
    )
    
    try:
        response = await call_next(request)
        
        # Log response
        duration = time.time() - start_time
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration
        )
        
        # Update Prometheus metrics
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Add custom headers
        response.headers["X-Request-ID"] = str(time.time())
        response.headers["X-Process-Time"] = str(duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Request failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            duration=duration
        )
        
        # Update error metrics
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


async def metrics_endpoint(request: Request):
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


def setup_middleware(app: FastAPI):
    """Setup all middleware for the application"""
    
    # Add middleware in order
    app.middleware("http")(logging_middleware)
    app.middleware("http")(rate_limit_middleware)
    
    # Add metrics endpoint if enabled
    if settings.enable_metrics:
        app.add_route("/metrics", metrics_endpoint, methods=["GET"])
        logger.info("Metrics endpoint enabled at /metrics")