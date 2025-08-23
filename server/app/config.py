"""Configuration management for Cortex API Server"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Server Configuration
    app_name: str = "Cortex Memory API Server"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    api_keys: str  # Comma-separated list
    
    # Database
    database_url: Optional[str] = None
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    rate_limit_per_minute: int = 100
    
    # Cortex Configuration
    openai_api_key: str
    cortex_model_name: str = "text-embedding-3-small"
    cortex_llm_backend: str = "openai"
    cortex_llm_model: str = "gpt-4o-mini"
    cortex_stm_capacity: int = 100
    cortex_enable_smart_collections: bool = True
    cortex_enable_background_processing: bool = True
    
    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8003
    chroma_uri: str = "http://localhost:8003"
    
    # gRPC Configuration
    grpc_server_port: int = 50051
    grpc_max_workers: int = 10
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def allowed_api_keys(self) -> List[str]:
        """Parse comma-separated API keys"""
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]
    
    @property
    def cors_origins(self) -> List[str]:
        """Allowed CORS origins"""
        if self.debug:
            return ["*"]
        return [
            "http://localhost:3000",
            "http://localhost:8080",
            "https://yourdomain.com"
        ]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()