import os
from typing import Optional

class Settings:
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:7432/cortex_api")
    
    # JWT
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_DAYS: int = 365
    
    # Cortex
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CHROMA_URI: str = os.getenv("CHROMA_URI", "http://localhost:7003")
    
    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Cortex Memory API"
    PROJECT_VERSION: str = "1.0.0"
    
    # CORS
    ALLOWED_ORIGINS: list = ["*"]
    
    @property
    def cors_origins(self) -> list:
        return self.ALLOWED_ORIGINS

settings = Settings()