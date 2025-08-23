"""Authentication and authorization module"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
import structlog

from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()


class AuthHandler:
    """Handle authentication and authorization"""
    
    def __init__(self):
        self.secret = settings.secret_key
        self.algorithm = settings.algorithm
        self.allowed_keys = settings.allowed_api_keys
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify if API key is valid"""
        return api_key in self.allowed_keys
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, self.secret, algorithm=self.algorithm)
        return encoded_jwt
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.error("JWT decode error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
        """Verify Bearer token from request"""
        token = credentials.credentials
        
        # First check if it's a direct API key
        if self.verify_api_key(token):
            return {
                "type": "api_key",
                "key": token,
                "user_id": "api_user",
                "permissions": ["read", "write"]
            }
        
        # Otherwise try to decode as JWT
        try:
            payload = self.decode_token(token)
            return {
                "type": "jwt",
                "user_id": payload.get("sub"),
                "permissions": payload.get("permissions", ["read"]),
                **payload
            }
        except HTTPException:
            # If both fail, raise unauthorized
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key or token",
                headers={"WWW-Authenticate": "Bearer"},
            )


# Global auth handler instance
auth_handler = AuthHandler()


def require_auth(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Dependency to require authentication"""
    return auth_handler.verify_token(credentials)


def require_write_permission(auth_data: Dict[str, Any] = Security(require_auth)) -> Dict[str, Any]:
    """Dependency to require write permission"""
    if "write" not in auth_data.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for this operation"
        )
    return auth_data


def require_admin_permission(auth_data: Dict[str, Any] = Security(require_auth)) -> Dict[str, Any]:
    """Dependency to require admin permission"""
    if "admin" not in auth_data.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required"
        )
    return auth_data