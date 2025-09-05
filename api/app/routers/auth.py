from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

from app.models import UserCreate, UserLogin, Token, UserResponse, User
from app.database import get_db
from app.auth import AuthService, authenticate_user, get_current_user

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=UserResponse,
    summary="Register New User",
    description="""
Register a new user account and get API credentials.

**Example Request:**
```json
{
  "username": "john_doe", 
  "email": "john@example.com",
  "password": "secure_password123"
}
```

**Returns:** User info for the registered account.
    """
)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user and receive API key"""
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    
    # Create new user
    hashed_password = AuthService.get_password_hash(user_data.password)
    api_key = AuthService.generate_api_key()
    
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        api_key=api_key
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse(
        id=db_user.id,
        username=db_user.username,
        email=db_user.email,
        is_active=db_user.is_active
    )

@router.post("/login", response_model=Token,
    summary="User Login", 
    description="""
Login with username/password to get JWT access token.

**Example Request:**
```json
{
  "username": "john_doe",
  "password": "secure_password123" 
}
```

**Returns:** JWT token (valid for 365 days) for authentication.

**Usage:** Add `Authorization: Bearer <token>` header to requests.
    """
)
async def login(
    credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """Login user and return JWT access token"""
    user = authenticate_user(db, credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(days=365)
    access_token = AuthService.create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        user_id=str(user.id)
    )

@router.get("/me", response_model=UserResponse,
    summary="Get Current User",
    description="""
Get information about the currently authenticated user.

**Authentication:** Requires JWT Bearer token.

**Example Header:**
- `Authorization: Bearer <jwt_token>`
    """
)
async def read_users_me(
    current_user: User = Depends(get_current_user)
):
    """Get current authenticated user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active
    )