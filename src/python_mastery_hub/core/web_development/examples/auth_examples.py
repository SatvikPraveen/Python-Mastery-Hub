"""
Authentication and Security Examples for Web Development.

Demonstrates JWT authentication, OAuth2, password hashing, and security best practices.
"""

from typing import Any, Dict


class AuthExamples:
    """Collection of authentication and security examples."""

    @staticmethod
    def get_jwt_example() -> Dict[str, Any]:
        """JWT authentication example."""
        return {
            "title": "JWT Authentication",
            "description": "Implement JWT-based authentication in FastAPI",
            "code": '''from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI()
security = HTTPBearer()

SECRET_KEY = "your-secret-key-change-this"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthCredentials = Depends(security)):
    """Verify JWT token."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/login")
async def login(username: str, password: str):
    """Login endpoint."""
    # Verify credentials (simplified)
    if username == "user" and password == "pass":
        access_token = create_access_token(data={"sub": username})
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/protected")
async def protected_route(user_id: str = Depends(verify_token)):
    """Protected route requiring authentication."""
    return {"user_id": user_id, "message": "This is protected"}
''',
        }

    @staticmethod
    def get_password_hashing_example() -> Dict[str, Any]:
        """Password hashing example."""
        return {
            "title": "Secure Password Hashing",
            "description": "Hash and verify passwords securely",
            "code": '''from passlib.context import CryptContext

# Initialize password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

# Usage
password = "SecurePassword123!"
hashed = hash_password(password)
print(f"Original: {password}")
print(f"Hashed: {hashed}")
print(f"Verification: {verify_password(password, hashed)}")
''',
        }

    @staticmethod
    def get_oauth2_example() -> Dict[str, Any]:
        """OAuth2 integration example."""
        return {
            "title": "OAuth2 Integration",
            "description": "Implement OAuth2 with GitHub",
            "code": '''from fastapi import FastAPI
from authlib.integrations.starlette_client import OAuth
import os

app = FastAPI()
oauth = OAuth()

# Configure GitHub OAuth
oauth.register(
    name='github',
    client_id=os.getenv('GITHUB_CLIENT_ID'),
    client_secret=os.getenv('GITHUB_CLIENT_SECRET'),
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    userinfo_endpoint='https://api.github.com/user',
)

@app.get("/auth/login/github")
async def login_github(request):
    """Initiate GitHub OAuth login."""
    redirect_uri = str(request.url_for('auth_callback_github'))
    return await oauth.github.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback/github")
async def auth_callback_github(request):
    """Handle GitHub OAuth callback."""
    token = await oauth.github.authorize_access_token(request)
    user = token.get('userinfo')
    # Save user to database
    return {"user": user, "token": token}
''',
        }

    @staticmethod
    def get_cors_example() -> Dict[str, Any]:
        """CORS configuration example."""
        return {
            "title": "CORS Configuration",
            "description": "Configure CORS for frontend-backend communication",
            "code": '''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# More restrictive configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["X-Total-Count"],
    max_age=600,
)

@app.get("/api/data")
async def get_data():
    """API endpoint accessible from configured origins."""
    return {"data": "example"}
''',
        }

    @staticmethod
    def get_session_management_example() -> Dict[str, Any]:
        """Session management example."""
        return {
            "title": "Session Management",
            "description": "Manage user sessions with Redis",
            "code": '''from fastapi import FastAPI, Depends, HTTPException
from datetime import datetime, timedelta
import redis
import json

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, db=0)

SESSION_EXPIRE = 3600  # 1 hour

def create_session(user_id: str) -> str:
    """Create a new session."""
    session_id = f"session:{user_id}:{datetime.now().timestamp()}"
    session_data = {
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
    }
    redis_client.setex(
        session_id,
        SESSION_EXPIRE,
        json.dumps(session_data)
    )
    return session_id

def get_session(session_id: str) -> dict:
    """Get session data."""
    data = redis_client.get(session_id)
    if not data:
        raise HTTPException(status_code=401, detail="Invalid session")
    return json.loads(data)

@app.post("/login")
async def login(username: str, password: str):
    """Login and create session."""
    # Verify credentials
    session_id = create_session(username)
    return {"session_id": session_id}

@app.get("/profile")
async def get_profile(session_id: str):
    """Get user profile."""
    session = get_session(session_id)
    return {"user_id": session["user_id"]}

@app.post("/logout")
async def logout(session_id: str):
    """Logout and delete session."""
    redis_client.delete(session_id)
    return {"message": "Logged out"}
''',
        }

    @staticmethod
    def get_auth_examples() -> Dict[str, Dict[str, Any]]:
        """Get all authentication examples."""
        return {
            "jwt_authentication": AuthExamples.get_jwt_example(),
            "password_hashing": AuthExamples.get_password_hashing_example(),
            "oauth2": AuthExamples.get_oauth2_example(),
            "cors": AuthExamples.get_cors_example(),
            "session_management": AuthExamples.get_session_management_example(),
        }

    # Alias for backwards compatibility
    @staticmethod
    def get_all_examples() -> Dict[str, Dict[str, Any]]:
        """Get all authentication examples (deprecated, use get_auth_examples)."""
        return AuthExamples.get_auth_examples()

    @staticmethod
    def demonstrate() -> Dict[str, Any]:
        """Demonstrate authentication concepts."""
        return {
            "title": "Authentication & Security",
            "description": "Learn authentication and security best practices",
            "examples": AuthExamples.get_all_examples(),
            "best_practices": [
                "Always hash passwords before storing",
                "Use strong SECRET_KEY values",
                "Implement token expiration",
                "Use HTTPS in production",
                "Validate all inputs",
                "Implement rate limiting for auth endpoints",
                "Use secure cookie settings",
                "Keep secrets in environment variables",
            ],
        }


__all__ = ["AuthExamples"]
