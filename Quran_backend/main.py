from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from jose import jwt, JWTError
import requests
from passlib.context import CryptContext

from backend_core import quick_search
from database import init_db, get_connection
import secrets
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
import os
from dotenv import load_dotenv

load_dotenv()



# ===================== Configuration =====================
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
SECRET_KEY = os.getenv("SECRET_KEY")
RECAPTCHA_SECRET = os.getenv("RECAPTCHA_SECRET")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is not set in environment variables/.env")
if not RECAPTCHA_SECRET:
    raise RuntimeError("RECAPTCHA_SECRET is not set in environment variables/.env")




pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

init_db()


# ===================== Helper functions =====================

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_recaptcha(token: str) -> bool:
    """
    Verify Google reCAPTCHA token with Google's API.
    """
    if not token:
        return False

    url = "https://www.google.com/recaptcha/api/siteverify"
    data = {
        "secret": RECAPTCHA_SECRET,
        "response": token,
    }
    try:
        resp = requests.post(url, data=data, timeout=5)
        result = resp.json()
        return result.get("success", False)
    except Exception:
        return False


def get_current_user(authorization: str = Header(None)):
    """
    Reads the Authorization header ("Bearer <token>") and validates JWT.
    """
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


# ===================== Pydantic models (schemas) =====================

class GoogleLoginRequest(BaseModel):
    id_token: str

class SignupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str
    recaptcha_token: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 30


class VerseResult(BaseModel):
    rank: int
    score: float
    sura: int
    ayah: int
    arabic: str
    english: str


class SearchResponse(BaseModel):
    query: str
    results: List[VerseResult]


# ===================== FastAPI app setup =====================

app = FastAPI(
    title="Qur'an Semantic Search API",
    description="Semantic search over Qur'anic verses using BGE-M3 and MAQAS-based query expansion.",
    version="1.0.0",
)

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== Routes =====================

@app.get("/ping")
def ping():
    """Simple health check."""
    return {"status": "ok", "message": "API is running"}


@app.post("/signup")
def signup(req: SignupRequest):
    """
    Create a new user account.
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (req.username, hash_password(req.password), datetime.utcnow().isoformat())
        )
        conn.commit()
    except Exception:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    conn.close()
    return {"status": "ok", "message": "Account created successfully"}


@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """
    Login with username/password + reCAPTCHA.
    Returns JWT token.
    """
    # 1) Verify reCAPTCHA
    if not verify_recaptcha(req.recaptcha_token):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="reCAPTCHA verification failed",
        )

    # 2) Check user in DB
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (req.username,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    stored_hash = row[0]
    if not verify_password(req.password, stored_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # 3) Issue token
    access_token = create_access_token(data={"sub": req.username})
    return LoginResponse(access_token=access_token)

@app.post("/google-login", response_model=LoginResponse)
def google_login(req: GoogleLoginRequest):
    if not req.id_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Google ID token",
        )

    try:
        idinfo = google_id_token.verify_oauth2_token(
            req.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Google token",
        )

    google_sub = idinfo.get("sub")          
    email = idinfo.get("email")
    name = idinfo.get("name") or email

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account has no email",
        )

    username = email

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()

    if not row:
        random_password = secrets.token_urlsafe(16)
        password_hash = hash_password(random_password)

        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.utcnow().isoformat())
        )
        conn.commit()

    conn.close()

    access_token = create_access_token(data={"sub": username})
    return LoginResponse(access_token=access_token)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, current_user: str = Header(None, alias="Authorization")):
    """
    Main semantic search endpoint.
    Protected: requires Bearer token.
    """
    user = get_current_user(authorization=current_user)

    results = quick_search(req.query, top_k=req.top_k)
    verse_results = [VerseResult(**r) for r in results]

    return {
        "query": req.query,
        "results": verse_results
    }
