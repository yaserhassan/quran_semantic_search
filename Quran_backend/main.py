# main.py
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from jose import jwt, JWTError
import requests
from passlib.context import CryptContext
import secrets
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
import os
from dotenv import load_dotenv

from backend_core import search_api   # ✅ changed
from database import init_db, get_connection

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
    if not token:
        return False
    url = "https://www.google.com/recaptcha/api/siteverify"
    data = {"secret": RECAPTCHA_SECRET, "response": token}
    try:
        resp = requests.post(url, data=data, timeout=5)
        result = resp.json()
        return result.get("success", False)
    except Exception:
        return False

def get_current_user(authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        return username
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

# ===================== Schemas =====================
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

class VerseResult(BaseModel):
    rank: int
    ref: str
    arabic: str
    english: str

class SearchRequest(BaseModel):
    query: str
    page: int = 1
    page_size: int = 10
    k_faiss: int = 1200
    top_expansions: int = 12

class SearchResponse(BaseModel):
    query: str
    total: int
    page: int
    page_size: int
    results: List[VerseResult]


# ===================== App =====================
app = FastAPI(
    title="Qur'an Semantic Search API",
    description="Semantic search over Qur'anic verses using BGE-M3, MAQAS lexical recall, and expansion mining + reranker ordering.",
    version="2.0.0",
)

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://nonvariable-serriedly-wynona.ngrok-free.dev"
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
    return {"status": "ok", "message": "API is running"}

@app.post("/signup")
def signup(req: SignupRequest):
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    conn.close()
    return {"status": "ok", "message": "Account created successfully"}

@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    if not verify_recaptcha(req.recaptcha_token):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="reCAPTCHA verification failed")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (req.username,))
    row = cur.fetchone()
    conn.close()

    if not row or not verify_password(req.password, row[0]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    access_token = create_access_token(data={"sub": req.username})
    return LoginResponse(access_token=access_token)

@app.post("/google-login", response_model=LoginResponse)
def google_login(req: GoogleLoginRequest):
    if not req.id_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing Google ID token")

    try:
        idinfo = google_id_token.verify_oauth2_token(
            req.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Google token")

    email = idinfo.get("email")
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google account has no email")

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
    user = get_current_user(authorization=current_user)

    results, info = search_api(
        req.query,
        page=req.page,
        page_size=req.page_size,
        k_faiss=req.k_faiss,
        top_expansions=req.top_expansions,
        rerank_batch=32
    )

    cleaned = [{
        "rank": r["rank"],
        "ref": r["ref"],
        "arabic": r["arabic"],
        "english": r["english"],
    } for r in results]

    verse_results = [VerseResult(**r) for r in cleaned]
    return {
        "query": req.query,
        "total": info["total"],
        "page": info["page"],
        "page_size": info["page_size"],
        "results": verse_results
    }


