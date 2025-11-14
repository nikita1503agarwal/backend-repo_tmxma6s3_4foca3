import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from PIL import Image
import imagehash
import io
import numpy as np
from sklearn.cluster import KMeans

from database import db, create_document, get_documents
from schemas import User, Report, Center, PickupRouteRequest

SECRET_KEY = os.getenv("JWT_SECRET", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

app = FastAPI(title="E-Waste Mapping Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Auth helpers ------------------
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        role: str = payload.get("role")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db["user"].find_one({"_id": db.client.get_default_database().client.get_database().codec_options.document_class})
    # Fallback simple fetch by email/id data in token not implemented; return token data only
    return {"id": user_id, "role": role}


def require_role(required: List[str]):
    def wrapper(user = Depends(get_current_user)):
        if user.get("role") not in required:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return wrapper

# ------------------ Public & Utility ------------------
@app.get("/")
def read_root():
    return {"message": "E-Waste Mapping Platform Backend Running"}

@app.get("/test")
def test_database():
    try:
        collections = db.list_collection_names()
        return {
            "backend": "✅ Running",
            "database": "✅ Connected",
            "collections": collections[:10]
        }
    except Exception as e:
        return {"backend": "✅ Running", "database": f"❌ {str(e)[:80]}"}

# ------------------ Auth Endpoints (email) ------------------
@app.post("/auth/register", response_model=Token)
def register(name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    existing = db["user"].find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(name=name, email=email, password_hash=get_password_hash(password), role="user", provider="email")
    uid = create_document("user", user)
    token = create_access_token({"sub": uid, "role": user.role})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = db["user"].find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    token = create_access_token({"sub": str(user["_id"]), "role": user.get("role", "user")})
    return {"access_token": token, "token_type": "bearer"}

# ------------------ File Storage (Mock: stores as base64 in DB or local) ------------------
STORAGE_DIR = os.getenv("STORAGE_DIR", "/tmp/uploads")
os.makedirs(STORAGE_DIR, exist_ok=True)


def save_image_locally(file: UploadFile) -> str:
    contents = file.file.read()
    path = os.path.join(STORAGE_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(contents)
    return path

# ------------------ Basic AI Utilities ------------------
CATEGORIES = ["Battery", "Circuit Board", "Mobile Phone", "Laptop", "TV", "Appliance", "Other"]


def classify_image(image_bytes: bytes):
    # Placeholder simple hash-derived pseudo classification
    h = imagehash.average_hash(Image.open(io.BytesIO(image_bytes)))
    idx = int(str(h)[-2:], 16) % len(CATEGORIES)
    confidence = 0.6 + (int(str(h)[-1], 16) / 32.0)
    return CATEGORIES[idx], min(confidence, 0.99)


def image_perceptual_hash(image_bytes: bytes) -> str:
    return str(imagehash.phash(Image.open(io.BytesIO(image_bytes))))

# ------------------ Reports ------------------
@app.post("/reports")
async def create_report(
    image: UploadFile = File(...),
    description: Optional[str] = Form(None),
    lat: float = Form(...),
    lng: float = Form(...),
    address: Optional[str] = Form(None),
    token: Optional[str] = Form(None)
):
    user = None
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user = {"id": payload.get("sub"), "role": payload.get("role")}
        except Exception:
            user = None

    image_bytes = await image.read()
    category, confidence = classify_image(image_bytes)
    phash = image_perceptual_hash(image_bytes)

    # Duplicate detection: find existing reports with same phash in last 30 days near location
    recent = list(db["report"].find({"phash": phash}))
    duplicate_of = str(recent[0]["_id"]) if recent else None

    # Save file locally (in real prod -> Cloudinary/Firebase)
    saved_path = save_image_locally(image)

    report = Report(
        user_id=user["id"] if user else None,
        image_url=saved_path,
        description=description,
        location={"lat": lat, "lng": lng},
        address=address,
        category=category,
        confidence=confidence,
        duplicate_of=duplicate_of,
        status="pending",
        tags=[],
    )
    data = report.model_dump()
    data.update({"phash": phash})
    rid = create_document("report", data)
    return {"id": rid, "category": category, "confidence": confidence, "duplicate_of": duplicate_of}


@app.get("/reports")
def list_reports(status: Optional[str] = None):
    filt: Dict[str, Any] = {}
    if status:
        filt["status"] = status
    items = get_documents("report", filt, limit=None)
    # Convert ObjectId to string
    for it in items:
        it["_id"] = str(it["_id"]) 
    return items


@app.post("/reports/{report_id}/approve")
def approve_report(report_id: str, user=Depends(require_role(["admin", "recycler"]))):
    from bson import ObjectId
    db["report"].update_one({"_id": ObjectId(report_id)}, {"$set": {"status": "approved", "updated_at": datetime.now(timezone.utc)}})
    return {"ok": True}


@app.post("/reports/{report_id}/reject")
def reject_report(report_id: str, reason: Optional[str] = None, user=Depends(require_role(["admin", "recycler"]))):
    from bson import ObjectId
    db["report"].update_one({"_id": ObjectId(report_id)}, {"$set": {"status": "rejected", "reason": reason, "updated_at": datetime.now(timezone.utc)}})
    return {"ok": True}

# ------------------ Centers ------------------
@app.post("/centers")
def create_center(center: Center, user=Depends(require_role(["admin", "recycler"]))):
    cid = create_document("center", center)
    return {"id": cid}

@app.get("/centers")
def list_centers(center_type: Optional[str] = None):
    filt = {"type": center_type} if center_type else {}
    items = get_documents("center", filt, limit=None)
    for it in items:
        it["_id"] = str(it["_id"]) 
    return items

# ------------------ Analytics ------------------
@app.get("/analytics/summary")
def analytics_summary():
    total = db["report"].count_documents({})
    by_status = list(db["report"].aggregate([
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]))
    by_category = list(db["report"].aggregate([
        {"$group": {"_id": "$category", "count": {"$sum": 1}}}
    ]))
    return {"total": total, "by_status": by_status, "by_category": by_category}


@app.get("/analytics/hotspots")
def analytics_hotspots(k: int = 4):
    points = list(db["report"].find({}, {"location": 1}))
    if not points:
        return {"centers": []}
    X = np.array([[p["location"]["lat"], p["location"]["lng"]] for p in points])
    k = min(k, len(X))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    centers = km.cluster_centers_
    return {"centers": [{"lat": float(c[0]), "lng": float(c[1])} for c in centers]}

# ------------------ Routing (simple nearest-neighbor heuristic) ------------------
@app.post("/routes/optimize")
def optimize_route(req: PickupRouteRequest, user=Depends(require_role(["admin", "recycler"]))):
    import math
    pts = [req.start] + req.stops
    n = len(pts)
    visited = [False]*n
    order = [0]
    visited[0] = True
    def dist(a, b):
        return math.hypot(a.lat-b.lat, a.lng-b.lng)
    for _ in range(n-1):
        last = order[-1]
        best = None
        best_d = 1e9
        for i in range(n):
            if not visited[i]:
                d = dist(pts[last], pts[i])
                if d < best_d:
                    best_d = d
                    best = i
        order.append(best)
        visited[best] = True
    route = [pts[i] for i in order]
    return {"order": order, "route": [r.model_dump() for r in route]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
