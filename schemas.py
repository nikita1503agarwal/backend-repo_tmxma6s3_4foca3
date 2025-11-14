"""
Database Schemas for the E-Waste Mapping Platform

Each Pydantic model corresponds to a MongoDB collection (lowercased class name).
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Literal
from datetime import datetime

Role = Literal["user", "recycler", "admin"]

class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    password_hash: str = Field(..., description="BCrypt hash")
    role: Role = Field("user", description="Access role")
    provider: Literal["email", "google"] = Field("email")
    photo_url: Optional[str] = None
    is_active: bool = True

class GeoPoint(BaseModel):
    lat: float
    lng: float

class Report(BaseModel):
    user_id: Optional[str] = Field(None, description="Reporter user id")
    image_url: str
    thumb_url: Optional[str] = None
    description: Optional[str] = None
    location: GeoPoint
    address: Optional[str] = None
    category: Optional[str] = Field(None, description="Predicted category")
    confidence: Optional[float] = None
    duplicate_of: Optional[str] = None
    status: Literal["pending", "approved", "rejected", "collected"] = "pending"
    tags: List[str] = []
    risk_score: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class Center(BaseModel):
    name: str
    type: Literal["collection", "recycler", "repair"]
    location: GeoPoint
    address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    hours: Optional[str] = None

class PickupRouteRequest(BaseModel):
    start: GeoPoint
    stops: List[GeoPoint]
