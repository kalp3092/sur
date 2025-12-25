from typing import Optional
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserRead(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str] = None
    disabled: bool = False
    created_at: Optional[str] = None

    class Config:
        orm_mode = True


class UserUpdate(BaseModel):
    password: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
