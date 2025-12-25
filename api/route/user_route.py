from fastapi import APIRouter, HTTPException, status, Query
from typing import List

from api.schema.user import UserCreate, UserRead, UserUpdate
from api.service import user_service

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate):
    existing = user_service.get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    return user_service.create_user(payload)


@router.get("", response_model=List[UserRead])
def list_users(limit: int = Query(100, ge=1, le=1000), offset: int = 0):
    return user_service.list_users(limit=limit, offset=offset)


@router.get("/{user_id}", response_model=UserRead)
def read_user(user_id: int):
    u = user_service.get_user_by_id(user_id)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return u


@router.put("/{user_id}", response_model=UserRead)
def update_user(user_id: int, payload: UserUpdate):
    u = user_service.update_user(user_id, payload)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return u


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int):
    ok = user_service.delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return None
