from typing import List, Optional
from sqlalchemy.orm import Session
import hashlib

from api.configs.db import User, SessionLocal
from api.schema.user import UserCreate, UserRead, UserUpdate


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(payload: UserCreate) -> UserRead:
    db: Session = SessionLocal()
    try:
        u = User(email=payload.email, full_name=payload.full_name, hashed_password=_hash_password(payload.password))
        db.add(u)
        db.commit()
        db.refresh(u)
        return UserRead.from_orm(u)
    finally:
        db.close()


def get_user_by_id(user_id: int) -> Optional[UserRead]:
    db: Session = SessionLocal()
    try:
        u = db.query(User).filter(User.id == user_id).first()
        return UserRead.from_orm(u) if u else None
    finally:
        db.close()


def get_user_by_email(email: str) -> Optional[UserRead]:
    db: Session = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email).first()
        return UserRead.from_orm(u) if u else None
    finally:
        db.close()


def list_users(limit: int = 100, offset: int = 0) -> List[UserRead]:
    db: Session = SessionLocal()
    try:
        rows = db.query(User).order_by(User.id.desc()).limit(limit).offset(offset).all()
        return [UserRead.from_orm(r) for r in rows]
    finally:
        db.close()


def update_user(user_id: int, payload: UserUpdate) -> Optional[UserRead]:
    db: Session = SessionLocal()
    try:
        u = db.query(User).filter(User.id == user_id).first()
        if not u:
            return None
        if payload.full_name is not None:
            u.full_name = payload.full_name
        if payload.password is not None:
            u.hashed_password = _hash_password(payload.password)
        if payload.disabled is not None:
            u.disabled = payload.disabled
        db.add(u)
        db.commit()
        db.refresh(u)
        return UserRead.from_orm(u)
    finally:
        db.close()


def delete_user(user_id: int) -> bool:
    db: Session = SessionLocal()
    try:
        r = db.query(User).filter(User.id == user_id).delete()
        db.commit()
        return r > 0
    finally:
        db.close()
