"""Database configuration and models for the API.

Uses SQLAlchemy and reads `DATABASE_URL` (or `SD_DATABASE_URL`) from environment.
Falls back to a local SQLite DB at `sur.db` for convenience.
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime


DATABASE_URL = os.getenv("DATABASE_URL", os.getenv("SD_DATABASE_URL", "sqlite:///./sur.db"))

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    disabled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Telegram chat id saved when user links their bot chat
    chat_id = Column(String, nullable=True, index=True)


def init_db() -> None:
    """Create tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
