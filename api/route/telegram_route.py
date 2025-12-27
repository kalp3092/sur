from fastapi import APIRouter, Request, HTTPException
import os
import logging
from typing import Dict

from api.service import telegram_service
from api.service import user_service

router = APIRouter()
logger = logging.getLogger(__name__)

WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")


@router.post("/telegram/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if not WEBHOOK_SECRET or secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")
    payload: Dict = await request.json()
    telegram_service.process_update(payload)
    return {"ok": True}


@router.post("/internal/theft-detected")
async def internal_theft_detected(payload: Dict):
    """
    Internal endpoint to notify user of a theft.
    Expects JSON: {"user_id": int, "message": "..." }
    """
    user_id = payload.get("user_id")
    text = payload.get("message", "Theft detected")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    u = user_service.get_user_by_id(user_id)
    if not u or not getattr(u, "chat_id", None):
        raise HTTPException(status_code=404, detail="User or chat_id not found")
    ok = telegram_service.send_message_to_chat(int(u.chat_id), text)
    return {"sent": ok}
