import os
import logging
import requests
from typing import Dict, Any

from api.service import user_service

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = os.getenv("TELEGRAM_API_URL", "https://api.telegram.org")


def _bot_url(method: str) -> str:
    return f"{TELEGRAM_API_URL}/bot{TELEGRAM_TOKEN}/{method}"


def send_message_to_chat(chat_id: int, text: str) -> bool:
    if not TELEGRAM_TOKEN:
        logger.error("send_message_to_chat: TELEGRAM_BOT_TOKEN not configured")
        return False
    try:
        resp = requests.post(_bot_url("sendMessage"), json={"chat_id": chat_id, "text": text})
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.exception("Failed to send telegram message: %s", e)
        return False


def process_update(update: Dict[str, Any]) -> None:
    """
    Process incoming Telegram webhook update and link chat_id to user when /start <user_id> is received.
    """
    try:
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        text = msg.get("text", "") or ""
        if not text:
            return
        # handle /start payload: either "/start <payload>" or "/start<payload>"
        if text.startswith("/start"):
            parts = text.split(None, 1)
            payload = parts[1] if len(parts) > 1 else text[len("/start"):].strip()
            if payload:
                try:
                    user_id = int(payload)
                    updated = user_service.set_user_chat_id(user_id, str(chat_id))
                    if updated:
                        send_message_to_chat(chat_id, "✅ Alerts enabled. You will receive notifications for theft.")
                    else:
                        send_message_to_chat(chat_id, "⚠️ Could not link your account (user not found).")
                except Exception:
                    send_message_to_chat(chat_id, "⚠️ Invalid link token.")
            else:
                send_message_to_chat(chat_id, "To enable alerts, open the bot from the app link provided and include your link token.")
    except Exception:
        logger.exception("process_update failed")
