import asyncio
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.services.llama_client import LlamaClient
from app.services.message_handler import MessageHandler

router = APIRouter()
message_handler = MessageHandler(llm_client=LlamaClient())


async def _process_payload_safe(payload: Dict[str, Any]) -> None:
    try:
        await message_handler.handle(payload)
    except Exception as exc:
        print(f"Webhook background processing error: {type(exc).__name__}: {exc}")


@router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(..., alias="hub.mode"),
    hub_verify_token: str = Query(..., alias="hub.verify_token"),
    hub_challenge: str = Query(..., alias="hub.challenge"),
) -> PlainTextResponse:
    if hub_mode == "subscribe" and hub_verify_token == settings.fb_verify_token:
        return PlainTextResponse(content=hub_challenge)
    raise HTTPException(status_code=403, detail="Webhook verification failed")


@router.post("/webhook")
async def receive_webhook(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            print("DEBUG: Invalid webhook payload type received")
            return {"status": "ok", "processed": False}
        # Return 200 immediately to avoid Facebook retry duplicates.
        asyncio.create_task(_process_payload_safe(payload))
        return {"status": "ok", "processed": True}
    except Exception as exc:
        # Always return 200 so Meta does not retry flood on transient failures.
        print(f"Webhook processing error: {type(exc).__name__}: {exc}")
        return {"status": "ok", "processed": False}
