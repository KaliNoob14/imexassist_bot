from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.services.message_handler import MessageHandler
from app.services.vertex_llama_client import VertexLlamaClient

router = APIRouter()
message_handler = MessageHandler(llm_client=VertexLlamaClient())


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
    payload = await request.json()
    return await message_handler.handle(payload)
