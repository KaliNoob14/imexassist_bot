from typing import Any, Dict

import requests
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.services.vertex_llama_client import VertexLlamaClient


class MessageHandler:
    def __init__(self, llm_client: VertexLlamaClient) -> None:
        self.llm_client = llm_client

    async def handle(self, webhook_payload: Dict[str, Any]) -> Dict[str, Any]:
        sent_count = 0
        for entry in webhook_payload.get("entry", []):
            for messaging_event in entry.get("messaging", []):
                sender_id = (messaging_event.get("sender") or {}).get("id")
                message_text = ((messaging_event.get("message") or {}).get("text") or "").strip()

                if not sender_id or not message_text:
                    continue

                ai_response = await self.llm_client.get_response(message_text)
                if ai_response:
                    await self._send_facebook_message(sender_id, ai_response)
                    sent_count += 1

        return {"status": "processed", "messages_sent": sent_count}

    async def _send_facebook_message(self, recipient_id: str, text: str) -> None:
        url = "https://graph.facebook.com/v19.0/me/messages"
        payload = {"recipient": {"id": recipient_id}, "message": {"text": text}}
        params = {"access_token": settings.page_access_token}

        response = await run_in_threadpool(
            requests.post,
            url,
            params=params,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
