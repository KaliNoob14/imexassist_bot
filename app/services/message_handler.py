import os
from typing import Any, Dict

import requests
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool

from app.services.vertex_llama_client import VertexLlamaClient

load_dotenv()
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
if not PAGE_ACCESS_TOKEN:
    print("CRITICAL: PAGE_ACCESS_TOKEN is undefined!")


class MessageHandler:
    def __init__(self, llm_client: VertexLlamaClient) -> None:
        self.llm_client = llm_client

    async def handle(self, webhook_payload: Dict[str, Any]) -> Dict[str, Any]:
        sent_count = 0
        try:
            entries = webhook_payload.get("entry", [])
            if not isinstance(entries, list):
                return {"status": "processed", "messages_sent": 0}

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                messaging_items = entry.get("messaging", [])
                if not isinstance(messaging_items, list):
                    continue

                for messaging_event in messaging_items:
                    if not isinstance(messaging_event, dict):
                        continue

                    sender_id = (messaging_event.get("sender") or {}).get("id")
                    message_text = ((messaging_event.get("message") or {}).get("text") or "").strip()
                    if not sender_id or not message_text:
                        continue

                    print(f"DEBUG: Processing message from {sender_id}")
                    ai_response = await self.llm_client.get_response(message_text)
                    if ai_response:
                        print(f"DEBUG: AI response to send: {ai_response}")
                        await self._send_facebook_message(sender_id, ai_response)
                        sent_count += 1
        except Exception as exc:
            print(f"Message handler error: {type(exc).__name__}: {exc}")

        return {"status": "processed", "messages_sent": sent_count}

    async def _send_facebook_message(self, recipient_id: str, text: str) -> None:
        url = "https://graph.facebook.com/v19.0/me/messages"
        payload = {"recipient": {"id": recipient_id}, "message": {"text": text}}
        params = {"access_token": PAGE_ACCESS_TOKEN}

        if not PAGE_ACCESS_TOKEN:
            print("CRITICAL: PAGE_ACCESS_TOKEN is undefined!")
            return

        fb_response = await run_in_threadpool(
            requests.post,
            url,
            params=params,
            json=payload,
            timeout=15,
        )
        print(f"Meta Response: {fb_response.status_code} - {fb_response.text}")
        print(fb_response.text)
        fb_response.raise_for_status()
