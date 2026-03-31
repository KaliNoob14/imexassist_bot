from typing import Any, Dict

from app.services.vertex_llama_client import VertexLlamaClient


class MessageHandler:
    def __init__(self, llm_client: VertexLlamaClient) -> None:
        self.llm_client = llm_client

    async def handle(self, webhook_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse incoming FB webhook payload and prepare bot response flow.
        This currently returns an acknowledgement while keeping the integration
        contract ready for a Vertex Llama call in the next step.
        """
        _ = webhook_payload
        return {
            "status": "accepted",
            "message": "Webhook received. Llama integration stub is ready.",
        }
