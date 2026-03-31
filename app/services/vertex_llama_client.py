from typing import Any, Optional

import vertexai
from google.api_core import exceptions
from fastapi.concurrency import run_in_threadpool
from vertexai.generative_models import GenerativeModel

from app.config import settings


class VertexLlamaClient:
    def __init__(
        self,
        project_id: str = settings.gcp_project_id,
        region: str = settings.gcp_region,
        model_name: str = settings.vertex_model_name,
    ) -> None:
        self.project_id = project_id
        # Llama MaaS models are most stable through us-central1.
        self.region = "us-central1"
        self.model_name = "publishers/meta/models/llama3-3-70b-instruct-maas"
        self._initialized = False
        self._model: Optional[Any] = None

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        vertexai.init(project=self.project_id, location=self.region)
        self._model = GenerativeModel(self.model_name)
        self._initialized = True

    async def get_response(self, user_message: str) -> str:
        """
        Generate a concise assistant response with a base IMEX persona prompt.
        """
        self._ensure_initialized()
        if self._model is None:
            return ""
        prompt = (
            "You are the IMEX Digital Assistant for IMEX MCE MBT community management. "
            "Be helpful, concise, and professional. If context is missing, ask a brief "
            "clarifying question.\n\n"
            f"User message: {user_message}\n"
            "Assistant response:"
        )
        try:
            response = await run_in_threadpool(self._model.generate_content, prompt)
        except (exceptions.InternalServerError, exceptions.ServiceUnavailable) as exc:
            print(f"Vertex AI MaaS prediction error: {type(exc).__name__}: {exc}")
            return "I'm thinking a bit too hard, try again in a second"
        return getattr(response, "text", "").strip() or (
            "Thanks for your message. How can I help you with IMEX today?"
        )
