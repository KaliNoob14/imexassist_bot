from typing import Optional

import vertexai
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
        self.region = region
        self.model_name = model_name
        self._initialized = False
        self._model: Optional[GenerativeModel] = None

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
        response = await run_in_threadpool(self._model.generate_content, prompt)
        return getattr(response, "text", "").strip() or (
            "Thanks for your message. How can I help you with IMEX today?"
        )
