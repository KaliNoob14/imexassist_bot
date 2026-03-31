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
        self.project_id = 1016106875965
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

    async def generate_text(self, prompt: str) -> str:
        """
        Async-friendly wrapper for Vertex Model Garden calls.
        A full prompt/response strategy can be added in the next implementation step.
        """
        self._ensure_initialized()
        if self._model is None:
            return ""
        response = await run_in_threadpool(self._model.generate_content, prompt)
        return getattr(response, "text", "")
