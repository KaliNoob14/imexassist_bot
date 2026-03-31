from typing import Any, Optional

import grpc
import vertexai
from google.api_core import exceptions
from fastapi.concurrency import run_in_threadpool
from vertexai.generative_models import GenerativeModel

from app.config import settings

LOCATION = "us-central1"
API_ENDPOINT = "us-central1-aiplatform.googleapis.com"
vertexai.init(
    project="imexassist-bot", location=LOCATION, api_endpoint=API_ENDPOINT
)


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
        self.model_id = "meta/llama-3.3-70b-instruct-maas"
        self.system_instruction = (
            "You are the IMEX Digital Assistant, a professional logistics expert. "
            "You are helpful, concise, and represent the IMEX brand with a touch "
            "of modern efficiency."
        )
        self._initialized = False
        self._model: Optional[Any] = None

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._model = GenerativeModel(
            "meta/llama-3.3-70b-instruct-maas",
            system_instruction=self.system_instruction,
        )
        print(f"SUCCESS: Connected to Llama 3.3 MaaS in {LOCATION}")
        self._initialized = True

    async def get_response(self, user_message: str) -> str:
        """
        Generate a concise assistant response with a base IMEX persona prompt.
        """
        self._ensure_initialized()
        if self._model is None:
            return ""
        try:
            response = await run_in_threadpool(self._model.generate_content, user_message)
        except (exceptions.NotFound, grpc.RpcError) as e:
            print(f"CRITICAL - Vertex Error: {type(e).__name__} - {str(e)}")
            return (
                "The IMEX assistant is currently processing a high volume of requests. "
                "Please try your message again in a moment."
            )
        return getattr(response, "text", "").strip() or (
            "Thanks for your message. How can I help you with IMEX today?"
        )
