from typing import Any, Dict

from pydantic import BaseModel, Field


class WebhookEvent(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict)
