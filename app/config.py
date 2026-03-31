import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "imexassist-bot")
    fb_page_token: str = os.getenv("FB_PAGE_TOKEN", "")
    fb_verify_token: str = os.getenv("FB_VERIFY_TOKEN", "IMEX_SECRET_2026")
    gcp_project_id: str = os.getenv("GCP_PROJECT_ID", "imexassist-bot")
    gcp_region: str = os.getenv("GCP_REGION", "europe-west1")
    vertex_model_name: str = os.getenv(
        "VERTEX_MODEL_NAME", "meta/llama-3-70b-instruct-maas"
    )


settings = Settings()
