from fastapi import FastAPI

from app.routes.webhook import router as webhook_router


def create_app() -> FastAPI:
    app = FastAPI(title="imexassist-bot", version="0.1.0")
    app.include_router(webhook_router)
    return app


app = create_app()
