from fastapi import FastAPI

from backend.app.routes.video import router as video_router
from backend.app.routes.health import router as health_router

app = FastAPI()

app.include_router(
    video_router,
    tags=["AI"]
)

app.include_router(health_router)