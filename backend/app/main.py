from fastapi import FastAPI

from routes.video import router as video_router
from routes.health import router as health_router

app = FastAPI()

app.include_router(
    video_router,
    tags=["AI"]
)

app.include_router(health_router)