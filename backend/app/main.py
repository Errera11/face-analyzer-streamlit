from fastapi import FastAPI

from routes.video import router as video_router
from routes.health import router as health_router

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    video_router,
    tags=["AI"]
)

app.include_router(health_router)