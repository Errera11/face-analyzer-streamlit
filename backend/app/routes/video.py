from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from backend.app.services.video_processor import VideoProcessor
import io

router = APIRouter()
service = VideoProcessor()

@router.post("/process")
async def process(video: UploadFile = File(...)):
    content = await video.read()
    result = service.process_video(content)

    return StreamingResponse(
        result,
        media_type="video/mp4"
    )
