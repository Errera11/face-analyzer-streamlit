import streamlit as st

from core.VideoProcessor import VideoProcessor

video_processor = VideoProcessor()

st.set_page_config(
    page_title="Face analyzer",
    layout="wide",
)

uploaded_video = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_video is not None:
    video_bytes = uploaded_video.read()

    processed_video_bytes = video_processor.process_video(video_bytes)

    st.write(f"Uploaded video: ")
    st.video(video_bytes)

    st.write(f"Processed video: ")
    st.video(processed_video_bytes)
else:
    st.info("Please upload a video file.")

