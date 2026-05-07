import streamlit as st
import requests
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")

st.set_page_config(page_title="Video AI Processor", layout="centered")

st.title("🎥 AI Video Processor")
st.write("Upload a video and process it via FastAPI backend.")

video_file = st.file_uploader(
    "Upload video",
    type=["mp4", "mov", "avi", "mkv"]
)

if video_file is not None:

    st.video(video_file)

    if st.button("Process Video 🚀"):

        with st.spinner("Processing video..."):

            files = {
                "video": (
                    video_file.name,
                    video_file,
                    video_file.type
                )
            }

            response = requests.post(
                API_URL,
                files=files
            )

        if response.status_code == 200:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as tmp_file:

                tmp_file.write(response.content)
                tmp_path = tmp_file.name

            st.success("Done!")

            st.video(tmp_path)

        else:
            st.error(f"Error: {response.status_code}")
            st.text(response.text)