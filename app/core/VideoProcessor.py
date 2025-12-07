from typing import List
import io
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path
import imageio.v2 as imageio
import imageio.plugins.ffmpeg
import cv2

Face_Rect_Type = tuple[int, int, int, int]

class VideoProcessor:
    def __init__(self):
        config_path = str(Path(
            __file__).parent.resolve() / "model" / "haarcascade_frontalface_default.xml")

        self.face_cascade_classifier = cv2.CascadeClassifier(config_path)

        self.processor_emotion = AutoImageProcessor.from_pretrained(
            "dima806/facial_emotions_image_detection", use_fast=True)
        self.model_emotion = AutoModelForImageClassification.from_pretrained(
            "dima806/facial_emotions_image_detection")

        self.processor = AutoImageProcessor.from_pretrained(
            "nateraw/vit-age-classifier", use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(
            "nateraw/vit-age-classifier")
    def process(self, video: bytearray):
        result_buffer = self.process_video(video)
        return result_buffer

    def get_cropped_face(self, img, pos: Face_Rect_Type):
        [x, y, w, h] = pos
        cropped_face = img[y: y + h, x: x + w]
        return cropped_face

    def draw_face_rect(self, img, pos: Face_Rect_Type):
        x, y, w, h = pos
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    def alter_with_text(self, img, text, org: tuple[int, int]):
        pos = org
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(img, text, pos, font_face, font_scale, color, thickness)

    def predict_age(self, face_img):
        inputs = self.processor(face_img, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        age = self.model.config.id2label[predicted_label]
        return age

    def predict_emotion(self, face_img):
        inputs = self.processor_emotion(face_img, return_tensors="pt")
        with torch.no_grad():
            logits = self.model_emotion(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        emotion = self.model_emotion.config.id2label[predicted_label]
        return emotion

    def detect_faces(self, img) -> List[Face_Rect_Type]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_rect = self.face_cascade_classifier.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return list(face_rect)

    def draw_rects(self, img, face_rects):
        processed_img = img.copy()

        for pos in face_rects:
            croppped_face = self.get_cropped_face(img, pos)

            age = self.predict_age(croppped_face)
            emotion = self.predict_emotion(croppped_face)

            [x, y, w, h] = pos
            age_text_pos = (x, y + h + 20)
            emotion_text_pos = (x, y + h + 40)
            self.draw_face_rect(processed_img, pos)
            self.alter_with_text(processed_img, f"Age: {age}", age_text_pos)
            self.alter_with_text(processed_img, f"Emotion: {emotion}", emotion_text_pos)

        return processed_img

    def process_video(self, video: bytearray):
        print('Starting video processing...')

        video_content = video
        video_buffer = io.BytesIO(video_content)

        try:
            reader = imageio.get_reader(video_buffer, format=".mp4", mode="I")
        except Exception as e:
            raise ValueError(f"Could not open video stream with imageio: {e}")

        meta_data = reader.get_meta_data()
        fps = meta_data.get('fps', 30.0)

        output_buffer = io.BytesIO()
        writer = imageio.get_writer(output_buffer, fps=fps,
                                    format='.mp4',
                                     quality=8)

        FRAME_SKIP_INTERVAL = max(1, int(fps))  # Process models ~once per second

        last_face_detections = None
        for i, frame in enumerate(reader, start=0):
            if i % FRAME_SKIP_INTERVAL == 0:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                face_rects = self.detect_faces(frame_bgr)
                last_face_detections = face_rects
            else:
                face_rects = last_face_detections
            if face_rects is not None:
                processed_frame = self.draw_rects(frame, face_rects)
                writer.append_data(processed_frame)

        writer.close()
        reader.close()

        output_buffer.seek(0)

        print('Finished video processing...')

        return output_buffer

