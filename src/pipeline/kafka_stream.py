import json
from kafka import KafkaProducer
import time
import logging
from utils import load_json
from feature_extractor import PretrainedModelLoader, VideoProcessor
import uuid
import base64
import tempfile
from moviepy.editor import VideoFileClip
import torch
from transformers import AutoTokenizer, AutoModel
import whisper
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# whisper_model = whisper.load_model("large", device=device)
# tokenizer = AutoTokenizer.from_pretrained("phobert-base")
# text_model = AutoModel.from_pretrained("phobert-base").to(device).eval()

def get_encoded_video(video_path):
    try:
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")

        with open(video_path, 'rb') as f:
            video_bytes = f.read()
            if not video_bytes:
                raise ValueError("File is empty")

            video_b64 = base64.b64encode(video_bytes).decode('utf-8')
            return video_b64
    except Exception as e:
        logging.error(f"❌ Failed to encode video {video_path}: {e}")
        return None



def extract_text_embedding(video_path, whisper_model, tokenizer, text_model):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio_path = tmpfile.name

    video = VideoFileClip(video_path)
    audio = video.audio
    if audio is None:
        raise ValueError(f"No audio track found in {video_path}")

    
    audio.write_audiofile(audio_path, logger=None)

    result = whisper_model.transcribe(audio_path, language='vi')  
    text = result['text']

    if text != "":
        inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128).to(device)
        with torch.no_grad():
            text_output = text_model(**inputs)
            text_embedding = text_output.last_hidden_state[:, 0, :].squeeze(0).cpu()
    else:
        text_embedding = torch.zeros(text_model.config.hidden_size)
    return text_embedding.tolist()

def get_data():
    json_data = load_json("/home/anhkhoa/spark_video_streaming/json/data.json")
    for key, value in json_data.items():
        yield {
            "idx": str(uuid.uuid4()),
            "url": value["url"],
            "label": value["label"],
            # "video_encoded": get_encoded_video(value["url"]),
            "text_embedding": value["text_embedding"],
            "split": value["flag"]
        }

def stream_data():
    try:
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'], 
            acks='all',
            max_block_ms=5000,
            max_request_size=15 * 1024 * 1024  # ✅ 15 MB
        )
    except Exception as e:
        logging.error(f'Kafka connection failed: {e}')
        return

    j = 0
    for res in get_data():
        try:
            future = producer.send('users_created', json.dumps(res).encode('utf-8'))
            future.get(timeout=10)  # Chờ xác nhận gửi thành công
            print(f"✅ Sent data: {j}")
        except Exception as send_err:
            logging.error(f"❌ Failed to send message {j}: {send_err}")
        j += 1
        time.sleep(1)

    producer.flush()
    producer.close()
    print(f"🚀 Finished sending {j} messages.")

stream_data()
