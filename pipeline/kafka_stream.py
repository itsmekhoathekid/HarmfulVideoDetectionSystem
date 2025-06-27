import json
from kafka import KafkaProducer
import time
import logging
from utils import load_json
from feature_extractor import PretrainedModelLoader, VideoProcessor
import uuid
import base64

def get_encoded_video(video_path):
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode('utf-8')
    return video_b64

def get_data():
    json_data = load_json("/home/anhkhoa/spark_video_streaming/json/train.json")
    for key, value in json_data.items():
        yield {
            "idx": str(uuid.uuid4()),
            "url": value["url"],
            "label": value["label"],
            # "video_encoded": get_encoded_video(value["url"]),
            "text_embedding": [0.0],
            "split": ""
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
