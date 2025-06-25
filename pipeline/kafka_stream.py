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
            "idx": str(uuid.uuid4()),  # convert key to UUID
            "url": value["url"],
            "label": value["label"],
            "video_encoded": get_encoded_video(value["url"]),
            "text_embedding": [],
            "split" : ""
        }

def stream_data():
    try:
        producer = KafkaProducer(bootstrap_servers=['localhost:9092'], max_block_ms=5000)


    except Exception as e:
        logging.error(f'Kafka connection failed: {e}')
        return

    curr_time = time.time()
    j = 0
    for res in get_data():
        if time.time() > curr_time + 120:
            break
        try:
            producer.send('users_created', json.dumps(res).encode('utf-8'))
            print(f"Sent data: {j}")
            j += 1
            time.sleep(1)  # optional delay
        except Exception as e:
            logging.error(f'An error occurred while sending: {e}')
            continue

stream_data()
