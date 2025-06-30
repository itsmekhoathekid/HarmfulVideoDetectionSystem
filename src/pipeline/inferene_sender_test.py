from kafka import KafkaProducer
from json import dumps
from time import sleep
import json
import os
import base64
import torch

topic_name = 'data-test'
kafka_server = 'localhost:9092'
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'], 
    acks='all',
    max_block_ms=5000,
    max_request_size=15 * 1024 * 1024  # ✅ 15 MB
)


NEW_ROOT = "/home/anhkhoa/spark_video_streaming/json/train.json"

# Đọc file JSON
with open(NEW_ROOT, "r") as f:
    data = json.load(f)

# Gửi từng video
for key, value in data.items():


    label = value["label"]
    full_path = value["url"]  
    
    try:
        with open(full_path, 'rb') as f:
            video_bytes = f.read()
            video_b64 = base64.b64encode(video_bytes).decode('utf-8')

        # Gửi message
        message = {
            "label": label,
            "video_name": "video 1",
            "video_path": full_path,
            "video_data": video_b64,
            "text_embedding": torch.zeros(768).tolist(),  # Giả sử text_embedding là tensor rỗng
        }

        producer.send(topic_name, value=json.dumps(message).encode('utf-8'))

        print(f"✅ Sent: video 1 with label '{label}'")

    except FileNotFoundError:
        print(f"❌ File not found: {full_path}")

    sleep(5)

