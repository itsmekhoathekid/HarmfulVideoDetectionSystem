from kafka import KafkaProducer
from json import dumps
from time import sleep
import json
import os
import base64


topic_name = 'quickstart-topic'
kafka_server = 'localhost:9092'
producer = KafkaProducer(
    bootstrap_servers=kafka_server,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_videos(folder_path):
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            if not filename.endswith('.mp4'):
                continue
            full_path = os.path.join(label_path, filename)
            with open(full_path, 'rb') as f:
                video_bytes = f.read()
                video_b64 = base64.b64encode(video_bytes).decode('utf-8')
                message = {
                    "label": label,
                    "video_name": filename,
                    "video_path": full_path,
                    "video_data": video_b64
                }
                producer.send(topic_name, value=message)
                print(f"Sent {filename} with label {label}")
                sleep(5)

if __name__ == "__main__":
    send_videos("D:\\bigdata_dataset\\home\\anhkhoa\\spark_video_streaming\\data\\test")