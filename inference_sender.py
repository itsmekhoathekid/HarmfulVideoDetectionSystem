from kafka import KafkaProducer
from json import dumps
from time import sleep
import json
import os
import base64

topic_name = 'data-test'
kafka_server = 'localhost:9092'
producer = KafkaProducer(
    bootstrap_servers=kafka_server,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
#
# def send_videos(folder_path):
#     for label in os.listdir(folder_path):
#         label_path = os.path.join(folder_path, label)
#         if not os.path.isdir(label_path):
#             continue
#         for filename in os.listdir(label_path):
#             if not filename.endswith('.mp4'):
#                 continue
#             full_path = os.path.join(label_path, filename)
#             with open(full_path, 'rb') as f:
#                 video_bytes = f.read()
#                 video_b64 = base64.b64encode(video_bytes).decode('utf-8')
#                 message = {
#                     "label": label,
#                     "video_name": filename,
#                     "video_path": full_path,
#                     "video_data": video_b64,
#                     "text_embedding": [],
#                 }
#                 producer.send(topic_name, value=message)
#                 print(f"Sent {filename} with label {label}")
#                 sleep(5)
#
# if __name__ == "__main__":
#     send_videos("D:\\bigdata_dataset\\home\\anhkhoa\\spark_video_streaming\\data\\test")

NEW_ROOT = "D:\\bigdata_dataset\\home\\anhkhoa"

# Đọc file JSON
with open("data.json", "r") as f:
    data = json.load(f)

# Gửi từng video
for key, value in data.items():

    if value["flag"] != "test":
        continue  # bỏ qua nếu không phải mẫu test

    label = value["label"]
    original_path = value["url"]  # ví dụ: /home/anhkhoa/spark_video_streaming/data/train/superstitious/0102.mp4

    # Thay đường dẫn gốc
    relative_path = original_path.replace("/home/anhkhoa", "").replace("/", os.sep)
    full_path = os.path.join(NEW_ROOT, relative_path.strip(os.sep))

    # Video name = key + ".mp4"
    video_name = f"{key}.mp4"

    # Đọc video và encode
    try:
        with open(full_path, 'rb') as f:
            video_bytes = f.read()
            video_b64 = base64.b64encode(video_bytes).decode('utf-8')

        # Gửi message
        message = {
            "label": label,
            "video_name": video_name,
            "video_path": full_path,
            "video_data": video_b64,
            "text_embedding": value.get("text_embedding", [])
        }

        producer.send(topic_name, value=message)
        print(f"✅ Sent: {video_name} with label '{label}'")

    except FileNotFoundError:
        print(f"❌ File not found: {full_path}")

    sleep(5)
