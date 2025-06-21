import json
from kafka import KafkaProducer
import time
import logging
from utils import load_json

def get_data():
    json_data = load_json("/home/anhkhoa/spark_video_streaming/json/train.json")
    for key, value in json_data.items():
        yield {
            "idx": key,
            "url": value["url"],
            "label": value["label"]
        }

def stream_data():
    try:
        producer = KafkaProducer(bootstrap_servers=['localhost:9092'], max_block_ms=5000)

    except Exception as e:
        logging.error(f'Kafka connection failed: {e}')
        return

    curr_time = time.time()
    for res in get_data():
        if time.time() > curr_time + 60:
            break
        try:
            producer.send('users_created', json.dumps(res).encode('utf-8'))
            print(f"Sent: {res}")
            time.sleep(1)  # optional delay
        except Exception as e:
            logging.error(f'An error occurred while sending: {e}')
            continue

stream_data()
