import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession




import logging


from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType, StringType, IntegerType
import torch
from pipeline.feature_extractor import VideoProcessor
import pandas as pd
import uuid
import os
import base64

import logging

# Đọc dữ liệu từ Kafka
schema = StructType([
            StructField("label", StringType()),
            StructField("video_name", StringType()),
            StructField("video_path", StringType()),
            StructField("video_data", StringType()),
            StructField("text_embedding", ArrayType(FloatType()))
        ])
# Đọc từ Kafka
s_conn = SparkSession.builder \
    .appName("SparkVideoStreaming") \
    .config("spark.jars.packages",
            "com.datastax.spark:spark-cassandra-connector_2.12:3.4.1,"
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1") \
    .config("spark.cassandra.connection.host", "localhost") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.instances", "2") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.default.parallelism", "2") \
    .getOrCreate()

s_conn.sparkContext.setLogLevel("ERROR")

# Đọc dữ liệu từ Cassandra vào DataFrame
df = s_conn.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "data-test") \
    .option("startingOffsets", "latest") \
    .option("maxOffsetsPerTrigger", 1) \
    .option("failOnDataLoss", "false") \
    .load()



video_processor = VideoProcessor({
    "num_frames": 30,
    "resize": (224, 224),
    "n_mfcc": 40,
    "max_length": 40
})


@pandas_udf(StringType())
def extract_video_features(video_paths: pd.Series) -> pd.Series:
    results = []
    count = 0
    for path in video_paths:
        try:
            tensor = video_processor.process_video(path)
        except Exception as e:
            print(f"❌ Video error: {e}")
            tensor = torch.zeros((30, 3, 224, 224))
        count += 1
        print(f"[Video processing] {count}/{len(video_paths)} videos at path {path}")
        results.append(tensor)
    
    return pd.Series(results)

@pandas_udf(StringType())
def extract_audio_features(video_strings: pd.Series) -> pd.Series:
    results = []
    count = 0 
    for video_string in video_strings:
        try:
            tensor = video_processor.process_audio(video_string)
        except Exception as e:
            print(f"❌ Audio error: {e}")
            tensor = torch.zeros((40, 40))
        count += 1
        print(f"[Video audio] {count}/{len(video_strings)} audio ")
        results.append(tensor)
    return pd.Series(results)

import torch
import torchvision.transforms as transforms
from PIL import Image
import io



# Transformation: adjust according to your model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



import torch
import os
import numpy as np
from models import MultimodalModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
num_classes = 6  #

model = MultimodalModel(
    num_classes=num_classes,
    visual_hidden_size=256,
    audio_hidden_size=128,
    text_hidden_size=128,
    embed_dim=256
)
model.load_state_dict(torch.load(
    "/home/anhkhoa/spark_video_streaming/checkpoint/model_checkpoint.pth"
))
model.to(device)
model.eval()


def map_class_idx(pred_class: int):
    target_names = ["horrible", "normal", "offensive", "pornographic", "supertitious", "violent"]

    if 0 <= pred_class < len(target_names):
        return target_names[pred_class]

@pandas_udf("string")
def predict_label_udf(video_feats: pd.Series, audio_feats: pd.Series, text_embeds: pd.Series) -> pd.Series:
    preds = []
    model.eval()

    for v_feat, a_feat, t_embed in zip(video_feats, audio_feats, text_embeds):
        try:
            # Convert list -> tensor
            a = a_feat.unsqueeze(0).to(device)
            v = v_feat.unsqueeze(0).to(device)
            t = torch.tensor(t_embed, dtype=torch.float)
            t = t.unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(v, a, t)
                probs = torch.softmax(out, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                label = map_class_idx(pred_idx)

            preds.append(label)
        except Exception as e:
            preds.append("unknown")  # fallback

    return pd.Series(preds)


# Đọc dữ liệu từ Kafka
schema = StructType([
            StructField("label", StringType()),
            StructField("video_name", StringType()),
            StructField("video_path", StringType()),
            StructField("video_data", StringType()),
            StructField("text_embedding", ArrayType(FloatType()))
        ])
# Đọc từ Kafka


# Parse JSON
selection_df = df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")
# Parse JSON

print("Schema of selection_df:")
selection_df.printSchema()

final_df = selection_df \
            .withColumn("video_feat", extract_video_features(col("video_data"))) \
            .withColumn("audio_feat", extract_audio_features(col("video_data")))


with_prediction = final_df.withColumn(
    "predicted_label",
    predict_label_udf(col("video_feat"), col("audio_feat"), col("text_embedding"))
)

# Nếu bạn chỉ muốn in từng dòng kèm nhãn dự đoán
result_df = with_prediction.select("video_name", "label", "predicted_label")

query = result_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
