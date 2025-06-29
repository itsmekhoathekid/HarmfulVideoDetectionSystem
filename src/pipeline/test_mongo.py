import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession

from pyspark.sql.functions import from_json, col, udf, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType, StringType, IntegerType
import torch
from feature_extractor import VideoProcessor
import pandas as pd
import uuid
import os
import base64
from utils import tensor_to_base64, base64_to_tensor
import logging
import numpy as np
from model import MultimodalModel
import torchvision.transforms as transforms
from PIL import Image
import io

# scala_version = '2.12'  # your scala version
# spark_version = '3.5.5' # your spark version
# packages = [
#     f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
#     'org.apache.kafka:kafka-clients:3.5.0',
# ]
# spark = SparkSession.builder.master("local").appName("kafka-example").config("spark.jars.packages", ",".join(packages)).getOrCreate()
# spark

# Cấu hình SparkSession với Kafka + MongoDB connector
scala_version = '2.12'
spark_version = '3.5.5'
packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
    'org.apache.kafka:kafka-clients:3.5.0',
    'org.mongodb.spark:mongo-spark-connector_2.12:10.2.0'
]

spark = SparkSession.builder \
    .appName("KafkaToMongoDB") \
    .master("local[*]") \
    .config("spark.jars.packages", ",".join(packages)) \
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017/streaming_db.label_counts") \
    .getOrCreate()



# Đọc dữ liệu từ Kafka
schema = StructType([
            StructField("label", StringType()),
            StructField("video_name", StringType()),
            StructField("video_path", StringType()),
            StructField("video_data", StringType()),
            StructField("text_embedding", ArrayType(FloatType()))
        ])
# Đọc từ Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "data-test") \
    .option("startingOffsets", "latest") \
    .load()

video_processor = VideoProcessor({
    "num_frames": 30,
    "resize": (224, 224),
    "n_mfcc": 40,
    "max_length": 40
})


@pandas_udf(StringType())
def extract_video_features_base64(video_strings: pd.Series) -> pd.Series:
    results = []
    count = 0
    for video_string in video_strings:
        try:
            tensor = video_processor.process_video_base64(video_string)
        except Exception as e:
            print(f"❌ Video error: {e}")
            tensor = torch.zeros((30, 3, 224, 224))
        results.append(tensor_to_base64(tensor))
        count += 1
        logging.info(f"Processed {count}/{len(video_strings)} videos")

    return pd.Series(results)


@pandas_udf(StringType())
def extract_audio_features_base64(video_strings: pd.Series) -> pd.Series:
    results = []
    for video_string in video_strings:
        try:
            tensor = video_processor.process_audio_base64(video_string)
        except Exception as e:
            print(f"❌ Audio error: {e}")
            tensor = torch.zeros((40, 40))
        results.append(tensor_to_base64(tensor))
    return pd.Series(results)





model = torch.load("model.pth", map_location=torch.device("cpu"))
model.eval()

# Transformation: adjust according to your model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])





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
    "multimodal_model_less_complex_v3.pth"
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
            # a = a_feat.unsqueeze(0).to(device)
            # v = v_feat.unsqueeze(0).to(device)
            a = base64_to_tensor(a_feat).unsqueeze(0).to(device)
            v = base64_to_tensor(v_feat).unsqueeze(0).to(device)
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


# Đọc dữ liệu từ Kafka và ghi kết quả dự đoán
schema = StructType([
            StructField("label", StringType()),
            StructField("video_name", StringType()),
            StructField("video_path", StringType()),
            StructField("video_data", StringType()),
            StructField("text_embedding", ArrayType(FloatType()))
        ])

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "data-test") \
    .option("startingOffsets", "latest") \
    .load()


selection_df = df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")
final_df = selection_df \
            .withColumn("video_feat", extract_video_features_base64(col("video_data"))) \
            .withColumn("audio_feat", extract_audio_features_base64(col("video_data")))
inferred_df = final_df.withColumn(
    "predicted_label",
    predict_label_udf(
        col("video_feat"),
        col("audio_feat"),
        col("text_embedding")
    ).groupBy("predicted_label") \
    .count()
)

inferred_df.printSchema()


def write_to_mongo(batch_df, epoch_id):
    batch_df.write \
        .format("mongodb") \
        .mode("append") \
        .option("spark.mongodb.write.connection.uri", "mongodb://localhost:27017/streaming_db.label_counts") \
        .save()



# Ghi kết quả stream vào MongoDB
query = inferred_df.writeStream \
    .outputMode("update") \
    .foreachBatch(write_to_mongo) \
    .trigger(processingTime="5 seconds") \
    .start()

query.awaitTermination()
