import findspark
findspark.init()
import pyspark
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
import numpy as np
import sys
from models import MultimodalModel
import torchvision.transforms as transforms
from PIL import Image
import io
import gzip
from concurrent.futures import ThreadPoolExecutor


def tensor_to_base64(tensor: torch.Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    compressed = gzip.compress(buffer.getvalue())  # 👈 nén trước khi base64
    return base64.b64encode(compressed).decode("utf-8")

def base64_to_tensor(b64_str: str) -> torch.Tensor:
    compressed = base64.b64decode(b64_str)
    decompressed = gzip.decompress(compressed)
    buffer = io.BytesIO(decompressed)
    return torch.load(buffer)

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
        print(f"Processed {count}/{len(video_strings)} videos")

    return pd.Series(results)


@pandas_udf(StringType())
def extract_audio_features_base64(video_strings: pd.Series) -> pd.Series:
    results = []
    count = 0
    for video_string in video_strings:
        try:
            tensor = video_processor.process_audio_base64(video_string)
        except Exception as e:
            print(f"❌ Audio error: {e}")
            tensor = torch.zeros((40, 40))
        count += 1
        results.append(tensor_to_base64(tensor))
        print(f"Processed {count}/{len(video_strings)} audio")
    return pd.Series(results)


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
        results.append(tensor_to_base64(tensor))
        count += 1
        print(f"[Video processing] {count}/{len(video_paths)} videos at path {path}")

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
        results.append(tensor_to_base64(tensor))
        count += 1
        print(f"[Video audio] {count}/{len(video_strings)} audio ")
    return pd.Series(results)


@pandas_udf(StringType())
def extract_audio_features_base64_parallel(video_strings: pd.Series) -> pd.Series:
    def process_one(video_string):
        try:
            tensor = video_processor.process_audio_base64(video_string)
        except Exception as e:
            print(f"❌ Audio error: {e}")
            tensor = torch.zeros((40, 40))
        return tensor_to_base64(tensor)

    with ThreadPoolExecutor(max_workers=4) as executor:  # bạn có thể tăng lên tùy CPU
        results = list(executor.map(process_one, video_strings.tolist()))

    return pd.Series(results)


@pandas_udf(StringType())
def extract_video_features_base64_parallel(video_strings: pd.Series) -> pd.Series:
    def process_one(video_string):
        try:
            tensor = video_processor.process_video_base64(video_string)
        except Exception as e:
            print(f"❌ Audio error: {e}")
            tensor = torch.zeros((40, 40))
        return tensor_to_base64(tensor)

    with ThreadPoolExecutor(max_workers=4) as executor:  # bạn có thể tăng lên tùy CPU
        results = list(executor.map(process_one, video_strings.tolist()))

    return pd.Series(results)


def create_spark_connection():
    s_conn = None
    try:
        scala_version = '2.12'  # your scala version
        spark_version = '3.5.5'  # your spark version
        packages = [
            f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
            'org.apache.kafka:kafka-clients:3.5.0',
        ]
        s_conn = SparkSession.builder \
            .appName("SparkVideoStreaming") \
            .config("spark.jars.packages", ",".join(packages)) \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.instances", "2") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.default.parallelism", "2") \
            .getOrCreate()

        s_conn.sparkContext.setLogLevel("ERROR")
        s_conn.sparkContext.addPyFile("/home/anhkhoa/spark_video_streaming/src/pipeline.zip")
        logging.info("✅ Spark connection created successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to create Spark connection: {e}")

    return s_conn

def connect_to_kafka(spark):
    return (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "data-test")
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 3) # ✅ chỉ xử lý 10 record/batch
        .option("failOnDataLoss", "false")  # ✅ Bỏ qua offset lỗi
        .load())

def create_selection_df_from_kafka(spark_df):
    if spark_df is not None:
        schema = StructType([
            StructField("label", StringType()),
            StructField("video_name", StringType()),
            StructField("video_path", StringType()),
            StructField("video_data", StringType()),
            StructField("text_embedding", ArrayType(FloatType()))
        ])

        selection_df = spark_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema).alias("data")) \
            .select("data.*")


        

        selection_df.printSchema()
        logging.info("Selection dataframe created from Kafka stream")
        return selection_df.select("label", "video_name","video_data", "text_embedding")
    else:
        logging.warning("No valid Kafka dataframe available")
        return None



# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Device:", device)

# num_classes = 6  #
# model = MultimodalModel(
#     num_classes=num_classes,
#     visual_hidden_size=256,
#     audio_hidden_size=128,
#     text_hidden_size=128,
#     embed_dim=256
# )
# model.load_state_dict(torch.load(
#     "multimodal_model_less_complex_v3.pth", map_location=torch.device('cpu')
# ))
# model.to(device)
# model.eval()
#
#
# def map_class_idx(pred_class: int):
#     target_names = ["horrible", "normal", "offensive", "pornographic", "supertitious", "violent"]
#
#     if 0 <= pred_class < len(target_names):
#         return target_names[pred_class]
#
#
# @pandas_udf("string")
# def predict_label_udf(video_feats: pd.Series, audio_feats: pd.Series, text_embeds: pd.Series) -> pd.Series:
#     preds = []
#
#     for v_feat, a_feat, t_embed in zip(video_feats, audio_feats, text_embeds):
#         try:
#
#             a = a_feat.unsqueeze(0).to(device)
#             v = v_feat.unsqueeze(0).to(device)
#             # Convert list -> tensor
#             t = torch.tensor(t_embed, dtype=torch.float)
#             t = t.unsqueeze(0).to(device)
#
#             with torch.no_grad():
#                 out = model(v, a, t)
#                 probs = torch.softmax(out, dim=1)
#                 pred_idx = torch.argmax(probs, dim=1).item()
#                 label = map_class_idx(pred_idx)
#
#             preds.append(label)
#         except Exception as e:
#             preds.append("unknown")  # fallback
#
#     return pd.Series(preds)
#
# def create_predicted_df(selection_df):
#
#     return final_df

def process_batch(batch_df, batch_id):

    print(f"\n==== 🟢 BATCH ID {batch_id} | ROWS: {batch_df.count()} ====\n")

    batch_df = batch_df \
        .withColumn("video_feat", extract_video_features_base64(col("video_data"))) \
        .withColumn("audio_feat", extract_audio_features_base64(col("video_data")))
    output_dir = f"/tmp/inference_output/batch_{batch_id}"
    batch_df.write.mode("overwrite").parquet(output_dir)
    print(f"✅ Saved batch {batch_id} to {output_dir}")


if __name__ == "__main__":
    spark_conn = create_spark_connection()

    if spark_conn is not None:
        spark_df = connect_to_kafka(spark_conn)
        selection_df = create_selection_df_from_kafka(spark_df)

        if selection_df is not None:
            query = (selection_df.writeStream
                     .foreachBatch(process_batch)
                     .outputMode("append")
                     .trigger(processingTime="5 seconds")
                     .start())

            logging.info("✅ Streaming started with foreachBatch.")
            query.awaitTermination()