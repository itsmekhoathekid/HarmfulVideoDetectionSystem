import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession

from pyspark.sql.functions import from_json, col, udf, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType, StringType, IntegerType
import torch

from pipeline import (
    VideoProcessor, 
    tensor_to_base64, 
    extract_audio_features_base64,
    extract_video_features_base64,
    extract_video_features,
    extract_audio_features
)

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


video_processor = VideoProcessor({
    "num_frames": 30,
    "resize": (224, 224),
    "n_mfcc": 40,
    "max_length": 40
})




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
        .option("subscribe", "data-test") # depends on kafka topic
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