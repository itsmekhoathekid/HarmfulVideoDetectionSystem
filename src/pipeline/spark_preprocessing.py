import logging

from cassandra.cluster import Cluster
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType 

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType, StringType
import torch
from .feature_extractor import VideoProcessor
import pandas as pd
import uuid
import os
import base64
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import gzip
from .udf import extract_video_features_base64, extract_audio_features_base64, extract_video_features, extract_audio_features
from .connection import create_spark_connection, create_cassandra_connection, create_keyspace, create_table, connect_to_kafka

# Ghi log vào file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/anhkhoa/spark_video_streaming/logs/spark.log"),
        logging.StreamHandler()  # vẫn hiển thị ra console
    ]
)

video_processor = VideoProcessor({
    "num_frames": 30,
    "resize": (224, 224),
    "n_mfcc": 40,
    "max_length": 40
})



    



from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, FloatType



            
def create_selection_df_from_kafka(spark_df):
    if spark_df is not None:
        schema = StructType([
            StructField("idx", StringType()),
            StructField("url", StringType()),
            StructField("label", StringType()),
            # StructField("video_encoded", StringType()),
            StructField("text_embedding", ArrayType(FloatType())),
            StructField("split", StringType())
        ])

        selection_df = spark_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema).alias("data")) \
            .select("data.*")

        selection_df.printSchema()

        selection_df = selection_df.withColumn("id", col("idx"))  # tạo id

        logging.info("✅ Selection dataframe created from Kafka stream")
        return selection_df.select("id", "url", "label", "split", "text_embedding")  # ✅ chỉ giữ cần thiết
    else:
        logging.warning("❌ No valid Kafka dataframe available")
        return None




def process_batch(batch_df, batch_id):
    print(f"[Batch ID: {batch_id}] Received batch with {batch_df.count()} rows")
    from pyspark.sql.functions import col

    if not batch_df.rdd.isEmpty():
        print(f"[Batch ID: {batch_id}] Row count: {batch_df.count()}")
        batch_df = batch_df \
            .withColumn("video_feat", extract_video_features(col("url"))) \
            .withColumn("audio_feat", extract_audio_features(col("url")))
        
        try:
            batch_df.write \
                .format("org.apache.spark.sql.cassandra") \
                .option("keyspace", "spark_streams") \
                .option("table", "created_users") \
                .mode("append") \
                .save()
            print(f"[Batch ID: {batch_id}] ✅ Written to Cassandra.\n")
        except Exception as e:
            print(f"[Batch ID: {batch_id}] ❌ Failed to write to Cassandra: {e}")
    else:
        print(f"[Batch ID: {batch_id}] ⚠️ No data to write to Cassandra.\n")




if __name__ == "__main__":
    spark_conn = create_spark_connection()
    if spark_conn is not None:
        spark_df = connect_to_kafka(spark_conn)
        selection_df = create_selection_df_from_kafka(spark_df)

        if selection_df is not None:
            session = create_cassandra_connection()
            if session is not None:
                create_keyspace(session)
                create_table(session)

            logging.info("⚡ Streaming started with feature extraction...")

            streaming_query = (selection_df.writeStream
                   .foreachBatch(process_batch)
                   .option("checkpointLocation", "/tmp/checkpoint-v2")
                #    .trigger(processingTime="5 seconds")
                   .start())


            streaming_query.awaitTermination()
        else:
            print("⚠️ Kafka selection_df is None. Check Kafka connection or schema.")


