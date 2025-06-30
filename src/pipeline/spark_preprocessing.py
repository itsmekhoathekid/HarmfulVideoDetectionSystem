import logging

from cassandra.cluster import Cluster
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType 

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType, StringType
import torch
from feature_extractor import VideoProcessor
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
        print(f"[Video processing] {count}/{len(video_strings)}")
    
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
        results.append(tensor_to_base64(tensor))
        count += 1
        print(f"[Audio processing] {count}/{len(video_strings)}")
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

def create_keyspace(session):
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS spark_streams
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'};
    """)

    print("Keyspace created successfully!")


def create_table(session):
    session.execute("DROP TABLE IF EXISTS spark_streams.created_users")
    session.execute("""
        CREATE TABLE IF NOT EXISTS spark_streams.created_users (
            id TEXT PRIMARY KEY,
            url TEXT,
            label TEXT,
            video_feat TEXT,
            audio_feat TEXT,
            text_embedding LIST<FLOAT>,
            split TEXT
        );
    """)
    print("Table created successfully!")
    


# def insert_data(session, **kwargs):
#     print("inserting data...")

#     idx = kwargs.get('idx')
#     url = kwargs.get('url')
#     label = kwargs.get('label')
#     video_encoded = kwargs.get('video_encoded')
#     text_embedding = kwargs.get('text_embedding')

#     try:
#         session.execute("""
#             INSERT INTO spark_streams.created_users(id, url, label, video_encoded, text_embedding)
#                 VALUES (%s, %s, %s, %s, %s)
#         """, (idx, url, label, video_encoded, text_embedding))
#         logging.info(f"Data inserted for id : {idx}")

#     except Exception as e:
#         logging.error(f'could not insert data due to {e}')

def create_spark_connection():
    s_conn = None
    try:
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
        s_conn.sparkContext.addPyFile("/home/anhkhoa/spark_video_streaming/src/pipeline/feature_extractor.py")
        logging.info("✅ Spark connection created successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to create Spark connection: {e}")
    
    return s_conn




def create_cassandra_connection():
    try:
        # connecting to the cassandra cluster
        cluster = Cluster(['localhost'], protocol_version=4, allow_beta_protocol_version=True)

        cas_session = cluster.connect()

        return cas_session
    except Exception as e:
        logging.error(f"Could not create cassandra connection due to {e}")
        return None


def connect_to_kafka(spark):
    return (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "users_created")
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 3) # ✅ chỉ xử lý 10 record/batch
        .option("failOnDataLoss", "false")  # ✅ Bỏ qua offset lỗi
        .load())

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

        # selection_df = selection_df \
        #     .withColumn("video_feat", extract_video_features(col("url"))) \
        #     .withColumn("audio_feat", extract_audio_features(col("url")))

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
        
        # batch_df = batch_df.drop("video_encoded")
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


