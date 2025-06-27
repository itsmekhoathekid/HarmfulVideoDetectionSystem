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

import logging

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
    "num_frames": 10,
    "resize": (112, 112),
    "n_mfcc": 20,
    "max_length": 20
})

@pandas_udf(ArrayType(FloatType()))
def extract_video_features(video_strings: pd.Series) -> pd.Series:
    
    results = []
    for video_string in video_strings:
        try:
            tensor = video_processor.process_video_base64(video_string)
            flat = tensor.flatten().tolist()
        except Exception as e:
            print(e)
            flat = [0.0] * (3 * 30 * 224 * 224)
        results.append(flat)
    print(f"✅ Processing video: {video_strings}")
    return pd.Series(results)

@pandas_udf(ArrayType(FloatType()))
def extract_audio_features(video_strings: pd.Series) -> pd.Series:

    results = []
    for video_string in video_strings:
        try:
            tensor = video_processor.process_audio_base64(video_string)
            flat = tensor.flatten().tolist()
        except Exception as e:
            print(e)
            flat = [0.0] * (40 * 40)
        results.append(flat)
    print(f"✅ Processing video: {video_strings}")
    return pd.Series(results)




# def create_keyspace(session):
#     session.execute("""
#         CREATE KEYSPACE IF NOT EXISTS spark_streams
#         WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'};
#     """)

#     print("Keyspace created successfully!")


# def create_table(session):
#     session.execute("DROP TABLE IF EXISTS spark_streams.created_users")
#     session.execute("""
#         CREATE TABLE IF NOT EXISTS spark_streams.created_users (
#             id UUID PRIMARY KEY,
#             url TEXT,
#             label TEXT,
#             video_feat LIST<FLOAT>,
#             audio_feat LIST<FLOAT>,
#             text_embedding LIST<FLOAT>,
#             split TEXT
#         );
#     """)
#     print("Table created successfully!")



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
        scala_version = '2.12'
        spark_version = '3.5.5'

        packages = [
            f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
            'org.apache.kafka:kafka-clients:3.5.0',
            'org.mongodb.spark:mongo-spark-connector_2.12:10.2.0'
        ]
        # s_conn = SparkSession.builder \
        #     .appName("SparkVideoStreaming") \
        #     .master("local[*]") \
        #     .config("spark.jars.packages", ",".join(packages)) \
        #     .config("spark.cassandra.connection.host", "localhost") \
        #     .getOrCreate()
        
        s_conn = SparkSession.builder \
            .appName("SparkVideoStreaming") \
            .master("local[*]") \
            .config("spark.jars.packages", ",".join(packages)) \
            .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017") \
            .config("spark.mongodb.write.database", "spark_streams") \
            .config("spark.mongodb.write.collection", "created_users") \
            .getOrCreate()

        s_conn.sparkContext.setLogLevel("ERROR")
        s_conn.sparkContext.addPyFile("/home/anhkhoa/spark_video_streaming/pipeline/feature_extractor.py")
        logging.info("✅ Spark connection (Mongo) created successfully.")
    except Exception as e:
        logging.error(f"Failed to create Spark connection: {e}")
    
    return s_conn


# def create_cassandra_connection():
#     try:
#         # connecting to the cassandra cluster
#         cluster = Cluster(['localhost'], protocol_version=5)

#         cas_session = cluster.connect()

#         return cas_session
#     except Exception as e:
#         logging.error(f"Could not create cassandra connection due to {e}")
#         return None


def connect_to_kafka(spark_conn):
    spark_df = None
    try:
        spark_df = spark_conn.readStream \
            .format('kafka') \
            .option('kafka.bootstrap.servers', 'localhost:9092') \
            .option('subscribe', 'users_created') \
            .option('startingOffsets', 'earliest') \
            .option("failOnDataLoss", "false") \
            .load()
        logging.info("kafka dataframe created successfully")
    except Exception as e:
        logging.warning(f"kafka dataframe could not be created because: {e}")

    return spark_df

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
        
        selection_df = selection_df \
             .withColumn("video_feat", extract_video_features(col("video_encoded"))) \
             .withColumn("audio_feat", extract_audio_features(col("video_encoded")))
        # return selection_df
        return selection_df.select("id", "url", "label", "video_feat", "audio_feat")  # ✅ chỉ giữ cần thiết
    else:
        logging.warning("❌ No valid Kafka dataframe available")
        return None




def process_batch(batch_df, batch_id):
    if not batch_df.rdd.isEmpty():
        batch_df.write \
            .format("mongodb") \
            .mode("append") \
            .save()
        print(f"[Batch ID: {batch_id}] Written to MongoDB.\n")


if __name__ == "__main__":
    spark_conn = create_spark_connection()
    if spark_conn is not None:
        spark_df = connect_to_kafka(spark_conn)
        selection_df = create_selection_df_from_kafka(spark_df)

        if selection_df is not None:
            logging.info("Streaming started with MongoDB...")

            streaming_query = (selection_df.writeStream
                               .foreachBatch(process_batch)
                               .option("checkpointLocation", "/tmp/checkpoint-v2")
                               .start())

            streaming_query.awaitTermination()
        else:
            print("Kafka selection_df is None. Check Kafka connection or schema.")


