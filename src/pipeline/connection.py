from pyspark.sql import SparkSession
from cassandra.cluster import Cluster
import logging

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
        s_conn.sparkContext.addPyFile("/home/anhkhoa/spark_video_streaming/src/pipeline/udf.py")
        s_conn.sparkContext.addPyFile("/home/anhkhoa/spark_video_streaming/src/pipeline/utils.py")

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
