from pyspark.sql import SparkSession
from cassandra.cluster import Cluster
import logging

def create_keyspace(session, config):
    keyspace = config['cassandra']['keyspace']
    
    # Câu lệnh CQL để tạo Keyspace với replication
    create_keyspace_query = """
        CREATE KEYSPACE IF NOT EXISTS {};
    """.format(keyspace)

    try:
        session.execute(create_keyspace_query)
        print(f"Keyspace '{keyspace}' created successfully!")
    except Exception as e:
        print(f"Error creating keyspace: {e}")


def create_table(session, config):
    keyspace = config['cassandra']['keyspace']
    table = config['cassandra']['table']
    
    # CQL query to create table
    create_table_query = """
        CREATE TABLE IF NOT EXISTS {}.{} (
            id TEXT PRIMARY KEY,
            url TEXT,
            label TEXT,
            video_feat TEXT,
            audio_feat TEXT,
            text_embedding LIST<FLOAT>,
            split TEXT
        );
    """.format(keyspace, table)
    
    try:
        session.execute(create_table_query)
        print(f"Table '{table}' created successfully in keyspace '{keyspace}'!")
    except Exception as e:
        print(f"Error creating table: {e}")
    


def create_spark_connection(config):
    s_conn = None
    try:
        s_conn = SparkSession.builder \
            .appName(config['spark']['spark_name']) \
            .config("spark.jars.packages",
                    "com.datastax.spark:spark-cassandra-connector_2.12:3.4.1,"
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1") \
            .config("spark.cassandra.connection.host", config['spark']['connection_host']) \
            .config("spark.executor.memory", config['spark']['executor_memory']) \
            .config("spark.driver.memory", config['spark']['driver_memory']) \
            .config("spark.executor.instances", config['spark']['executer_instances']) \
            .config("spark.sql.shuffle.partitions", config['spark']['shuffle_partitions']) \
            .config("spark.default.parallelism", config['spark']['parallelism']) \
            .getOrCreate()

        s_conn.sparkContext.setLogLevel("ERROR")
        s_conn.sparkContext.addPyFile("/home/anhkhoa/spark_video_streaming/src/pipeline.zip")
        s_conn.sparkContext.addPyFile("/home/anhkhoa/spark_video_streaming/src/pipeline/feature_extractor.py")
        s_conn.sparkContext.addPyFile("/home/anhkhoa/spark_video_streaming/src/pipeline/udf.py")
        s_conn.sparkContext.addPyFile("/home/anhkhoa/spark_video_streaming/src/pipeline/utils.py")

        logging.info("✅ Spark connection created successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to create Spark connection: {e}")
    
    return s_conn




def create_cassandra_connection(config):
    try:
        # connecting to the cassandra cluster
        cluster = Cluster([config['cassandra']['contact_points']], 
                          protocol_version=config['cassandra']['protocal_version'], 
                          allow_beta_protocol_version=config['cassandra']['allow_beta_protocol_version'])

        cas_session = cluster.connect()

        return cas_session
    except Exception as e:
        logging.error(f"Could not create cassandra connection due to {e}")
        return None


def connect_to_kafka(spark, config):
    return (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", config['kafka']['bootstrap_servers'])
        .option("subscribe", config['kafka']['topic'])
        .option("startingOffsets", config['kafka']['starting_offsets'])
        .option("maxOffsetsPerTrigger", config['kafka']['max_offsets']) # ✅ chỉ xử lý 10 record/batch
        .option("failOnDataLoss", config['kafka']['fail_on_error'])  # ✅ Bỏ qua offset lỗi
        .load())

from pyspark.sql.types import ArrayType, FloatType, StringType
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType 

def create_selection_df_from_kafka(spark_df, config):
    if spark_df is not None:
        if config['kafka']['inference']:
            schema = StructType([
                StructField("idx", StringType()),
                StructField("url", StringType()),
                StructField("label", StringType()),
                StructField("video_encoded", StringType()),
                StructField("text_embedding", ArrayType(FloatType())),
                StructField("split", StringType())
            ])
        else:
            schema = StructType([
                StructField("idx", StringType()),
                StructField("url", StringType()),
                StructField("label", StringType()),
                StructField("text_embedding", ArrayType(FloatType())),
                StructField("split", StringType())
            ])

        selection_df = spark_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema).alias("data")) \
            .select("data.*")

        selection_df.printSchema()

        selection_df = selection_df.withColumn("id", col("idx"))  # tạo id

        logging.info("✅ Selection dataframe created from Kafka stream")
        return selection_df # ✅ chỉ giữ cần thiết
    else:
        logging.warning("❌ No valid Kafka dataframe available")
        return None
