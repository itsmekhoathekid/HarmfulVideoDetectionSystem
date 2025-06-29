from moviepy.editor import VideoFileClip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import whisper
import tempfile
import os 
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import base64
import gzip
import io
import torch


class PreExtractedFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_paths = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(feature_dir)))}
        
        for cls in sorted(os.listdir(feature_dir)):
            cls_path = os.path.join(feature_dir, cls)
            for file in os.listdir(cls_path):
                if file.endswith('.pt'):
                    self.feature_paths.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        feature = torch.load(self.feature_paths[idx])
        frames = feature['frames']
        mfcc = feature['mfcc']
        text_embedding = feature['text_embedding']
        label = self.labels[idx]
        return frames, mfcc, text_embedding, label

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from torch.utils.data import Dataset
import torch
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType

def base64_to_tensor(b64_str) -> torch.Tensor:
    compressed = base64.b64decode(b64_str)
    decompressed = gzip.decompress(compressed)
    buffer = io.BytesIO(decompressed)
    return torch.load(buffer)

# Định nghĩa UDF trong PySpark, trả về ArrayType(FloatType) (mảng các float)
@pandas_udf(ArrayType(FloatType()))  # Chỉ định kiểu trả về là mảng các float
def base64_to_tensor_udf(b64_str):
    print(b64_str.dtype)  
    
    tensor = base64_to_tensor(b64_str)
    # Chuyển tensor thành list hoặc numpy array
    return tensor # Trả về dưới dạng list (mảng số thực)

from tqdm import tqdm

def get_data_dic():
    # Cấu hình SparkSession
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
    df = s_conn.read \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="created_users", keyspace="spark_streams") \
        .load()

    df_rows = df.collect()
    data_dict = [row.asDict() for row in df_rows]

    return data_dict


class cassandraExtractedFeatures(Dataset):
    def __init__(self, data_dic, split):
        self.data_dict = data_dic
        self.split = split
        self.data = self.load_and_process()
        

    def map_class_to_idx(self, class_name):
        class_to_idx = {
            "horrible": 0,
            "normal": 1,
            "offensive": 2,
            "pornographic": 3,
            "violent": 4,
            "superstitious": 5,
        }
        return class_to_idx.get(class_name, -1)

    def load_and_process(self):
        decoded_data = []
        for row_dict in tqdm(self.data_dict, desc = "Extracting features from Cassandra"):
            if row_dict["split"] == self.split:

                video_feat_tensor = base64_to_tensor(row_dict['video_feat'])
                audio_feat_tensor = base64_to_tensor(row_dict['audio_feat'])
                text_embedding = row_dict['text_embedding'] if len(row_dict['text_embedding']) > 2 else torch.zeros(768)
                label = self.map_class_to_idx(row_dict['label'])  # Nhận nhãn (label)

                decoded_data.append({
                    'video_feat_tensor': video_feat_tensor,
                    'audio_feat_tensor': audio_feat_tensor,
                    'label': label,
                    "text_embedding": text_embedding
                })
        
        return decoded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_feat = self.data[idx]['video_feat_tensor']
        audio_feat = self.data[idx]['audio_feat_tensor']
        text_embed = self.data[idx]['text_embedding']
        label = self.data[idx]['label']
        return torch.tensor(video_feat), torch.tensor(audio_feat), torch.tensor(text_embed), label





from tqdm import tqdm
import torch
device='cuda'

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=3, checkpoint_path=None):
    best_loss = float('inf')
    best_model_state = None
    counter = 0
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for visual, audio, text, labels in loop:
            visual = visual.to(device)
            audio = audio.to(device)
            text = text.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(visual, audio, text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * visual.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item(), accuracy=100. * correct / total)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")

        val_loss, val_acc = evaluate_loss(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            counter = 0
            if checkpoint_path:
                torch.save(best_model_state, checkpoint_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for visual, audio, text, labels in dataloader:
            visual = visual.to(device)
            audio = audio.to(device)
            text = text.to(device)
            labels = labels.to(device)

            outputs = model(visual, audio, text)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * visual.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for visual, audio, text, labels in dataloader:
            visual = visual.to(device)
            audio = audio.to(device)
            text = text.to(device)
            labels = labels.to(device)

            outputs = model(visual, audio, text)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * visual.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for visual, audio, text, labels in tqdm(dataloader, desc="Evaluating"):
            visual = visual.to(device)
            audio = audio.to(device)
            text = text.to(device)
            labels = labels.to(device)

            outputs = model(visual, audio, text)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = 100. * correct / total
    print(f"✅ Final Accuracy on validation set: {accuracy:.2f}%")
    return accuracy, all_predicted, all_labels