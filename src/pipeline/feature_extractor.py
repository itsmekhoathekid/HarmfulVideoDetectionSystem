import uuid
import cv2
import librosa
import numpy as np
import os
from moviepy.editor import VideoFileClip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import whisper
from vncorenlp import VnCoreNLP
from transformers import pipeline
import base64
import tempfile
import imageio
from io import BytesIO
import ffmpeg
import soundfile as sf

class PretrainedModelLoader:
    def __init__(self, model_name="vinai/phobert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)
        self.text_model.eval()
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model="vinai/PhoWhisper-small",
            device='cuda'
        )

        self.rdgsegmenter = VnCoreNLP(
            "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar",
            annotators="wseg,pos,ner,parse",
            max_heap_size='-Xmx2g'
        )
    
    def load_model(self):
        """
        Load the pretrained model and tokenizer.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model.to(device)
        
        return self.tokenizer, self.text_model, self.transcriber, self.rdgsegmenter


class VideoProcessor:
    def __init__(self, data_configs):
        # self.tokenizer, self.text_model, self.transcriber, self.rdgsegmenter = pretrained_loader.load_model()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_configs = data_configs

    def process_video_base64(self, base64_string):
        num_frames = self.data_configs.get("num_frames", 30)
        resize = self.data_configs.get("resize", (224, 224))

        try:
            # Giải mã base64
            video_bytes = base64.b64decode(base64_string)

            # Ghi vào file tạm
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(video_bytes)
                temp_path = temp_file.name

            # Dùng OpenCV để đọc video từ file tạm
            cap = cv2.VideoCapture(temp_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(total_frames // num_frames, 1)

            for i in range(0, total_frames, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, resize)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                if len(frames) == num_frames:
                    break
            cap.release()

            os.remove(temp_path)  # Xóa file tạm

            # Bù khung hình nếu thiếu
            while len(frames) < num_frames:
                frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.uint8))

            frames = np.array(frames).transpose(0, 3, 1, 2) / 255.0
            frames_tensor = torch.tensor(frames, dtype=torch.float)

        except Exception as e:
            print(f"Error processing video: {e}")
            frames_tensor = torch.zeros(num_frames, 3, resize[0], resize[1], dtype=torch.float)

        return frames_tensor

    def process_audio_base64(self, base64_string):
        n_mfcc = self.data_configs.get("n_mfcc", 40)
        max_length = self.data_configs.get("max_length", 40)
    
        video_path = f"/tmp/temp_video_{uuid.uuid4().hex}.mp4"
        audio_path = f"/tmp/temp_audio_{uuid.uuid4().hex}.wav"

        try:
            # Lưu chuỗi base64 thành video file tạm
            with open(video_path, "wb") as f:
                f.write(base64.b64decode(base64_string))

            # Dùng moviepy trích xuất audio
            video = VideoFileClip(video_path)
            audio = video.audio
            if audio is not None:
                audio.write_audiofile(audio_path, logger=None)
                y, sr = librosa.load(audio_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

                # Cắt hoặc padding để chuẩn hóa kích thước
                if mfcc.shape[1] < max_length:
                    pad_width = max_length - mfcc.shape[1]
                    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
                else:
                    mfcc = mfcc[:, :max_length]

                mfcc_tensor = torch.tensor(mfcc, dtype=torch.float)
            else:
                raise ValueError("No audio track found in video")

        except Exception as e:
            print(f"Error processing audio from base64: {e}")
            mfcc_tensor = torch.zeros(n_mfcc, max_length)

        finally:
            # Xóa file tạm
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)

        return mfcc_tensor
    
    def process_video(self, video_path):
        num_frames = self.data_configs.get("num_frames", 30)
        resize = self.data_configs.get("resize", (224, 224))

        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(total_frames // num_frames, 1)

            for i in range(0, total_frames, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, resize)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                if len(frames) == num_frames:
                    break
            cap.release()

            # Bù khung hình nếu thiếu
            while len(frames) < num_frames:
                frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.uint8))

            frames = np.array(frames).transpose(0, 3, 1, 2) / 255.0
            frames_tensor = torch.tensor(frames, dtype=torch.float)

        except Exception as e:
            print(f"Error processing video: {e}")
            frames_tensor = torch.zeros(num_frames, 3, resize[0], resize[1], dtype=torch.float)

        return frames_tensor

    def process_audio(self, video_path):
        n_mfcc = self.data_configs.get("n_mfcc", 40)
        max_length = self.data_configs.get("max_length", 40)

        audio_path = f"/tmp/temp_audio_{uuid.uuid4().hex}.wav"
        try:
            video = VideoFileClip(video_path)
            audio = video.audio
            if audio is not None:
                audio.write_audiofile(audio_path, logger=None)
                y, sr = librosa.load(audio_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            
                if mfcc.shape[1] < max_length:
                    pad_width = max_length - mfcc.shape[1]
                    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
                else:
                    mfcc = mfcc[:, :max_length]
                mfcc_tensor = torch.tensor(mfcc, dtype=torch.float)

        except Exception as e:
            print(f"Error processing audio: {e}")
            mfcc_tensor = torch.zeros(n_mfcc, max_length)

        finally:
            # Xóa file audio tạm
            if os.path.exists(audio_path):
                os.remove(audio_path)
        return mfcc_tensor

