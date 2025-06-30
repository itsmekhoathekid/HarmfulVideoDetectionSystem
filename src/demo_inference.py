import streamlit as st
import os
import torch
import time
import glob
import pandas as pd
from models import MultimodalModel
import io
import base64
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

# === Khởi tạo model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultimodalModel(num_classes=6, visual_hidden_size=256,
                        audio_hidden_size=128, text_hidden_size=128, embed_dim=256)
model.load_state_dict(torch.load("/home/anhkhoa/spark_video_streaming/checkpoint/model_checkpoint.pth"))
model.to(device)
model.eval()


def map_class_idx(pred_class: int):
    target_names = ["horrible", "normal", "offensive", "pornographic", "superstitious", "violent"]
    if 0 <= pred_class < len(target_names):
        return target_names[pred_class]
    
sensitive_labels = {"horrible", "offensive", "pornographic", "superstitious", "violent"}

@torch.no_grad()
def infer_model(row):
    # Data preprocess
    video_feat = base64_to_tensor(row["video_feat"]).unsqueeze(0).to(device)
    audio_feat = base64_to_tensor(row["audio_feat"]).unsqueeze(0).to(device)
    text_emb = torch.tensor(row["text_embedding"], dtype=torch.float)
    text_emb = text_emb.unsqueeze(0).to(device)
    # Label prediction
    output = model(video_feat, audio_feat, text_emb)
    probs = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    print(f"Predicted class: {pred_class} for video: {row['video_name']}")
    return map_class_idx(pred_class)

# === UI Streamlit ===
st.set_page_config(page_title="Real-time Detection", layout="wide")
st.title("📺 Real-time Multimodal Prediction Dashboard")
placeholder = st.empty()

if "seen_batches" not in st.session_state:
    st.session_state.seen_batches = set()

data_dir = "/tmp/inference_output"

while True:
    new_rows = []
    batch_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith("batch_")])

    for batch_dir in batch_dirs:
        batch_id = batch_dir.replace("batch_", "")
        if batch_id in st.session_state.seen_batches:
            continue

        parquet_files = glob.glob(os.path.join(data_dir, batch_dir, "*.parquet"))
        if not parquet_files:
            continue

        df = pd.read_parquet(parquet_files[0])
        df["predicted_label"] = df.apply(infer_model, axis=1)
        df["batch_id"] = batch_id
        new_rows.append(df)

        st.session_state.seen_batches.add(batch_id)

    if new_rows:
        final_df = pd.concat(new_rows, ignore_index=True)
        display_df = final_df[["batch_id", "video_name", "predicted_label"]]

        # === Highlight nhãn nhạy cảm ===
        def highlight_label(val):
            if val in sensitive_labels:
                return "background-color: #ffcccc; color: red; font-weight: bold"
            else:
                return ""

        styled_df = display_df.style.applymap(highlight_label, subset=["predicted_label"])
        placeholder.dataframe(styled_df, use_container_width=True)

    time.sleep(5)
