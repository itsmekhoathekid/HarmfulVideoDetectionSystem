from models import cassandraExtractedFeatures, train_model, evaluate_loss, device, MultimodalModel, get_data_dic

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

num_classes = 6
num_epochs = 1 
checkpoint_path = "/home/anhkhoa/spark_video_streaming/checkpoint/model_checkpoint.pth"

model = MultimodalModel(
    num_classes=num_classes, 
    visual_hidden_size=256, 
    audio_hidden_size=128, 
    text_hidden_size=128, 
    embed_dim=256
).to(device)

# load data_dict

data_dict = get_data_dic()

train_dataset = cassandraExtractedFeatures(data_dic=data_dict, split="train")
train_loader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)

val_loader = DataLoader(train_dataset, batch_size=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, patience=4, checkpoint_path=checkpoint_path)