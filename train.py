import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from model import ModelWrapper
from losses import compute_alignment_loss
from util import video_to_frames
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

 # Create datasets
class VideoDataset(Dataset):
    def __init__(self, video_files):
        print(f"Initializing VideoDataset with {len(video_files)} videos...")
        self.video_files = video_files
        self.frames = []
        for i, video_file in enumerate(video_files):
            print(f"Processing video {i+1}/{len(video_files)}: {video_file.name}")
            self.frames.append(self.preprocess_video(video_file))
        # Find the shortest video length
        shortest_video_length = min(frame.shape[0] for frame in self.frames)

        # Trim all videos to the shortest length, discarding time from the front
        self.frames = [frame[-shortest_video_length:] for frame in self.frames]

        print("VideoDataset initialization complete!")

    def preprocess_video(self, video_path):
        print(f"\tLoading frames from {video_path.name}...")
        frames = video_to_frames(video_path)
        print(f"\tConverting frames to tensor... Shape: {frames.shape}")
        # Load frames from video
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to torch tensor and change dimensions to [T,C,H,W]
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(0, 3, 1, 2)
        
        return frames
        
    def __len__(self):
        return len(self.video_files)
        
    def __getitem__(self, idx):
        return self.frames[idx]

# def collate_fn(batch):
#     # Pad sequences to the same length
#     # Each item in batch is [T,C,H,W]
#     sequences = [item for item in batch]
    
#     # Get max sequence length in this batch
#     max_len = max([seq.shape[0] for seq in sequences])
#     C, H, W = sequences[0].shape[1:]
    
#     # Create padded batch
#     padded_batch = torch.zeros(len(sequences), max_len, C, H, W)
#     for i, seq in enumerate(sequences):
#         T = seq.shape[0]
#         padded_batch[i, :T] = seq

#     return padded_batch

class LitModel(pl.LightningModule):
    def __init__(self, model=None):
        super().__init__()
        print("Initializing LitModel...")
        self.model = model if model else ModelWrapper()
                
    def training_step(self, batch, batch_idx):
        x = batch
        y_hat, cnn_feats = self(x)
        loss = compute_alignment_loss(y_hat, cnn_feats, batch_size=x.shape[0], 
                                      loss_type='classification', 
                                      normalize_embeddings=False,
                                      similarity_type='cosine')
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x = batch
        with torch.no_grad():
            y_hat, cnn_feats = self(x)
        loss = compute_alignment_loss(y_hat, cnn_feats, batch_size=x.shape[0], 
                                      loss_type='classification', 
                                      normalize_embeddings=False,
                                      similarity_type='cosine')
        self.log('val_loss', loss)
        return loss
        
    def configure_optimizers(self):
        print("Configuring optimizer...")
        parameters = list(self.model.cnn.parameters()) + list(self.model.emb.parameters())
        # TODO: use config file to manage the configs
        optimizer = optim.Adam(parameters, lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.97)
        return [optimizer], [scheduler]


    def forward(self, frames):
        embs = self.model(frames)
        return embs
    
def video_duration(filename):
    video = cv2.VideoCapture(filename)
    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print(filename, 'is', frame_count, 'frames')
    return frame_count

def train():
    print("Starting training process...")
    print("Loading video files...")
    video_dir = Path('../videos_160')
    video_files = list(video_dir.glob('*.mp4'))
    video_files = [file for file in video_files if video_duration(file) > 100 and video_duration(file) < 200]
    video_files.sort(key=lambda x : int(str(x).split('/')[-1].split('.')[0]))
    print(f"Found {len(video_files)} video files")
    for file in video_files:
        print(file)
    # Split into train/val sets
    num_videos = len(video_files)
    train_size = int(0.9 * num_videos)
    
    train_files = video_files[:train_size]
    val_files = video_files[train_size:]
    print(f"Split dataset: {len(train_files)} training videos, {len(val_files)} validation videos")

    print("Creating training dataset...")
    train_dataset = VideoDataset(train_files)
    print("Creating validation dataset...")
    val_dataset = VideoDataset(val_files)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset.transform = preprocess
    # val_dataset.transform = preprocess
    
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8,
        drop_last=True
    )
    
    print("Initializing model...")
    model = LitModel()
    
    print("Setting up training callbacks and logger...")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    
    logger = TensorBoardLogger("lightning_logs", name="my_model")
    
    print("Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator='gpu'
    )
    
    print("Starting model training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training complete!")

if __name__ == "__main__":
    print("Script started...")
    train()