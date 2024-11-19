import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
# from torchvision.datasets import ImageFolder
# from torchvision.transforms import ToTensor
from model import ConvEmbedder
from losses import compute_alignment_loss
from util import video_to_frames
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

 # Create datasets
class VideoDataset(Dataset):
    def __init__(self, video_files):
        print(f"Initializing VideoDataset with {len(video_files)} videos...")
        self.video_files = video_files
        self.frames = []
        for i, video_file in enumerate(video_files):
            print(f"Processing video {i+1}/{len(video_files)}: {video_file.name}")
            self.frames.append(self.preprocess_video(video_file))
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

def collate_fn(batch):
    # Pad sequences to the same length
    # Each item in batch is [T,C,H,W]
    sequences = [item for item in batch]
    
    # Get max sequence length in this batch
    max_len = max([seq.shape[0] for seq in sequences])
    C, H, W = sequences[0].shape[1:]
    
    # Create padded batch
    padded_batch = torch.zeros(len(sequences), max_len, C, H, W)
    for i, seq in enumerate(sequences):
        T = seq.shape[0]
        padded_batch[i, :T] = seq
        
    return padded_batch

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        print("Initializing LitModel...")
        self.model = ConvEmbedder()
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        print(f"\nTraining step {batch_idx}")
        print(f"Input batch shape: {batch.shape}")
        x = batch
        y_hat = self.model(x)
        print(f"Output shape: {y_hat.shape}")
        loss = compute_alignment_loss(y_hat, batch_size=x.shape[0])
        print(f"Training loss: {loss.item():.4f}")
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        print(f"\nValidation step {batch_idx}")
        print(f"Input batch shape: {batch.shape}")
        x = batch
        with torch.no_grad():
            y_hat = self.model(x)
        print(f"Output shape: {y_hat.shape}")
        loss = compute_alignment_loss(y_hat, batch_size=x.shape[0])
        print(f"Validation loss: {loss.item():.4f}")
        self.log('val_loss', loss)
        
    def configure_optimizers(self):
        print("Configuring optimizer...")
        optimizer = optim.Adam(self.parameters())
        return optimizer

def train():
    print("Starting training process...")
    print("Loading video files...")
    video_dir = Path('data/videos')
    video_files = list(video_dir.glob('*.mp4'))[:10]
    print(f"Found {len(video_files)} video files")

    # Split into train/val sets
    num_videos = len(video_files)
    train_size = int(0.8 * num_videos)
    
    train_files = video_files[:train_size]
    val_files = video_files[train_size:]
    print(f"Split dataset: {len(train_files)} training videos, {len(val_files)} validation videos")

    print("Creating training dataset...")
    train_dataset = VideoDataset(train_files)
    print("Creating validation dataset...")
    val_dataset = VideoDataset(val_files)
    
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2, 
        collate_fn=collate_fn
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
        max_epochs=10,
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