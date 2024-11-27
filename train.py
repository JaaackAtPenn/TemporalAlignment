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
import argparse
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

 # Create datasets
class VideoDataset(Dataset):        # all the videos are trimmed to the shortest video length
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
    def __init__(self, model=None, loss_type='regression_mse_var', similarity_type='l2', temperature=0.1, variance_lambda=0.001, use_random_window=False, use_align_alpha=False, align_alpha_strength=0.1, do_not_reduce_frame_rate=False):
        super().__init__()
        print("Initializing LitModel...")
        self.model = model if model else ModelWrapper(do_not_reduce_frame_rate=do_not_reduce_frame_rate)
        self.loss_type = loss_type
        self.similarity_type = similarity_type
        self.temperature = temperature
        self.variance_lambda = variance_lambda
        self.use_random_window = use_random_window
        self.use_align_alpha = use_align_alpha
        self.align_alpha_strength = align_alpha_strength
                
    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        loss = compute_alignment_loss(y_hat, batch_size=x.shape[0], loss_type=self.loss_type, similarity_type=self.similarity_type, temperature=self.temperature, variance_lambda=self.variance_lambda, use_random_window=self.use_random_window, use_align_alpha=self.use_align_alpha, align_alpha_strength=self.align_alpha_strength)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x = batch
        with torch.no_grad():
            y_hat = self(x)
        loss = compute_alignment_loss(y_hat, batch_size=x.shape[0], loss_type=self.loss_type, similarity_type=self.similarity_type, temperature=self.temperature, variance_lambda=self.variance_lambda, use_random_window=self.use_random_window, use_align_alpha=self.use_align_alpha, align_alpha_strength=self.align_alpha_strength)
        self.log('val_loss', loss)
        return loss
        
    def configure_optimizers(self):
        print("Configuring optimizer...")
        parameters = list(self.model.cnn.parameters()) + list(self.model.emb.parameters())
        # TODO: use config file to manage the configs
        optimizer = optim.Adam(parameters, lr=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)
        return [optimizer], [scheduler]


    def forward(self, frames):
        embs = self.model(frames)
        return embs

def train(args):
    print("Starting training process...")
    print("Loading video files...")
    video_dir = Path('../data/GolfDB')
    video_files = list(video_dir.glob('*.mp4'))
    video_files = [file for file in video_files if int(str(file).split('/')[-1].split('.')[0]) % 2 == (0 if not args.use_120fps else 1)]
    video_files.sort(key=lambda x : int(str(x).split('/')[-1].split('.')[0]))
    video_files = video_files[:10]
    print(video_files)
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
        batch_size=8, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        collate_fn=collate_fn
    )
    
    print("Initializing model...")
    model = LitModel()
    
    print("Setting up training callbacks and logger...")
    filename = 'model-{epoch:02d}-{val_loss:.2f}'
    filename += f"{args.loss_type}_{args.similarity_type}_temp_{args.temperature}_var_{args.variance_lambda}_random_window_{args.use_random_window}_align_alpha_{args.use_align_alpha}_align_alpha_strength_{args.align_alpha_strength}_do_not_reduce_frame_rate_{args.do_not_reduce_frame_rate}"
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename=filename,
        save_top_k=3,
        mode='min',
    )
    
    logger = TensorBoardLogger("lightning_logs", name="my_model")
    
    print("Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=500,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator='gpu'
    )
    
    print("Starting model training...")
    trainer.fit(model, train_loader, train_loader)
    print("Training complete!")

def plot_loss(log_dir):
    print("Plotting loss...")

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    train_loss = ea.Scalars('train_loss')
    val_loss = ea.Scalars('val_loss')

    train_steps = [x.step for x in train_loss]
    train_values = [x.value for x in train_loss]

    val_steps = [x.step for x in val_loss]
    val_values = [x.value for x in val_loss]

    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_values, label='Train Loss')
    plt.plot(val_steps, val_values, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Train a video alignment model.")
        parser.add_argument('--loss_type', type=str, default='regression_mse_var', help='Type of loss function to use')
        parser.add_argument('--similarity_type', type=str, default='l2', help='Type of similarity function to use')
        parser.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter for contrastive loss')
        parser.add_argument('--variance_lambda', type=float, default=0.001, help='Lambda parameter for variance loss')
        parser.add_argument('--use_random_window', action='store_true', help='Whether to use random window cropping')
        parser.add_argument('--use_align_alpha', action='store_true', help='Whether to use alignment alpha')
        parser.add_argument('--align_alpha_strength', type=float, default=0.1, help='Strength of alignment alpha')
        parser.add_argument('--do_not_reduce_frame_rate', action='store_true', help='Whether to reduce frame rate to 10fps')
        parser.add_argument('--use_120fps', action='store_true', help='Whether to use 120fps videos')
        return parser.parse_args()

    args = parse_args()
    print("Script started...")
    train(args)
    print("Script complete!")
    plot_loss('lightning_logs/my_model')