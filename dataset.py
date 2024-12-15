from util import video_to_frames
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

class OurVideoDataset(Dataset):  # all the videos are trimmed to the shortest video length
    def __init__(self, video_files):
        print(f"Initializing CustomVideoDataset with {len(video_files)} videos...")
        self.video_files = video_files
        self.frames = []
        for i, video_file in enumerate(video_files):
            print(f"Processing video {i+1}/{len(video_files)}: {video_file.name}")
            self.frames.append(self.preprocess_video(video_file))
        
        # Find the shortest video length
        shortest_video_length = min(frame.shape[0] for frame in self.frames)
        
        # Trim all videos to the shortest length, discarding time from the front
        self.frames = [frame[-shortest_video_length:] for frame in self.frames]

        print("CustomVideoDataset initialization complete!")

    def preprocess_video(self, video_path):
        print(f"\tLoading frames from {video_path.name}...")
        # Load frames from video
        frames = video_to_frames(video_path)  # Assuming you have a function to read frames
        print(f"\tConverting frames to tensor... Shape: {frames.shape}")
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to torch tensor and change dimensions to [T, C, H, W]
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(0, 3, 1, 2)
        return frames

    def __len__(self):
        return len(self.video_files)
        
    def __getitem__(self, idx):
        return self.frames[idx]
    

def collate_fn_video(batch):
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

def OurDataset():
    print("Starting evaluation process with custom dataset...")
    print("Loading video files...")
    video_dir = Path('./data/ours')
    if video_dir.exists() and video_dir.is_dir():
        video_files = list(video_dir.glob('*.mov')) + list(video_dir.glob('*.MOV'))
        if not video_files:
            print("No video files found with .mov extension.")
            return None, None
        else:
            print(f"Found {len(video_files)} video files.")
    else:
        print("Directory does not exist or is not a valid directory.")
        return None, None
    
    video_files.sort(key=lambda x: x.name)
    print(f"Sorted video files: {[f.name for f in video_files]}")

    # Split into train/val sets
    num_videos = len(video_files)
    train_size = int(0.8 * num_videos)
    
    train_files = video_files[:train_size]
    val_files = video_files[train_size:]
    print(f"Split dataset: {len(train_files)} training videos, {len(val_files)} validation videos")

    print("Creating training dataset...")
    train_dataset = OurVideoDataset(train_files)
    print("Creating validation dataset...")
    val_dataset = OurVideoDataset(val_files)
    return train_dataset, val_dataset


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        """
        Combines multiple datasets into one unified dataset.

        Args:
            datasets: List of Dataset objects (e.g., PennAction, GolfDB, OurDataset)
        """
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which dataset the index belongs to
        for i, cumulative_length in enumerate(self.cumulative_lengths):
            if idx < cumulative_length:
                dataset_idx = i
                dataset_local_idx = idx if i == 0 else idx - self.cumulative_lengths[i - 1]
                break

        data = self.datasets[dataset_idx][dataset_local_idx]
        
        # Normalize the return format to match `frames` only
        if isinstance(data, tuple):  # For PennAction
            frames = data[0]
        else:  # For GolfDB and OurDataset
            frames = data

        return frames
