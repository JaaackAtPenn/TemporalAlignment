import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import ModelWrapper
from losses import get_scaled_similarity

from typing import List, Tuple

import os
import re
import seaborn as sns

# Use dynamic time warping to find the minimum cost assignment
from dtw import dtw

def load_video(video_path: str) -> Tuple[np.ndarray, int, int]:
    """Load video and return frames, fps, and frame count."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    return np.array(frames), fps, frame_count

def extract_features(frames: np.ndarray, model: ModelWrapper) -> torch.Tensor:
    """Extract features from video frames using the provided model.
    
    Args:
        frames: Video frames of shape [T, H, W, C] 
        model: ModelWrapper instance for feature extraction
        
    Returns:
        Tensor of features with shape [T//3, embedding_dim]
    """
    # Convert frames to torch tensor and normalize
    frames = torch.from_numpy(frames).float()
    frames = frames.permute(0, 3, 1, 2)  # [T,H,W,C] -> [T,C,H,W]
    frames = frames / 255.0  # Normalize to [0,1]
    
    # Add batch dimension
    frames = frames.unsqueeze(0)  # [1,T,C,H,W]
    
    # Move to GPU if available
    if torch.cuda.is_available():
        frames = frames.cuda()
        
    # Extract features
    with torch.no_grad():
        features, cnn = model(frames)  # [1,T//3,embedding_dim]
        
    # Remove batch dimension
    features = features.squeeze(0)  # [T//3,embedding_dim]
    
    return features

def create_side_by_side_frame(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Combine two frames side by side.
    
    Args:
        frame1: First frame
        frame2: Second frame
    Returns:
        Combined frame
    """
    return np.hstack((frame1, frame2))

def save_video(frames: List[np.ndarray], output_path: str, fps: int, frame_size: Tuple[int, int]):
    """Save frames as video file.
    
    Args:
        frames: List of frames to save
        output_path: Path to save video
        fps: Frames per second
        frame_size: (width, height) of output video
    """
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        frame_size
    )
    for frame in frames:
        out.write(frame)
    out.release()

def display_frames(frames: List[np.ndarray], fps: int):
    """Display frames in a window.
    
    Args:
        frames: List of frames to display
        fps: Frames per second to display at
    """
    for frame in frames:
        cv2.imshow('Aligned Videos', frame)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def get_aligned_frames(frames1: np.ndarray, frames2: np.ndarray, index1: np.ndarray, index2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get aligned frames using computed indices.
    
    Args:
        frames1: Reference video frames
        frames2: Video frames to align
        aligned_idxs: Tensor of aligned indices
        
    Returns:
        Tuple of (reference frames, aligned frames)
    """    
    # Extract the logits from the alignment tuple
    aligned_frames2 = frames2[index2*3]  # Multiply by 3 since features are extracted every 3 frames
    aligned_frames1 = frames1[index1*3]  # Take every 3rd frame to match feature extraction
    # Make sure frames match in count
    return aligned_frames1, aligned_frames2

def align_videos(video1_path: str, video2_path: str, model: ModelWrapper, output_path: str = None):
    """Align two videos using extracted features and save aligned result.
    
    Args:
        video1_path: Path to reference video
        video2_path: Path to video to be aligned
        model: ModelWrapper instance for feature extraction
        output_path: Path to save aligned video
    """
    # Load videos
    frames1, fps1, _ = load_video(video1_path)
    frames2, fps2, _ = load_video(video2_path)
    
    # Extract features
    features1 = extract_features(frames1, model)
    features2 = extract_features(frames2, model)
    
    # Convert features to CPU if needed
    features1 = features1.cpu()
    features2 = features2.cpu()

    # Compute similarities between embs1 and embs2
    dist = get_scaled_similarity(features1, features2, 'l2', 0.1)       # [N1, N2]

    softmaxed_sim_12 = torch.nn.functional.softmax(dist, dim=1)         # alpha

    # Compute soft-nearest neighbors
    nn_embs = torch.mm(softmaxed_sim_12, features2)          # [N1, D], tilda v_i

    # Compute similarities between nn_embs and embs1
    sim_21 = get_scaled_similarity(nn_embs, features1, 'l2', 0.1)        # [N1, N1], beta before softmax

    # Find the minimum cost assignment using dynamic time warping
    dtw_result = dtw(features1, features2, dist_method='euclidean')

    index1, index2 = dtw_result.index1, dtw_result.index2

    # Make a heat map of dist
    plt.figure(figsize=(10, 10))
    sns.heatmap(sim_21, cmap='coolwarm')
    plt.title('Similarity Heatmap')
    plt.savefig(output_path + '.png')
    
    # Get aligned frames
    aligned_frames1, aligned_frames2 = get_aligned_frames(frames1, frames2, index1, index2)
    
    # Create side-by-side visualization
    combined_frames = [
        create_side_by_side_frame(f1, f2) 
        for f1, f2 in zip(aligned_frames1, aligned_frames2)
    ]
    
    if output_path:
        h, w = aligned_frames1[0].shape[:2]
        save_video(
            combined_frames,
            output_path,
            fps1//3,  # Divide FPS by 3 since we're using every 3rd frame
            (w*2, h)  # Double width for side-by-side
        )




def main():
    """Example usage."""
    import torchvision.models as models
    
    # Load feature extraction model
    model = ModelWrapper()  

    # Find the checkpoint with the smallest val_loss
    checkpoint_dir = 'checkpoints'
    checkpoint_files = os.listdir(checkpoint_dir)
    checkpoint_files = [file for file in checkpoint_files if file.startswith('model-') and file.endswith('.ckpt')]
    checkpoint_files.sort(key=lambda x: float(re.search(r'epoch=([0-9]*)', x).group(1)), reverse=True)
    print( checkpoint_files[0] )
    # Load the checkpoint with the smallest val_loss
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
    
    video_names = ['0.mp4', '2.mp4', '4.mp4', '6.mp4', '8.mp4', '10.mp4']

    # Create a folder for aligned videos
    aligned_videos_folder = 'aligned_videos'
    if not os.path.exists(aligned_videos_folder):
        os.makedirs(aligned_videos_folder)

    # Perform alignment for all pairs of videos
    for i in range(len(video_names)):
        for j in range(i + 1, len(video_names)):
            video1_path = f'../videos_160/{video_names[i]}'
            video2_path = f'../videos_160/{video_names[j]}'
            output_path = f'{aligned_videos_folder}/{video_names[i]}_{video_names[j]}.mp4'
            align_videos(
                video1_path=video1_path,
                video2_path=video2_path,
                model=model,
                output_path=output_path
            )


if __name__ == '__main__':
    main() 