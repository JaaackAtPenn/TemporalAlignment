import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from tqdm import tqdm

from model import ModelWrapper
from losses import align_pair_of_sequences

from typing import List, Tuple


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
        features = model(frames)  # [1,T//3,embedding_dim]
        
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

def get_aligned_frames(frames1: np.ndarray, frames2: np.ndarray, aligned_idxs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Get aligned frames using computed indices.
    
    Args:
        frames1: Reference video frames
        frames2: Video frames to align
        aligned_idxs: Tensor of aligned indices
        
    Returns:
        Tuple of (reference frames, aligned frames)
    """
    # Extract the logits from the alignment tuple
    aligned_idxs = torch.argmax(aligned_idxs[0], dim=1)  # Convert logits to indices by taking argmax
    aligned_frames2 = frames2[aligned_idxs.numpy() * 3]  # Multiply by 3 since features are extracted every 3 frames
    aligned_frames1 = frames1[::3]  # Take every 3rd frame to match feature extraction
    
    # Make sure frames match in count
    min_frames = min(len(aligned_frames1), len(aligned_frames2))
    return aligned_frames1[:min_frames], aligned_frames2[:min_frames]

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
    
    # Align sequences
    aligned_idxs = align_pair_of_sequences(
        features1, 
        features2,
        similarity_type='cosine',
        temperature=0.1
    )
    
    # Get aligned frames
    aligned_frames1, aligned_frames2 = get_aligned_frames(frames1, frames2, aligned_idxs)
    
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
    model.load_state_dict(torch.load("checkpoints/model-epoch=09-val_loss=2.17.ckpt")['state_dict'])
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Align videos
    align_videos(
        video1_path='../videos_160/0.mp4',
        video2_path='../videos_160/1.mp4',
        model=model,
        output_path='aligned_videos.mp4'
    )


if __name__ == '__main__':
    main() 