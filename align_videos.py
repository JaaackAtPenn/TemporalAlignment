import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from tqdm import tqdm

from model import ModelWrapper
from losses import get_scaled_similarity

from typing import List, Tuple

import os
import re
import argparse

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
    elif torch.backends.mps.is_available():
        frames = frames.to('mps')
        
    # Extract features
    with torch.no_grad():
        features = model(frames)  # [1,T//3,embedding_dim] or [1,T,embedding_dim] if using downsampling is False
        
    # Remove batch dimension
    features = features.squeeze(0)  # [T//3,embedding_dim] or [T,embedding_dim]
    
    return features

def create_side_by_side_frame(frame1: np.ndarray, frame2: np.ndarray, frame1_index, frame2_index_cur, frame2_index_last) -> np.ndarray:
    """Combine two frames side by side.
    
    Args:
        frame1: First frame
        frame2: Second frame
    Returns:
        Combined frame
    """

    # Create labels
    label1 = f"cur:{frame1_index}"
    label2 = f"cur:{frame2_index_cur} last:{frame2_index_last}"
    
    # Set font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 0)  
    thickness = 1
    
    # Get text size
    text_size1 = cv2.getTextSize(label1, font, font_scale, thickness)[0]
    text_size2 = cv2.getTextSize(label2, font, font_scale, thickness)[0]
    
    # Set text positions
    text_x1 = frame1.shape[1] - text_size1[0] - 10
    text_y1 = 20
    text_x2 = frame2.shape[1] - text_size2[0] - 10
    text_y2 = 20
    
    # Add text to frames
    cv2.putText(frame1, label1, (text_x1, text_y1), font, font_scale, color, thickness)
    cv2.putText(frame2, label2, (text_x2, text_y2), font, font_scale, color, thickness)

    return np.hstack((frame1, frame2))

def save_frame_with_label(video_path: str, frame_index: int):
    """Save the fth frame of a video to an image file with a label.
    
    Args:
        video_path: Path to the video file
        frame_index: Index of the frame to save
        output_path: Path to save the frame image
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame_index%3 == 0:
        # Add label to frame
        label = f"Frame: {frame_index}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 0, 0)  
        thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = 10
        text_y = 20
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, color, thickness)
        
        cv2.imwrite(f'frame_{frame_index}ofVideo{video_path.split("/")[-1][:-4]}.png', frame)
    else:
        raise ValueError(f"Frame at index {frame_index} could not be read.")
    
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

def get_aligned_frames(frames1: np.ndarray, frames2: np.ndarray, aligned_idxs: torch.Tensor, use_desampling) -> Tuple[np.ndarray, np.ndarray]:
    """Get aligned frames using computed indices.
    
    Args:
        frames1: Reference video frames
        frames2: Video frames to align
        aligned_idxs: Tensor of aligned indices
        
    Returns:
        Tuple of (reference frames, aligned frames)
    """    
    # Extract the logits from the alignment tuple
    if use_desampling:
        aligned_frames2 = frames2[aligned_idxs.numpy() * 3] # Multiply by 3 since features are extracted every 3 frames
        aligned_frames1 = frames1[::3]  # Take every 3rd frame to match feature extraction
    else:
        aligned_frames2 = frames2[aligned_idxs.numpy()]
        aligned_frames1 = frames1

    # Make sure frames match in count
    return aligned_frames1, aligned_frames2

def align_videos(video1_path: str, video2_path: str, model: ModelWrapper, output_path: str = None, use_desampling: bool = True):
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
    features1 = extract_features(frames1, model)         # time dimension is reduced by 3 or not
    features2 = extract_features(frames2, model)
    
    # Convert features to CPU if needed
    features1 = features1.cpu()
    features2 = features2.cpu()

    # Compute similarities between embs1 and embs2
    dist = get_scaled_similarity(features1, features2, 'l2', 0.1)       # [N1/3, N2/3] or [N1, N2]
    matches = dist.argmin(1)      # [N1/3] or [N1]
    
    # Get aligned frames
    aligned_frames1, aligned_frames2 = get_aligned_frames(frames1, frames2, matches, use_desampling)
    
    # Create side-by-side visualization
    if use_desampling:
        frame1_indices = np.arange(0, len(aligned_frames1) * 3, 3)
        frame2_indices_cur = matches.numpy() * 3
    else:
        frame1_indices = np.arange(0, len(aligned_frames1))
        frame2_indices_cur = matches.numpy()
    frame2_indices_last = np.roll(frame2_indices_cur, 1)
    combined_frames = [
        create_side_by_side_frame(f1, f2, f1i, f2ic, f2il) 
        for f1, f2, f1i, f2ic, f2il in zip(aligned_frames1, aligned_frames2, frame1_indices, frame2_indices_cur, frame2_indices_last)
    ]
    
    if use_desampling:
        fps1 = fps1 // 3
    if output_path:
        h, w = aligned_frames1[0].shape[:2]
        save_video(
            combined_frames,
            output_path,
            fps1,  # Divide FPS by 3 since we're using every 3rd frame
            (w*2, h)  # Double width for side-by-side
        )

# def check_aligned_frame(video1_path: str, video2_path: str, model: ModelWrapper, i: int):
#     """Check the ith frame in video1 and its aligned frame in video2.
    
#     Args:
#         video1_path: Path to reference video
#         video2_path: Path to video to be aligned
#         model: ModelWrapper instance for feature extraction
#         i: Frame index to check (must be a multiple of 3)
#     """
#     if i % 3 != 0:
#         raise ValueError("Frame index i must be a multiple of 3")
    
#     # Load videos
#     frames1, _, _ = load_video(video1_path)
#     frames2, _, _ = load_video(video2_path)
    
#     # Extract features
#     features1 = extract_features(frames1, model)
#     features2 = extract_features(frames2, model)
    
#     # Convert features to CPU if needed
#     features1 = features1.cpu()
#     features2 = features2.cpu()
    
#     # Compute similarities between features
#     dist = get_scaled_similarity(features1, features2, 'l2', 0.1)
#     matches = dist.argmin(1)
    
#     # Get aligned frame index
#     aligned_idx = matches[i // 3].item() * 3
    
#     # Display the frames
#     frame1 = frames1[i]
#     frame2 = frames2[aligned_idx]
    
#     combined_frame = create_side_by_side_frame(frame1, frame2)
#     cv2.imshow('Aligned Frame', combined_frame)
#     cv2.imwrite(f'aligned_frame_{i}.png', combined_frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def main():
    """Example usage."""
    import torchvision.models as models

    def parse_args():
        parser = argparse.ArgumentParser(description="Align videos with optional downsampling.")
        parser.add_argument('--use_downsampling', action='store_true', help='Use downsampling for feature extraction')
        return parser.parse_args()

    args = parse_args()
    
    # Load feature extraction model
    model = ModelWrapper()  

    # Find the checkpoint with the smallest val_loss
    checkpoint_dir = 'checkpoints'
    checkpoint_files = os.listdir(checkpoint_dir)
    checkpoint_files = [file for file in checkpoint_files if file.startswith('model-') and file.endswith('.ckpt')]
    checkpoint_files.sort(key=lambda x: float(re.search(r'epoch=([0-9])', x).group(1)), reverse=True)

    # Load the checkpoint with the smallest val_loss
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
    elif torch.backends.mps.is_available():
        model = model.to('mps')
    
    # Align videos
    align_videos(
        video1_path='../data/GolfDB/1.mp4',
        video2_path='../data/GolfDB/3.mp4',
        model=model,
        output_path='aligned_videos.mp4'
        use_desampling=args.use_downsampling
    )

    # # Check aligned frame
    # check_aligned_frame(
    #     video1_path='../data/GolfDB/0.mp4',
    #     video2_path='../data/GolfDB/2.mp4',
    #     model=model,
    #     i=90
    # )


if __name__ == '__main__':
    main() 
    # save_frame_with_label('../data/GolfDB/3.mp4', 177)