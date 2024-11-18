import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from tqdm import tqdm

from model import Encoder
from align import get_scaled_similarity, align_pair_of_sequences


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


def extract_features(frames: np.ndarray, model: torch.nn.Module) -> torch.Tensor:
    """Extract features from video frames using provided model."""
    device = next(model.parameters()).device
    features = []
    
    for frame in frames:
        # Preprocess frame (adjust as needed for your model)
        frame = cv2.resize(frame, (224, 224))
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame = frame.unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            feat = model(frame)
        features.append(feat.cpu())
    
    return torch.cat(features, dim=0)


def align_videos(
    video1_path: str,
    video2_path: str,
    model: torch.nn.Module,
    output_path: str,
    similarity_type: str = 'cosine',
    temperature: float = 0.1,
    display: bool = False
) -> None:
    """Align two videos and save or display the result.
    
    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        model: Feature extraction model
        output_path: Path to save aligned video
        similarity_type: Type of similarity metric to use
        temperature: Temperature for scaling similarities
        display: Whether to display video while processing
    """
    # Load videos
    frames1, fps1, _ = load_video(video1_path)
    frames2, fps2, _ = load_video(video2_path)
    
    # Extract features
    features1 = extract_features(frames1, model)
    features2 = extract_features(frames2, model)
    
    # Compute alignment
    sim_matrix, _ = align_pair_of_sequences(
        features1, features2, 
        similarity_type=similarity_type,
        temperature=temperature
    )
    
    # Get alignment indices
    alignment_indices = torch.argmax(sim_matrix, dim=1).numpy()
    
    # Create output video
    height = max(frames1.shape[1], frames2.shape[1])
    width = frames1.shape[2] + frames2.shape[2]
    fps = max(fps1, fps2)
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create aligned video
    for i, idx in enumerate(tqdm(alignment_indices)):
        # Get frames
        frame1 = frames1[i]
        frame2 = frames2[idx]
        
        # Resize frames to same height
        frame1 = cv2.resize(frame1, (int(height * frame1.shape[1] / frame1.shape[0]), height))
        frame2 = cv2.resize(frame2, (int(height * frame2.shape[1] / frame2.shape[0]), height))
        
        # Concatenate frames
        combined = np.hstack([frame1, frame2])
        
        if display:
            cv2.imshow('Aligned Videos', combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if output_path:
            out.write(combined)
    
    if output_path:
        out.release()
    
    if display:
        cv2.destroyAllWindows()


def main():
    """Example usage."""
    import torchvision.models as models
    
    # Load feature extraction model
    model = Encoder()
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Align videos
    align_videos(
        video1_path='path/to/video1.mp4',
        video2_path='path/to/video2.mp4',
        model=model,
        output_path='aligned_videos.mp4',
        display=True  # Set to False to only save without displaying
    )


if __name__ == '__main__':
    main() 