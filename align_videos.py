import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
from dtw import dtw
from torch.nn import functional as F
import random
from model import ModelWrapper
from losses import get_scaled_similarity
from train import PennAction, collate_fn
from torch.utils.data import DataLoader

from typing import List, Tuple

import os
import re
import argparse
import seaborn as sns


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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
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
        print('frame shape:', frames.shape)
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
    # print(f"Frame1 shape: {frame1.shape}, dtype: {frame1.dtype}")
    # print("frame1 max and min:", frame1.max(), frame1.min())
    cv2.putText(frame1, label1, (text_x1, text_y1), font, font_scale, color, thickness)
    cv2.putText(frame2, label2, (text_x2, text_y2), font, font_scale, color, thickness)
    # Plot frame1
    # plt.imshow(frame1)
    # plt.title('Frame 1')
    # plt.show()

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
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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

def get_aligned_frames(frames1: np.ndarray, frames2: np.ndarray, aligned_idxs: torch.Tensor, downsample) -> Tuple[np.ndarray, np.ndarray]:

    """Get aligned frames using computed indices.
    
    Args:
        frames1: Reference video frames
        frames2: Video frames to align
        aligned_idxs: Tensor of aligned indices
        
    Returns:
        Tuple of (reference frames, aligned frames)
    """    
    # Extract the logits from the alignment tuple
    if downsample:
        aligned_frames2 = frames2[aligned_idxs.numpy() * 3] # Multiply by 3 since features are extracted every 3 frames
        aligned_frames1 = frames1[::3]  # Take every 3rd frame to match feature extraction
    else:
        aligned_frames2 = frames2[aligned_idxs.numpy()]
        aligned_frames1 = frames1


    # Make sure frames match in count
    return aligned_frames1, aligned_frames2

def euclidean_distance(x, y):
    x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    y = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    return np.sqrt(np.sum((x - y) ** 2))

def align_videos(video1_path: str, video2_path: str, model: ModelWrapper, output_path: str = None, downsample: bool = True, dataset: str = 'PennAction', valonval: bool = False, use_dtw: bool = False, temperature: float = 0.1, similarity_type: str = 'cosine'):
    """Align two videos using extracted features and save aligned result.
    
    Args:
        video1_path: Path to reference video
        video2_path: Path to video to be aligned
        model: ModelWrapper instance for feature extraction
        output_path: Path to save aligned video
    """
    # Load videos
    if dataset == 'PennAction':
        train_dataset, val_dataset = PennAction(data_size=10000000, dont_split=False)
        print('Dataset loaded')
        print('Number of frames of each video:', len(train_dataset[0][0]))
        print('Number of videos in train dataset:', len(train_dataset))
        print('Number of videos in val dataset:', len(val_dataset))
        trainloader = DataLoader(
            train_dataset, 
            batch_size=2, 
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False)
        valloader = DataLoader(
            val_dataset,
            batch_size=2,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False)
        i = 789
        if valonval:
            i += len(train_dataset)
        end = i + 10
        if valonval:
            loader = valloader
        else:
            loader = trainloader
        for batch in loader:
            frames, steps, seq_lens = batch
            frames = frames.permute(0, 1, 3, 4, 2).contiguous()
            frames = frames.detach().cpu().numpy()
            frames = (frames * 255).astype('uint8')
            # check if the frames are correct
            fps = 10 if downsample else 30
            h, w = frames[0][0].shape[:2]
            output_path = 'result/frames{}.mp4'.format(i)
            save_video(
                frames[0],
                output_path,
                fps,  
                (w, h)   # Double width for side-by-side
            )
            output_path = 'result/frames{}.mp4'.format(i+1)
            save_video(
                frames[1],
                output_path,
                fps,  
                (w, h)  # Double width for side-by-side
            )
            # # Plot the frames to check if they are correct
            # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # axes[0].imshow(frames[0][0])
            # axes[0].set_title('First Frame of Video 1')
            # axes[1].imshow(frames[1][0])
            # axes[1].set_title('First Frame of Video 2')
            # fig.savefig('result/frames{}and{}.png'.format(i, i+1))
            steps1 = steps[0].numpy()
            # print('steps1:', steps1)
            steps2 = steps[1].numpy()
            # print('steps2:', steps2)
            features1 = extract_features(frames[0], model)
            features2 = extract_features(frames[1], model)
            features1 = features1.cpu()
            features2 = features2.cpu()
            # print('features1:', features1.shape)
            # print('features2:', features2.shape)
            sim12 = get_scaled_similarity(features1, features2, similarity_type, temperature)      # [N1, N2]
            sim12 = F.softmax(sim12, dim=1)
            sim11 = get_scaled_similarity(features1, features1, similarity_type, temperature)
            sim11 = F.softmax(sim11, dim=1)
            sim22 = get_scaled_similarity(features2, features2, similarity_type, temperature)
            sim22 = F.softmax(sim22, dim=1)

            selected_features2 = torch.mm(sim12, features2)          # [N1, D], tilda v_i
            cycle_sim11 = get_scaled_similarity(selected_features2, features1, similarity_type, temperature)       # [N1, N1]
            cycle_sim11 = F.softmax(sim11, dim=1)
            # print('cycle_dist:', cycle_dist)

            # Plot distance matrix as heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(sim12.numpy(), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title('Similarity HeatMap')
            plt.xlabel('Frames of Video {}'.format(i))
            plt.ylabel('Frames of Video {}'.format(i+1))
            plt.savefig('result/similarity_heatmap_{}and{}.png'.format(i, i+1))            
            # plt.show()

            # Plot self distance matrix as heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(sim11.numpy(), cmap='hot', interpolation='nearest', vmax=1)
            plt.colorbar()
            plt.title('Self Similarity HeatMap of Video {}'.format(i))
            plt.xlabel('Frames of Video {}'.format(i))
            plt.ylabel('Frames of Video {}'.format(i))
            plt.savefig('result/self_similarity_heatmap_{}.png'.format(i))
            # plt.show()

            # Plot self distance matrix as heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(sim22.numpy(), cmap='hot', interpolation='nearest', vmax=1)
            plt.colorbar()
            plt.title('Self Similarity HeatMap of Video {}'.format(i+1))
            plt.xlabel('Frames of Video {}'.format(i+1))
            plt.ylabel('Frames of Video {}'.format(i+1))
            plt.savefig('result/self_similarity_heatmap_{}.png'.format(i+1))
            # plt.show()

            # Plot cycle distance matrix as heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(cycle_sim11.numpy(), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title('Cycle Similarity HeatMap')
            plt.xlabel('Frames of Video {}'.format(i))
            plt.ylabel('Frames of Video {}'.format(i))
            plt.savefig('result/cycle_similarity_heatmap_{}and{}.png'.format(i, i+1))
            # plt.show()

            matches = sim12.argmax(dim=1)
            # print('matches:', matches)
            aligned_frames1, aligned_frames2 = get_aligned_frames(frames[0], frames[1], matches, downsample)
            # print('If downsample:', downsample)
            if downsample:
                frame1_indices = steps1[np.arange(1, len(aligned_frames1) * 3, 3)]
                frame2_indices_cur = steps2[matches.numpy() * 3 + 1]
            else:
                frame1_indices = steps1[np.arange(0, len(aligned_frames1))]
                frame2_indices_cur = steps2[matches.numpy()]
                # print('frame1_indices:', frame1_indices)
                # print('frame2_indices_cur:', frame2_indices_cur)
            frame2_indices_last = np.roll(frame2_indices_cur, 1)
            combined_frames = [
                create_side_by_side_frame(f1, f2, f1i, f2ic, f2il) 
                for f1, f2, f1i, f2ic, f2il in zip(aligned_frames1, aligned_frames2, frame1_indices, frame2_indices_cur, frame2_indices_last)
            ]
            output_path = 'result/aligned_videos{}and{}DTW{}.mp4'.format(i, i+1, use_dtw)
            # plt.imshow(combined_frames[0])
            # plt.title('First Frame of Combined Frames')
            # plt.show()
            save_video(
                combined_frames,
                output_path,
                fps,  
                (w*2, h)  # Double width for side-by-side
            )
            i += 2
            if i >= end:
                break
    elif dataset == 'GolfDB':
        frames1, fps1, _ = load_video(video1_path)
        frames2, fps2, _ = load_video(video2_path)
        # plt.imshow(frames1[0])
        # plt.title('First Frame of Video 1')
        # plt.show()
        print('frame size:', frames1[0].shape)
    
        # Extract features
        features1 = extract_features(frames1, model)         # time dimension is reduced by 3 or not
        features2 = extract_features(frames2, model)
        
        # Convert features to CPU if needed
        features1 = features1.cpu()
        features2 = features2.cpu()

        # Compute similarities between embs1 and embs2
        sim12 = get_scaled_similarity(features1, features2, similarity_type, temperature)       # [N1/3, N2/3] or [N1, N2]
        sim12 = F.softmax(sim12, dim=1)
        sim11 = get_scaled_similarity(features1, features1, similarity_type, temperature)
        sim11 = F.softmax(sim11, dim=1)

        selected_features2 = torch.mm(sim12, features2)          # [N1, D], tilda v_i
        cycle_sim11 = get_scaled_similarity(selected_features2, features1, similarity_type, temperature)       # [N1, N1]
        cycle_sim11 = F.softmax(cycle_sim11, dim=1)

        # Plot distance matrix as heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(sim12.numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Distance Heatmap')
        plt.xlabel('Frames of Video 2')
        plt.ylabel('Frames of Video 1')
        plt.savefig(output_path[:-4] + '_distance_heatmap.png')
        # plt.show()

        # Plot self distance matrix as heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(sim11.numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Self Distance Heatmap')
        plt.xlabel('Frames of Video 1')
        plt.ylabel('Frames of Video 1')
        plt.savefig(output_path[:-4] + '_self_distance_heatmap.png')
        # plt.show()

        # Plot cycle distance matrix as heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(cycle_sim11.numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Cycle Distance Heatmap')
        plt.xlabel('Frames of Video 1')
        plt.ylabel('Frames of Video 1')
        plt.savefig(output_path[:-4] + '_cycle_distance_heatmap.png')
        # plt.show()

        if use_dtw:
            # Find the minimum cost assignment using dynamic time warping
            dtw_result = dtw(features1, features2, dist=euclidean_distance)
            # print('dtw_result:', dtw_result)
            index1, index2 = dtw_result[-1]
            print('index1:', index1)
            print('index1 shape:', index1.shape)
            print('index2:', index2)
            print('index2 shape:', index2.shape)
            
            # Get aligned frames
            aligned_frames1, aligned_frames2 = frames1[index1], frames2[index2]

            # Create side-by-side visualization
            if downsample:
                frame1_indices = index1 * 3 + 1
                frame2_indices_cur = index2 * 3 + 1
            else:
                frame1_indices = index1
                frame2_indices_cur = index2
            frame2_indices_last = np.roll(frame2_indices_cur, 1)
            combined_frames = [
                create_side_by_side_frame(f1, f2, f1i, f2ic, f2il) 
                for f1, f2, f1i, f2ic, f2il in zip(aligned_frames1, aligned_frames2, frame1_indices, frame2_indices_cur, frame2_indices_last)
            ]
        else:
            matches = sim12.argmax(1)      # [N1/3] or [N1]
            
            # Get aligned frames
            aligned_frames1, aligned_frames2 = get_aligned_frames(frames1, frames2, matches, downsample)
        
            # Create side-by-side visualization
            if downsample:
                frame1_indices = np.arange(1, len(aligned_frames1) * 3, 3)      # starts from the second frame actually
                frame2_indices_cur = matches.numpy() * 3 + 1
            else:
                frame1_indices = np.arange(0, len(aligned_frames1))
                frame2_indices_cur = matches.numpy()
            frame2_indices_last = np.roll(frame2_indices_cur, 1)
            combined_frames = [
                create_side_by_side_frame(f1, f2, f1i, f2ic, f2il) 
                for f1, f2, f1i, f2ic, f2il in zip(aligned_frames1, aligned_frames2, frame1_indices, frame2_indices_cur, frame2_indices_last)
            ]
        
        if downsample:
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
        parser.add_argument('--downsample', action='store_true', help='Use downsampling for feature extraction')
        parser.add_argument('--dataset', type=str, default='PennAction', help='Dataset to use for alignment')
        # parser.add_argument('--data_size', type=int, default=0, help='Number of videos to test, 0 for all')
        parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint to load')
        parser.add_argument('--mac', action='store_true', help='Use mac to run the code')
        parser.add_argument('--valonval', action='store_true', help='Validate on validation set')
        parser.add_argument('--vidpath1', type=str, default='GolfDB/0.mp4', help='Path to the first video')
        parser.add_argument('--vidpath2', type=str, default='GolfDB/2.mp4', help='Path to the second video')
        parser.add_argument('--use_dtw', action='store_true', help='Use dynamic time warping for alignment')
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for softmax')
        parser.add_argument('--similarity_type', type=str, default='l2', help='Type of similarity to use')
        return parser.parse_args()

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load feature extraction model
    model = ModelWrapper() if args.downsample else ModelWrapper(dont_stack=not args.downsample)

    # Find the checkpoint with the smallest val_loss
    checkpoint_dir = 'checkpoints'
    checkpoint_files = os.listdir(checkpoint_dir)

    if args.ckpt is None:
        checkpoint_files = [file for file in checkpoint_files if file.startswith('model-') and file.endswith('.ckpt')]
        checkpoint_files.sort(key=lambda x: float(re.search(r'epoch=([0-9])', x).group(1)), reverse=True)

        # Load the checkpoint with the smallest val_loss
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        model.eval()
    else:
        # Load the specified checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, args.ckpt)
        if args.mac:
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
        else:
            model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        model.eval()


    if torch.cuda.is_available():
        model = model.cuda()
    elif torch.backends.mps.is_available():
        model = model.to('mps')
    
    # Align videos
    video_dir = '../data/'
    video1_path = video_dir + args.vidpath1
    video2_path = video_dir + args.vidpath2
    output_path = args.vidpath1.split('/')[1][:-4] + '&' + args.vidpath2.split('/')[1][:-4] + 'withDTW' + str(args.use_dtw) + '.mp4'
    output_path = 'result/' + output_path
    align_videos(
        video1_path=video1_path,
        video2_path=video2_path,
        model=model,
        output_path=output_path,
        downsample=args.downsample, 
        dataset=args.dataset,
        valonval=args.valonval,
        use_dtw=args.use_dtw,
        temperature=args.temperature,
        similarity_type=args.similarity_type
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