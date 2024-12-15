import os
import re
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from torch.utils.data import DataLoader
from model import ModelWrapper
from losses import get_scaled_similarity
from train import PennAction, collate_fn
from torch.utils.data import DataLoader
from umap import UMAP
from train import PennAction
from align_videos import extract_features, load_video
import torch.nn.functional as F
import argparse
from ncut_pytorch import NCUT, rgb_from_tsne_3d

def extract_cnn_features(frames: np.ndarray, model: ModelWrapper) -> torch.Tensor:
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
        features = model.cnn(frames)  # [1,T//3,embedding_dim] or [1,T,embedding_dim] if using downsampling is False
        
    # Remove batch dimension
    features = features.squeeze(0)  # [T//3,embedding_dim] or [T,embedding_dim]
    
    return features

def visualize(video1_path: str, video2_path: str, model: ModelWrapper, dataset: str = 'PennAction', valonval: bool = False, ckpt_name=''):
    """Align two videos using extracted features and save aligned result.
    
    Args:
        video1_path: Path to reference video
        video2_path: Path to video to be aligned
        model: ModelWrapper instance for feature extraction
        output_path: Path to save aligned video
    """
    batch_size=4
    # Load videos
    if dataset == 'PennAction':
        train_dataset = PennAction(data_size=10000000, dont_split=True)
        print('Dataset loaded')
        print('Number of frames of each video:', len(train_dataset[0][0]))
        print('Number of videos in train dataset:', len(train_dataset))
        trainloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False)
        i = 0
        if valonval:
            i += len(train_dataset)
        end = i + 8
        loader = trainloader
        indicies = [34,145,156,19,70,91,22,124,20,59]
        for idx in range(0,len(indicies),batch_size):
            batch = [loader.dataset[indicies[idx + b]] for b in range(batch_size)]  # Access the dataset directly using the index
            frames = torch.stack([batch[b][0] for b in range(batch_size)])
            
            frames = frames.permute(0, 1, 3, 4, 2).contiguous()
            frames = frames.detach().cpu().numpy()
            frames = (frames * 255).astype('uint8')
            # check if the frames are correct
            feature_length = [0]

            combined_cnn_features = extract_cnn_features(frames[0], model)
            combined_features = extract_features(frames[0], model).cpu().numpy()
            feature_length.append(len(combined_features))

            for j in range(1,batch_size):
                cnn_features = extract_cnn_features(frames[j], model)
                combined_cnn_features = torch.cat((combined_cnn_features, cnn_features), dim=0)

                features2 = extract_features(frames[j], model).cpu().numpy()
                # Combine features for UMAP
                combined_features = np.concatenate((combined_features, features2))
                feature_length.append(len(combined_features))

            # Apply UMAP
            reducer = UMAP(n_neighbors=10, min_dist=0.1)
            embedding = reducer.fit_transform(combined_features)

            colors = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']
            # Plotting
            plt.figure(figsize=(10, 8))
            for j in range(batch_size):
                plt.scatter(embedding[feature_length[j]:feature_length[j+1], 0], embedding[feature_length[j]:feature_length[j+1], 1], c=np.linspace(0, 1, feature_length[j+1]-feature_length[j]), cmap=colors[j])
                representative_color = plt.get_cmap(colors[j])(0.5)  # Midpoint color
                plt.scatter([], [], color=representative_color, label=f'Video {j+i}')  # Dummy scatter for legend

            plt.title('UMAP Visualization of Features')
            plt.legend()
            if batch_size == 2:
                plt.savefig(f'feature_visualization/{ckpt_name}/umap_pennaction_{i}_and_{i+1}.png')
            else:
                plt.savefig(f'feature_visualization/{ckpt_name}/umap_pennaction_{batch_size}_from_{i}.png')
            i += batch_size

            # combined_cnn_features = combined_cnn_features.permute(0, 2, 3, 1) # reorder axis
            # inp = combined_cnn_features.reshape(-1, 1024) # flatten
            # for k in range(10,100,10):
            #     eigvectors, eigvalues = NCUT(num_eig=k, device='cuda:0').fit_transform(inp)
            #     tsne_x3d, tsne_rgb = rgb_from_tsne_3d(eigvectors, device='cuda:0')
            #     tsne_rgb = tsne_rgb.reshape(-1, 14, 14, 3) # (B, H, W, 3)

            #     # Save RGB t-SNE images in a big image, 5x5
            #     big_image = np.zeros((4 * 14 + 3 * 1, 10 * 14 + 9 * 1, 3), dtype=np.uint8)  # Adjusted for 10 columns with spaces between images
            #     for r in range(4):
            #         for c in range(10):
            #             index = r * 38 + c * 3
            #             if index < len(tsne_rgb):
            #                 big_image[(r * 14) + r:(r + 1) * 14 + r, (c * 14) + c:(c + 1) * 14 + c, :] = tsne_rgb[index] * 255.0  # Adjusted indices to include spaces
            #     plt.clf()
            #     plt.figure(figsize=(8, 4))
            #     plt.imshow(big_image)
            #     plt.title('ResNet Feature')
            #     plt.axis('off')
            #     plt.savefig(f'feature_visualization/{ckpt_name}/tsne_ncut_top{k}_eig.png')

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

        features1 = features1.cpu().numpy()
        features2 = features2.cpu().numpy()
        

        # Combine features for UMAP
        combined_features = np.concatenate((features1, features2))

        # Apply UMAP
        reducer = UMAP(n_neighbors=10, min_dist=0.1)
        embedding = reducer.fit_transform(combined_features)
        
        # Plotting
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:len(features1), 0], embedding[:len(features1), 1], c=np.linspace(0, 1, len(features1)), cmap='Blues', label='Video 1')
        plt.scatter(embedding[len(features1):, 0], embedding[len(features1):, 1], c=np.linspace(0, 1, len(features2)), cmap='Reds', label='Video 2')
        plt.title('UMAP Visualization of Features')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend()
        plt.savefig(f'feature_visualization/{ckpt_name}/umap_0_and_2.png')
    elif dataset == "ours":
        combined_features = []
        feature_length = [0]

        video_dir = '../data/ours'
        videos = [f for f in os.listdir(video_dir) if f.endswith('.mov')]
        videos = ['../data/ours/' + f for f in videos]
        print(f'Videos found in {video_dir}: {videos}')

        frames1, fps1, _ = load_video(videos[0])
        frames2, fps2, _ = load_video(videos[1])
        features1 = extract_features(frames1, model).cpu().numpy()        # time dimension is reduced by 3 or not
        features2 = extract_features(frames2, model).cpu().numpy()

        combined_features.append(features1)
        combined_features.append(features2)
        feature_length.append(len(features1))
        feature_length.append(len(features2))

        combined_features = np.concatenate(combined_features, axis=0)

        feature_length = np.cumsum(feature_length)
        
        # Apply UMAP for visualization
        reducer = UMAP(n_neighbors=10, min_dist=0.1)
        embedding = reducer.fit_transform(combined_features)

        # Plotting
        plt.figure(figsize=(10, 8))
        colors = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'Grays']
        for i in range(2):
            plt.scatter(embedding[feature_length[i]:feature_length[i+1], 0], 
                         embedding[feature_length[i]:feature_length[i+1], 1], 
                         c=np.linspace(0, 1, feature_length[i+1]-feature_length[i]), cmap=colors[i])
            representative_color = plt.get_cmap(colors[i])(0.8)  # Midpoint color
            plt.scatter([], [], color=representative_color, label=f'Video {i}')  # Dummy scatter for legend

            
        plt.legend()
        plt.title('UMAP Visualization of Combined Features')
        plt.savefig(f'feature_visualization/{ckpt_name}/umap_ours.png')

    else: # get all three dataset together
        combined_features = []
        feature_length = [0]

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

        loader = trainloader
        i = 0
        for batch in loader:
            if i == 0:
                i += 1
                continue
            frames, steps, seq_lens = batch
            frames = frames.permute(0, 1, 3, 4, 2).contiguous()
            frames = frames.detach().cpu().numpy()
            frames = (frames * 255).astype('uint8')
            # check if the frames are correct

            for j in range(2):
                
                features = extract_features(frames[j], model).cpu().numpy()
                # Combine features for UMAP
                combined_features.append(features)
                feature_length.append(len(features))

            break

        frames1, fps1, _ = load_video(video1_path)
        frames2, fps2, _ = load_video(video2_path)
        features1 = extract_features(frames1, model).cpu().numpy()        # time dimension is reduced by 3 or not
        features2 = extract_features(frames2, model).cpu().numpy()
        
        combined_features.append(features1)
        combined_features.append(features2)
        feature_length.append(len(features1))
        feature_length.append(len(features2))

        # Find videos in the specified directory
        video_dir = '../data/ours'
        videos = [f for f in os.listdir(video_dir) if f.endswith('.mov')]
        videos = ['../data/ours/' + f for f in videos]
        print(f'Videos found in {video_dir}: {videos}')

        frames1, fps1, _ = load_video(videos[0])
        frames2, fps2, _ = load_video(videos[1])
        features1 = extract_features(frames1, model).cpu().numpy()        # time dimension is reduced by 3 or not
        features2 = extract_features(frames2, model).cpu().numpy()

        combined_features.append(features1)
        combined_features.append(features2)
        feature_length.append(len(features1))
        feature_length.append(len(features2))

        combined_features = np.concatenate(combined_features, axis=0)

        feature_length = np.cumsum(feature_length)
        
        # Apply UMAP for visualization
        reducer = UMAP(n_neighbors=10, min_dist=0.1)
        embedding = reducer.fit_transform(combined_features)

        # Plotting
        plt.figure(figsize=(10, 8))
        colors = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'Grays']
        for i in range(6):
            plt.scatter(embedding[feature_length[i]:feature_length[i+1], 0], 
                         embedding[feature_length[i]:feature_length[i+1], 1], 
                         c=np.linspace(0, 1, feature_length[i+1]-feature_length[i]), cmap=colors[i])
            representative_color = plt.get_cmap(colors[i])(0.8)  # Midpoint color
            plt.scatter([], [], color=representative_color, label=f'Video {i}')  # Dummy scatter for legend

            
        plt.legend()
        plt.title('UMAP Visualization of Combined Features')
        plt.savefig(f'feature_visualization/{ckpt_name}/umap_all.png')




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
        parser.add_argument('--use_temporal_embedding', action='store_true', help='Whether to use temporal embedding')
        parser.add_argument('--temporal_embedding_location', type=str, default='both', choices=['front', 'back', 'both'], help='Whether to use temporal embedding')
        parser.add_argument('--valonval', action='store_true', help='Validate on validation set')
        parser.add_argument('--vidpath1', type=str, default='GolfDB/0.mp4', help='Path to the first video')
        parser.add_argument('--vidpath2', type=str, default='GolfDB/2.mp4', help='Path to the second video')
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for softmax')
        parser.add_argument('--similarity_type', type=str, default='l2', help='Type of similarity to use')
        return parser.parse_args()

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    front_temp_emb = False
    back_temp_emb = False
    if args.use_temporal_embedding:
        if args.temporal_embedding_location == 'both' or args.temporal_embedding_location == 'front':
            front_temp_emb = True
        if args.temporal_embedding_location == 'both' or args.temporal_embedding_location == 'back':
            back_temp_emb = True
    
    # Load feature extraction model
    model = ModelWrapper(front_temporal_emb=front_temp_emb, back_temporal_emb=back_temp_emb) if args.downsample else ModelWrapper(dont_stack=not args.downsample, front_temporal_emb=front_temp_emb, back_temporal_emb=back_temp_emb)

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
        ckpt_name = ''
    else:
        # Load the specified checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, args.ckpt)
        if args.mac:
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
        else:
            model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        model.eval()
        ckpt_name = (args.ckpt).split('.')[0].split('-')[1]

    if torch.cuda.is_available():
        model = model.cuda()
    elif torch.backends.mps.is_available():
        model = model.to('mps')
    
    # Align videos
    video_dir = '../data/'
    video1_path = video_dir + args.vidpath1
    video2_path = video_dir + args.vidpath2
    if not os.path.exists('feature_visualization/' + ckpt_name):
        os.mkdir('feature_visualization/' + ckpt_name)
        
    visualize(
        video1_path=video1_path,
        video2_path=video2_path,
        model=model,
        dataset=args.dataset,
        valonval=args.valonval,
        ckpt_name=ckpt_name
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