import torch
from torch.utils.data import DataLoader
from scipy.stats import kendalltau
import numpy as np
from align_videos import extract_features
from losses import get_scaled_similarity
import argparse
from pathlib import Path
from model import ModelWrapper
from train import PennAction, DBgolf, collate_fn
from dataset import OurDataset, collate_fn_video, CombinedDataset
import random
import os
import re

def evaluate_kendalls_tau(loader, model, similarity_type='cosine', temperature=0.001, dataset='all'):
    all_taus = []

    for batch_idx, batch in enumerate(loader):
        print(f"Processing batch {batch_idx + 1}/{len(loader)}...")
        # TODO:
        if dataset == 'PennAction':
            frames, _, _ = batch
        else:
            frames = batch

        frames = frames.permute(0, 1, 3, 4, 2).contiguous()
        frames = frames.detach().cpu().numpy()
        frames = (frames * 255).astype('uint8')
        
        features1 = extract_features(frames[0], model)
        features2 = extract_features(frames[1], model)
        features1 = features1.cpu()
        features2 = features2.cpu()
        
        sim12 = get_scaled_similarity(features1, features2, similarity_type, temperature)
        sim12 = torch.nn.functional.softmax(sim12, dim=1)
        
        matches = sim12.argmax(dim=1)
        
        tau, _ = kendalltau(range(len(matches)), matches.numpy())
        print(f"Kendall's Tau for batch {batch_idx + 1}: {tau:.4f}")
        all_taus.append(tau)

    # Compute average Kendall's Tau
    all_taus = np.array(all_taus)
    avg_tau = np.nanmean(all_taus)  # Ignore NaNs, if any
    print(f"Average Kendall's Tau: {avg_tau:.4f}")
    
    return avg_tau

def main():
    def parse_args():
        parser = argparse.ArgumentParser(description="Align videos with optional downsampling.")
        parser.add_argument('--downsample', action='store_true', help='Use downsampling for feature extraction')
        parser.add_argument('--dataset', type=str, default='all', help='Dataset to use for alignment')
        # parser.add_argument('--data_size', type=int, default=0, help='Number of videos to test, 0 for all')
        parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint to load')
        parser.add_argument('--mac', action='store_true', help='Use mac to run the code')
        parser.add_argument('--valonval', action='store_true', help='Validate on validation set')
        parser.add_argument('--vidpath1', type=str, default='GolfDB/0.mp4', help='Path to the first video')
        parser.add_argument('--vidpath2', type=str, default='GolfDB/2.mp4', help='Path to the second video')
        parser.add_argument('--use_dtw', action='store_true', help='Use dynamic time warping for alignment')
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--temperature', type=float, default=0.001, help='Temperature for softmax')
        parser.add_argument('--similarity_type', type=str, default='l2', help='Type of similarity to use')
        parser.add_argument('--use_120fps', type=bool, default=True, help='Used for loading GolfDB: Whether to use 120fps videos')
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

    if args.dataset == 'all':
        penn_train, penn_val = PennAction(data_size=10000000, dont_split=False)
        golf_train, golf_val = DBgolf(args)
        our_train, our_val = OurDataset()

        print("Combining datasets...")
        train_dataset = CombinedDataset([penn_train, golf_train, our_train])
        val_dataset = CombinedDataset([penn_val, golf_val, our_val])
    else:
        if args.dataset == 'PennAction':
            train_dataset, val_dataset = PennAction(data_size=10000000, dont_split=False)
        elif args.dataset == 'GolfDB':
            train_dataset, val_dataset = DBgolf(args)
        elif args.dataset == 'ours':
            train_dataset, val_dataset = OurDataset()
        else:
            raise NotImplementedError
    print('Dataset loaded')
    print('Number of frames of each video:', len(train_dataset[0][0]))
    print('Number of videos in train dataset:', len(train_dataset))
    print('Number of videos in val dataset:', len(val_dataset))

    if args.valonval:
        dataset = val_dataset
    else:
        dataset = train_dataset

    if args.dataset == 'PennAction':
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False)
    elif args.dataset == 'GolfDB' or args.dataset == 'ours' or args.dataset == 'all' :
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_fn_video,
            drop_last=True,
            shuffle=False)
    else:
        raise NotImplementedError

    # Evaluate Kendall's Tau
    print("Evaluating Kendall's Tau...")
    avg_tau = evaluate_kendalls_tau(
        loader=loader,
        model=model,
        similarity_type=args.similarity_type,
        temperature=args.temperature,
        dataset=args.dataset
    )
    print(f"Final Average Kendall's Tau: {avg_tau:.4f}")

if __name__ == "__main__":
    main()