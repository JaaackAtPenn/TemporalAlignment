import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from model import ModelWrapper
from torchvision import transforms
from losses import compute_alignment_loss
from util import video_to_frames
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import argparse
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import random

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
        self.shortest_sequence_length = min(frame.shape[0] for frame in self.frames)

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
        sequence = self.frames[idx]

        if len(sequence) > self.shortest_sequence_length:
            sampled_indices = sorted(random.sample(range(len(sequence)), self.shortest_sequence_length))
            sampled_sequence = torch.stack([sequence[i] for i in sampled_indices])
        else:
            sampled_indices = list(range(len(sequence)))
            sampled_sequence = sequence
        
        return sampled_sequence, sampled_indices, len(sequence)   

def collate_fn(batch):
    # Pad sequences to the same length
    # Each item in batch is [T,C,H,W]
    # print("Batch shapes:")
    # for item in batch:
    #     print(item.shape)
    
    sequences, sampled_indices, seq_lens = zip(*batch)
    
    # Get max sequence length in this batch
    max_len = max([seq.shape[0] for seq in sequences])
    C, H, W = sequences[0].shape[1:]
    
    # Create padded batch
    padded_batch = torch.zeros(len(sequences), max_len, C, H, W)
    sampled_indices_tensor = torch.zeros(len(sequences), max_len)
    seq_lens_tensor = torch.tensor(seq_lens)
    for i, seq in enumerate(sequences):
        T = seq.shape[0]
        padded_batch[i, :T] = seq
        sampled_indices_tensor[i, :T] = torch.tensor(sampled_indices[i])

    return padded_batch, sampled_indices_tensor, seq_lens_tensor

class FrameSequenceDataset(Dataset):
    def __init__(self, root_dir, resize=(224, 224), use_golf_folders=True, data_size=0):
        """
        Args:
            root_dir (str): Path to the root directory containing video folders.
            resize (tuple): Resize each frame to this size (height, width).
        """
        print(f"Initializing dataset from {root_dir}...")
        self.root_dir = root_dir
        self.resize = resize
        self.sequences = []  # each item is a list of image paths

        # get golf folders
        if use_golf_folders:
            self.sequences_folder = self._filter_folders()
        else:
            self.sequences_folder = self._all_folders()        # try to train with all actions in PennAction like in the paper, maybe it will learn the similarity between poses better
        if data_size > 0:
            self.sequences_folder = self.sequences_folder[:data_size]
        # get all image files in this sequence, sorted by name
        for sequence_folder in self.sequences_folder:
            sequence_path = os.path.join(root_dir, sequence_folder)
            if os.path.isdir(sequence_path):
                frame_files = sorted(glob.glob(os.path.join(sequence_path, "*.jpg")))
                if frame_files:
                    self.sequences.append(frame_files)
        
        self.shortest_sequence_length = min(len(seq) for seq in self.sequences)
        print(f"Shortest sequence length: {self.shortest_sequence_length}")

        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),  
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns:
            frames: Tensor of shape [T, C, H, W], where T is the number of frames.
        """
        sequence = self.sequences[idx]

        if len(sequence) > self.shortest_sequence_length:
            sampled_indices = sorted(random.sample(range(len(sequence)), self.shortest_sequence_length))
            sampled_sequence = [sequence[i] for i in sampled_indices]
        else:
            sampled_indices = list(range(len(sequence)))
            sampled_sequence = sequence

        frames = []
        for img_path in sampled_sequence[:self.shortest_sequence_length]:
            img = Image.open(img_path).convert("RGB")
            # plt.imshow(img)
            # plt.show()
            # print(f"Before transform: Image size: {img.size}, mode: {img.mode}")
            img = self.transform(img)
            # print(f"After transform: Tensor shape: {img.shape}, dtype: {img.dtype}")
            frames.append(img)
            # print(f"Image shape: {img.shape}")
            # print(f"Image min: {img.min()}, max: {img.max()}")
            # print(f"Image dot sample: {img[0, 0, 0]}")
            # print(f"Image dot type: {img.dtype}")

        frames = torch.stack(frames)
        return frames, sampled_indices, len(sequence)         # sampled_indices and len(sequence) are used for alignment loss
    
    def _filter_folders(self):
        folders = [folder for folder in self.root_dir.iterdir() if folder.is_dir()]
        filtered_folders = [
            folder.resolve() for folder in folders
            # Glof video indices: 0789 - 0954
            if 789 <= int(folder.name) <= 954
        ]
        return sorted(filtered_folders, key=lambda x: int(x.name))
    
    def _all_folders(self):
        folders = [folder for folder in self.root_dir.iterdir() if folder.is_dir()]
        filtered_folders = [
            folder.resolve() for folder in folders
            # Glof video indices: 0789 - 0954
            # if 789 <= int(folder.name) <= 954
        ]
        return sorted(filtered_folders, key=lambda x: int(x.name))
    
class LitModel(pl.LightningModule):
    def __init__(self, model=None, loss_type='regression_mse_var', similarity_type='l2', temperature=0.1, variance_lambda=0.001, use_random_window=False, use_align_alpha=False, align_alpha_strength=0.1, do_not_reduce_frame_rate=False, small_embedder=False, dont_stack=False):
        super().__init__()
        print("Initializing LitModel...")
        self.model = model if model else ModelWrapper(do_not_reduce_frame_rate=do_not_reduce_frame_rate, small_embedder=small_embedder, dont_stack=dont_stack)
        self.loss_type = loss_type
        self.similarity_type = similarity_type
        self.temperature = temperature
        self.variance_lambda = variance_lambda
        self.use_random_window = use_random_window
        self.use_align_alpha = use_align_alpha
        self.align_alpha_strength = align_alpha_strength
        # self.training_outputs = []
        # self.validation_outputs = []
                
    def training_step(self, batch, batch_idx):
        x, steps, seq_lens = batch
        y_hat = self(x)
        similarity_loss, time_loss = compute_alignment_loss(y_hat, steps=steps, seq_lens=seq_lens, batch_size=x.shape[0], loss_type=self.loss_type, similarity_type=self.similarity_type, temperature=self.temperature, variance_lambda=self.variance_lambda, use_random_window=self.use_random_window, use_align_alpha=self.use_align_alpha, align_alpha_strength=self.align_alpha_strength)
        total_loss = similarity_loss + time_loss
        self.log('train_similarity_loss', similarity_loss)
        self.log('train_time_loss', time_loss)
        self.log('train_total_loss', total_loss)
        # self.training_outputs.append({'loss': total_loss, 'similarity_loss': similarity_loss, 'time_loss': time_loss})
        # return {'loss': total_loss, 'similarity_loss': similarity_loss, 'time_loss': time_loss}
        return total_loss
    
    # def on_train_epoch_end(self):
    #     avg_loss = torch.stack([x['loss'] for x in self.training_outputs]).mean()
    #     avg_similarity_loss = torch.stack([x['similarity_loss'] for x in self.training_outputs]).mean()
    #     avg_time_loss = torch.stack([x['time_loss'] for x in self.training_outputs]).mean()

    #     print(f"Epoch {self.current_epoch}: Avg Loss: {avg_loss}, Avg Similarity Loss: {avg_similarity_loss}, Avg Time Loss: {avg_time_loss}")
        
    #     self.training_outputs.clear()
        
    def validation_step(self, batch, batch_idx):
        x, steps, seq_lens = batch
        with torch.no_grad():
            y_hat = self(x)
        similarity_loss, time_loss = compute_alignment_loss(y_hat, steps=steps, seq_lens=seq_lens, batch_size=x.shape[0], loss_type=self.loss_type, similarity_type=self.similarity_type, temperature=self.temperature, variance_lambda=self.variance_lambda, use_random_window=self.use_random_window, use_align_alpha=self.use_align_alpha, align_alpha_strength=self.align_alpha_strength)
        total_loss = similarity_loss + time_loss
        self.log('val_similarity_loss', similarity_loss)
        self.log('val_time_loss', time_loss)
        self.log('val_total_loss', total_loss)
        # self.validation_outputs.append({'loss': total_loss, 'similarity_loss': similarity_loss, 'time_loss': time_loss})
        # return {'loss': total_loss, 'similarity_loss': similarity_loss, 'time_loss': time_loss}
        return total_loss
    
    # def on_validation_epoch_end(self):
    #     avg_loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean()
    #     avg_similarity_loss = torch.stack([x['similarity_loss'] for x in self.validation_outputs]).mean()
    #     avg_time_loss = torch.stack([x['time_loss'] for x in self.validation_outputs]).mean()

    #     print(f"Validation: Avg Loss: {avg_loss}, Avg Similarity Loss: {avg_similarity_loss}, Avg Time Loss: {avg_time_loss}")

    #     self.validation_outputs.clear()
        
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

def DBgolf():
    print("Starting training process...")
    print("Loading video files...")
    video_dir = Path('../data/GolfDB')
    if video_dir.exists() and video_dir.is_dir():
        video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.MP4'))
        if not video_files:
            print("No video files found with .mp4 extension.")
        else:
            print(f"Found {len(video_files)} video files.")
    else:
        print("Directory does not exist or is not a valid directory.")
    video_files = [file for file in video_files if int(str(file).split('/')[-1].split('.')[0]) % 2 == (0 if not args.use_120fps else 1)]
    video_files.sort(key=lambda x : int(str(x).split('/')[-1].split('.')[0]))
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
    return train_dataset, val_dataset

def PennAction(use_golf_folders=True, data_size=0, dont_split=False, use_batch_minlen=False):
    print("Starting training process...")
    print("Loading video files...")
    # frame_dir = Path('./data/Penn_Action/Penn_Action/frames')
    frame_dir = Path('../data/PennAction')
    
    if frame_dir.exists() and frame_dir.is_dir():
        dataset = FrameSequenceDataset(frame_dir, use_golf_folders=use_golf_folders, data_size=data_size)

        print(f"Total sequences in dataset: {len(dataset)}")
        print(f"Shortest sequence length: {dataset.shortest_sequence_length}")

        # Reduce the dataset to the first data_size sequences
        max_sequences = len(dataset)
        dataset = torch.utils.data.Subset(dataset, range(min(max_sequences, len(dataset))))
        print(f"Reduced dataset size: {len(dataset)}")

        if dont_split:
            return dataset
        else:
            # Split dataset into train and validation sets
            dataset_size = len(dataset)
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            print(f"Dataset split: {train_size} training sequences, {val_size} validation sequences.")
            return train_dataset, val_dataset
    else:
        print("Directory does not exist or is not a valid directory.")
        return None, None
    
def count_model_parameters(model, model_name=""):
    """
    Count trainable and frozen parameters of a model.

    Args:
        model (nn.Module): The PyTorch model to count parameters.
        model_name (str): Name of the model (for display).

    Returns:
        dict: Dictionary with trainable and frozen parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"{model_name} Parameters:")
    print(f"  Total parameters: {total_params}")
    print(f"  Trainable parameters: {trainable_params}")
    print(f"  Frozen parameters: {frozen_params}")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }

def print_parameter_details(model, model_name="Model"):
    """
    Print details of each parameter in the model.

    Args:
        model (nn.Module): PyTorch model.
        model_name (str): Name of the model.
    """
    print(f"{model_name} Parameter Details:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.numel()} parameters, requires_grad={param.requires_grad}")
    
def train(args):
    print(f"Using dataset: {args.dataset}")
    if args.dataset == 'GolfDB':
        train_dataset, val_dataset = DBgolf()
    elif args.dataset == 'PennAction':
        train_dataset, val_dataset = PennAction(args.use_golf_folders, args.data_size, args.use_batch_minlen)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
        
    print("Creating data loaders...")
    num_workers = 15 if args.mac else 21
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True
    )
    
    print("Initializing model...")
    model = LitModel(loss_type=args.loss_type, similarity_type=args.similarity_type, temperature=args.temperature, variance_lambda=args.variance_lambda, use_random_window=args.use_random_window, use_align_alpha=args.use_align_alpha, align_alpha_strength=args.align_alpha_strength, do_not_reduce_frame_rate=args.do_not_reduce_frame_rate, small_embedder=args.small_embedder, dont_stack=args.dont_stack)

    base_model_stats = count_model_parameters(model.model.cnn, "BaseModel")
    conv_embedder_stats = count_model_parameters(model.model.emb, "ConvEmbedder")
    # print_parameter_details(model.model.cnn, "BaseModel")
    print_parameter_details(model.model.emb, "ConvEmbedder")
    print('Model initialized!')
    print(model)

    print("Setting up training callbacks and logger...")
    filename = 'model-{epoch:02d}-{val_loss:.2f}' if args.validate else 'model-{epoch:02d}-{train_loss:.2f}'
    filename += f"{args.loss_type}_{args.similarity_type}_temp_{args.temperature}_var_{args.variance_lambda}_random_window_{args.use_random_window}_align_alpha_{args.use_align_alpha}_align_alpha_strength_{args.align_alpha_strength}_do_not_reduce_frame_rate_{args.do_not_reduce_frame_rate}"
    if args.validate:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_total_loss',
            dirpath='checkpoints',
            filename=filename,
            save_top_k=3,
            mode='min',
            save_last=True
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor='train_total_loss',
            dirpath='checkpoints',
            filename=filename,
            save_top_k=3,
            mode='min',
            save_last=True
        )
    
    logger = TensorBoardLogger("lightning_logs", name="my_model")

    hparams = vars(args)
    logger.log_hyperparams(hparams)
    
    print("Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=500,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator='gpu'
    )
    
    print("Starting model training...")
    if args.validate:
        trainer.fit(model, train_loader, val_loader)
    else:
        # trainer.fit(model, train_loader, train_loader)
        trainer.fit(model, train_loader)
    print("Training complete!")

    # Print final training loss
    final_training_loss = trainer.callback_metrics.get('train_total_loss')
    print(f"Final Training Loss: {final_training_loss}")

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Train a video alignment model.")
        parser.add_argument('--loss_type', type=str, default='regression_mse_var', help='Type of loss function to use')
        parser.add_argument('--similarity_type', type=str, default='cosine', help='Type of similarity function to use')
        parser.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter for contrastive loss')
        parser.add_argument('--variance_lambda', type=float, default=0.001, help='Lambda parameter for variance loss')
        parser.add_argument('--use_random_window', action='store_true', help='Whether to use random window cropping')
        parser.add_argument('--use_align_alpha', action='store_true', help='Whether to use alignment alpha')
        parser.add_argument('--align_alpha_strength', type=float, default=0.1, help='Strength of alignment alpha')
        parser.add_argument('--do_not_reduce_frame_rate', action='store_true', help='Whether to reduce frame rate to 10fps')
        parser.add_argument('--use_120fps', action='store_true', help='Whether to use 120fps videos')
        parser.add_argument('--dataset', type=str, default='PennAction', choices=['GolfDB', 'PennAction'], help='Dataset to use for training')
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('--use_golf_folders', action='store_true', help='Whether to use only golf folders in PennAction dataset')
        parser.add_argument('--data_size', type=int, default=0, help='Number of sequences to use from PennAction dataset, 0 for all')
        parser.add_argument('--dont_stack', action='store_true', help='Whether to stack temperal features')        # do not downsample
        parser.add_argument('--mac', action='store_true', help='Whether to use Mac to train')
        parser.add_argument('--precise_tensor', action='store_true', help='Whether to use high precision tensorfloat')
        parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
        parser.add_argument('--validate', action='store_true', help='Whether to validate')
        parser.add_argument('--small_embedder', action='store_true', help='Whether to use a smaller embedder')           # reduce channels from 512 to 256
        parser.add_argument('--use_batch_minlen', action='store_true', help='Whether to use the shortest video length within a batch, rather than the whole dataset')         # To keep the frames as many as possible
        return parser.parse_args()

    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.mac:
        torch.set_float32_matmul_precision('high' if args.precise_tensor else 'medium')
    print("--------------------------------- Starting Training ---------------------------------")
    train(args)
    print("--------------------------------- Training Complete ---------------------------------")

    # align_alpha_strengths = [0.1, 0.3, 1.0, 3.0, 10.0]
    # for align_alpha_strength in align_alpha_strengths:
    #     print(f"Training with align_alpha_strength={align_alpha_strength}")
    #     print("------------------------------------------------------------------------------------")
    #     args.align_alpha_strength = align_alpha_strength
    #     train(args)
    #     print("------------------------------------------------------------------------------------")