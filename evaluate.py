from train import LitModel, VideoDataset, DataLoader, video_duration
from losses import compute_alignment_loss, get_scaled_similarity
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from model import ModelWrapper
import os, re
from tqdm import tqdm
from torchvision import transforms

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = batch.cuda()
            emb, cnn_feats = model(batch)
            loss = compute_alignment_loss(emb,
                                          cnn_feats,
                                          batch_size=batch.shape[0], 
                                          loss_type='classification', 
                                          similarity_type='cosine')
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    video_dir = Path('../videos_160')
    video_files = list(video_dir.glob('*.mp4'))
    video_files = [file for file in video_files if video_duration(file) > 100 and video_duration(file) < 200]
    video_files.sort(key=lambda x : int(str(x).split('/')[-1].split('.')[0]))
   
    model = ModelWrapper()  # Initialize your model
    print(f"Number of trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
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
    model.cuda()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(video_files[:2])
    dataset = VideoDataset(video_files[:2])
    dataset.transform = preprocess

    val_loader = DataLoader(dataset, batch_size=2, drop_last=True)  # Get dataloader
    loss = evaluate(model, val_loader)
    print(f"Validation Loss: {loss}")

    # Assuming you have a function for matching and plotting
    for i, batch in enumerate(val_loader):  # For 3 video pairs
        # Perform matching and plot logits
        with torch.no_grad():
            feats, cnn_feats = model(batch.cuda())  # Get logits for video pair
        # Convert features to CPU if needed
        features1 = feats[0].cpu()
        features2 = feats[1].cpu()

        output_path = f"{i}"

        # Compute similarities between embs1 and embs2
        dist = get_scaled_similarity(features1, features2, 'cosine', 0.1)       # [N1, N2]

        softmaxed_sim_12 = torch.nn.functional.softmax(dist, dim=1)         # alpha

        # Compute soft-nearest neighbors
        nn_embs = torch.mm(softmaxed_sim_12, features2)          # [N1, D], tilda v_i

        # Compute similarities between nn_embs and embs1
        sim_21 = get_scaled_similarity(nn_embs, features1, 'cosine', 0.1)        # [N1, N1], beta before softmax
        sim_21 = torch.nn.functional.softmax(sim_21, dim=1)

        # Make a heat map of dist
        plt.figure(figsize=(10, 10))
        sns.heatmap(sim_21, cmap='coolwarm')
        plt.title('Similarity Heatmap')
        plt.savefig(output_path + '_0.png')

        # Make a heat map of dist
        plt.figure(figsize=(10, 10))
        sns.heatmap(softmaxed_sim_12, cmap='coolwarm')
        plt.title('Similarity Heatmap')
        plt.savefig(output_path + '_1.png')

        if i == 3:
            break

if __name__ == "__main__":
    main()

