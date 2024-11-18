# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic alignment between all pairs of sequences in a batch."""

from typing import Tuple, Union
import torch
import torch.nn.functional as F

from losses import classification_loss, regression_loss


def pairwise_l2_distance(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
    """Computes pairwise distances between all rows of embs1 and embs2."""
    norm1 = torch.sum(torch.square(embs1), dim=1).view(-1, 1)
    norm2 = torch.sum(torch.square(embs2), dim=1).view(1, -1)
    
    # Use broadcasting for efficient computation
    dist = norm1 + norm2 - 2.0 * torch.matmul(embs1, embs2.t())
    return torch.clamp(dist, min=0.0)  # Ensure non-negative distances


def get_scaled_similarity(
    embs1: torch.Tensor,
    embs2: torch.Tensor,
    similarity_type: str,
    temperature: float
) -> torch.Tensor:
    """Returns scaled similarity between all rows of embs1 and embs2.
    
    Args:
        embs1: Embeddings of shape [M, D] where M is number of embeddings
        embs2: Embeddings of shape [N, D] where N is number of embeddings
        similarity_type: Either 'l2' or 'cosine'
        temperature: Temperature for scaling logits
    
    Returns:
        Similarity tensor of shape [M, N]
    """
    channels = embs1.size(1)

    if similarity_type == 'cosine':
        similarity = torch.matmul(embs1, embs2.t())
    elif similarity_type == 'l2':
        similarity = -1.0 * pairwise_l2_distance(embs1, embs2)
    else:
        raise ValueError('similarity_type must be either l2 or cosine')

    # Scale by number of channels and temperature
    similarity = similarity / channels / temperature
    return similarity


def align_pair_of_sequences(
    embs1: torch.Tensor,
    embs2: torch.Tensor, 
    similarity_type: str,
    temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align a pair of embedding sequences.
    
    Args:
        embs1: Embeddings of shape [M, D]
        embs2: Embeddings of shape [N, D]
        similarity_type: Either 'l2' or 'cosine'
        temperature: Temperature for scaling logits
    
    Returns:
        logits: Pre-softmax similarity scores
        labels: One-hot ground truth labels
    """
    max_num_steps = embs1.size(0)

    # Compute similarities and softmax
    sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
    softmaxed_sim_12 = F.softmax(sim_12, dim=1)

    # Calculate soft-nearest neighbors
    nn_embs = torch.matmul(softmaxed_sim_12, embs2)

    # Find distances between nn_embs and embs1
    sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)

    # Create one-hot labels
    labels = F.one_hot(torch.arange(max_num_steps, device=embs1.device), 
                      max_num_steps).float()
    
    return sim_21, labels


def compute_deterministic_alignment_loss(
    embs: torch.Tensor,
    steps: torch.Tensor,
    seq_lens: torch.Tensor,
    num_steps: int,
    batch_size: int,
    loss_type: str,
    similarity_type: str,
    temperature: float,
    label_smoothing: float,
    variance_lambda: float,
    huber_delta: float,
    normalize_indices: bool
) -> torch.Tensor:
    """Compute cycle-consistency loss for all steps in each sequence.
    
    Args:
        embs: Sequential embeddings of shape [N, T, D]
        steps: Step indices of shape [N, T]
        seq_lens: Sequence lengths
        num_steps: Number of timesteps
        batch_size: Batch size
        loss_type: Type of loss function to use
        similarity_type: Type of similarity metric
        temperature: Temperature for scaling
        label_smoothing: Label smoothing factor
        variance_lambda: Weight of variance in loss
        huber_delta: Delta parameter for Huber loss
        normalize_indices: Whether to normalize indices
    
    Returns:
        Scalar loss tensor
    """
    labels_list = []
    logits_list = []
    steps_list = []
    seq_lens_list = []

    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                logits, labels = align_pair_of_sequences(
                    embs[i], embs[j], similarity_type, temperature)
                
                logits_list.append(logits)
                labels_list.append(labels)
                steps_list.append(steps[i:i+1].repeat(num_steps, 1))
                seq_lens_list.append(seq_lens[i:i+1].repeat(num_steps))

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    steps = torch.cat(steps_list, dim=0)
    seq_lens = torch.cat(seq_lens_list, dim=0)

    if loss_type == 'classification':
        loss = classification_loss(logits, labels, label_smoothing)
    elif 'regression' in loss_type:
        loss = regression_loss(
            logits, labels, num_steps, steps, seq_lens,
            loss_type, normalize_indices, variance_lambda, huber_delta)
    else:
        raise ValueError(
            f'Unsupported loss_type {loss_type}. '
            'Use: regression_mse, regression_huber, or classification')

    return loss