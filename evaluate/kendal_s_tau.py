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

r"""Evaluation train and val loss using the algo.
"""

from typing import List, Dict, Union
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
import logging

def compute_kendalls_tau(
    embs_list: List[torch.Tensor], 
    stride: int, 
    step: int, 
    split: str,
    distance_metric: str = 'euclidean'
) -> float:
    """Compute Kendall's Tau correlation between sequence embeddings.
    
    Args:
        embs_list: List of embedding tensors for different sequences
        stride: Sampling stride to reduce computation
        step: Current training step
        split: Dataset split name ('train' or 'val')
        distance_metric: Distance metric for comparing embeddings
    
    Returns:
        Average Kendall's Tau correlation coefficient
    """
    num_seqs = len(embs_list)
    taus = np.zeros((num_seqs * (num_seqs - 1)))
    idx = 0
    
    # Convert torch tensors to numpy if needed
    embs_list = [e.cpu().numpy() if torch.is_tensor(e) else e for e in embs_list]
    
    for i in range(num_seqs):
        query_feats = embs_list[i][::stride]
        for j in range(num_seqs):
            if i == j:
                continue
            candidate_feats = embs_list[j][::stride]
            dists = cdist(query_feats, candidate_feats, distance_metric)
            
            nns = np.argmin(dists, axis=1)
            taus[idx] = kendalltau(np.arange(len(nns)), nns).correlation
            idx += 1
    
    # Remove NaNs and compute mean
    taus = taus[~np.isnan(taus)]
    tau = float(np.mean(taus))

    logging.info(f'Step [{step}] {split} set alignment tau: {tau:.4f}')
    return tau


class KendallsTau:
    """Evaluates sequence alignments using Kendall's Tau correlation."""

    def evaluate_embeddings(
        self,
        datasets: Dict[str, Dict[str, List[torch.Tensor]]],
        step: int,
        stride: int = 1
    ) -> float:
        """Evaluate embeddings on train and validation sets.
        
        Args:
            datasets: Dictionary containing train and validation embeddings
            step: Current training step
            stride: Sampling stride for computation efficiency
        
        Returns:
            Kendall's Tau correlation on validation set
        """
        # Evaluate train set
        train_embs = datasets['train_dataset']['embs']
        compute_kendalls_tau(
            train_embs,
            stride=stride,
            step=step,
            split=f"{datasets['name']}_train"
        )

        # Evaluate validation set
        val_embs = datasets['val_dataset']['embs']
        val_tau = compute_kendalls_tau(
            val_embs,
            stride=stride, 
            step=step,
            split=f"{datasets['name']}_val"
        )
        
        return val_tau