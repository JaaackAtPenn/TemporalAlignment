import torch
import torch.nn as nn
import torch.nn.functional as F

def classification_loss(logits, labels, label_smoothing):
    # Detach labels to stop gradients, as we are generating labels
    labels = labels.detach()

    # Convert one-hot labels to class indices
    targets = torch.argmax(labels, dim=1)

    # Use CrossEntropyLoss with label smoothing
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss = loss_fn(logits, targets)

    return loss, 0.0

def regression_loss(softmaxed_logits, labels, alpha, num_steps, steps, seq_lens, steps_j, seq_lens_j,
                    loss_type, normalize_indices, variance_lambda, huber_delta, use_random_window, window_size, use_align_alpha, align_alpha_strength):
    # Detach labels and steps to stop gradients
    labels = labels.detach()
    steps = steps.detach()            # [N(N-1)*T, T]
    steps_j = steps_j.detach()

    # Normalize indices if required
    if normalize_indices:
        float_seq_lens = seq_lens.float()
        tile_seq_lens = float_seq_lens.unsqueeze(1).repeat(1, num_steps)
        steps = steps.float() / tile_seq_lens
    else:
        steps = steps.float()

    # Compute softmax over logits to get probabilities (beta)
    # beta = F.softmax(logits, dim=1)       # [N(N-1)*T, T]
    beta = softmaxed_logits

    # Compute true_time and pred_time, in terms of time sequence, not the sampled steps
    true_time = torch.sum(steps * labels, dim=1)      # [N(N-1)*T]
    pred_time = torch.sum(steps * beta, dim=1)       # [N(N-1)*T], mu, for every sampled time point in the batch

    # Compute tilda v
    if use_align_alpha:
        tilda_v = torch.sum(steps_j * alpha, dim=1)       # [N(N-1)*T], tilda v_i
        tilda_v_reshape = tilda_v.reshape(-1, alpha.shape[1])       # [N(N-1), T]
        d_tilda_v = tilda_v_reshape[:, 1:] - tilda_v_reshape[:, :-1]       # [N(N-1), T-1]
        align_alpha_loss = torch.mean(torch.relu(-d_tilda_v) ** 2) * align_alpha_strength        
    else:
        align_alpha_loss = 0.0

    if loss_type in ['regression_mse', 'regression_mse_var']:
        if 'var' in loss_type:
            # Variance-aware regression
            pred_time_tiled = pred_time.unsqueeze(1).repeat(1, num_steps)           # [N(N-1)*T, T]
            pred_time_variance = torch.sum((steps - pred_time_tiled) ** 2 * beta, dim=1)

            # Use log of variance for numerical stability
            pred_time_log_var = torch.log(pred_time_variance + 1e-8)  # Add epsilon to avoid log(0), not in the original TCC code
            squared_error = (true_time - pred_time) ** 2

            loss = torch.mean(torch.exp(-pred_time_log_var) * squared_error + variance_lambda * pred_time_log_var)
            return loss, align_alpha_loss
        else:
            # Standard MSE loss, without log variance normalization
            loss = F.mse_loss(pred_time, true_time)
            return loss, align_alpha_loss
    elif loss_type == 'regression_huber':
        # Huber loss (Smooth L1 loss in PyTorch)
        loss = F.smooth_l1_loss(pred_time, true_time, beta=huber_delta)
        return loss, align_alpha_loss
    else:
        raise ValueError(f"Unsupported regression loss '{loss_type}'. Supported losses are: "
                         "regression_mse, regression_mse_var, regression_huber.")

def pairwise_l2_distance(embs1, embs2):
    """Computes pairwise distances between all rows of embs1 and embs2."""
    # norm1 = torch.sum(embs1 ** 2, dim=1).unsqueeze(0)  # [N1, 1]
    # norm2 = torch.sum(embs2 ** 2, dim=1).unsqueeze(1)  # [1, N2]
    # dist = torch.clamp(norm1 + norm2 - 2.0 * torch.mm(embs1, embs2.t()), min=0.0)
    diff = embs1.unsqueeze(1) - embs2.unsqueeze(0)
    dist = torch.sum(diff ** 2, dim=2)
    return dist      # [N1, N2], every element is the squared L2 distance between a pair of embeddings

def sample_center(projected_i, window_size, N2):
    start = max(0, projected_i - window_size // 2)
    end = min(N2, projected_i + window_size // 2 + 1)
    window_range = torch.arange(start, end)
    mean = projected_i
    std = window_size / 4  
    probabilities = torch.exp(-0.5 * ((window_range - mean) / std) ** 2)
    probabilities /= probabilities.sum()  
    sampled_index = torch.multinomial(probabilities, num_samples=1).item()
    return window_range[sampled_index]

def get_scaled_similarity(embs1, embs2, similarity_type, temperature, use_random_window=False, use_center_window=False, window_size=0):       # return logits, need to be softmaxed
    ''' NOTE: In order to apply window, it's better to return softmaxed similarity, not logits. '''

    channels = embs1.shape[1]       # featrue dimension
    N1 = embs1.shape[0]
    N2 = embs2.shape[0]
    
    # if use_random_window or use_center_window:
    #     selected_indices = torch.zeros((N1, window_size), device=embs1.device)
    #     for i in range(N1):
    #         projected_i = int(i * N2 / N1)
    #         if use_random_window:
    #             center = torch.randint(low=projected_i - window_size // 2, high=projected_i + window_size // 2 + 1, size=(1,))
    #         elif use_center_window:
    #             center = projected_i
    #         start_index = max(0, center - window_size // 2)
    #         end_index = min(N2, center + window_size // 2)
    #         if end_index - start_index < window_size:
    #             if start_index == 0:
    #                 end_index = window_size
    #             elif end_index == N2:
    #                 start_index = N2 - window_size
    #         selected_indices[i] = torch.arange(start_index, end_index, device=embs1.device)

    #     windowed_embs2 = torch.tensor([embs2[indices] for indices in selected_indices], device=embs1.device)       # [N1, window_size, D]

    #     if similarity_type == 'cosine':
    #         similarity = torch.bmm(embs1.unsqueeze(1), windowed_embs2.permute(0, 2, 1)).squeeze(1)        # [N1, window_size]
    #     elif similarity_type == 'l2':
    #         diff = embs1.unsqueeze(1) - windowed_embs2
    #         similarity = -torch.sum(diff ** 2, dim=2)  # [N1, window_size]

    #     similarity = similarity / channels
    #     similarity = similarity / temperature

    #     return similarity, selected_indices, windowed_embs2         # similarity: [N1, window_size]; selected_indices: [N1, window_size]; windowed_embs2: [N1, window_size, D]

    if similarity_type == 'cosine':
        similarity = torch.mm(embs1, embs2.t())        # just like transformer attention
    elif similarity_type == 'l2':
        similarity = -pairwise_l2_distance(embs1, embs2)
    else:
        raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale by the number of channels
    similarity = similarity / channels
    # Scale by temperature
    similarity = similarity / temperature

    similarity = F.softmax(similarity, dim=1)         # [N1, N2]

    if use_random_window or use_center_window:
        selected_indices_mask = torch.zeros((N1, N2), device=embs1.device)
        for i in range(N1):
            projected_i = int(i * N2 / N1)
            if use_random_window:
                # center = torch.randint(low=projected_i - window_size // 2, high=projected_i + window_size // 2 + 1, size=(1,))
                center = sample_center(projected_i, window_size, N2)
            elif use_center_window:
                center = projected_i
            start_index = max(0, center - window_size // 2)
            end_index = min(N2, center + window_size // 2)
            if end_index - start_index < window_size:
                if start_index == 0:
                    end_index = window_size
                elif end_index == N2:
                    start_index = N2 - window_size
            selected_indices_mask[i, start_index:end_index] = 1

        similarity = similarity * selected_indices_mask
        similarity = similarity / torch.sum(similarity, dim=1, keepdim=True)       # normalize the selected similarity

    return similarity       # [N1, N2], every element is the similarity between a pair of embeddings

def align_pair_of_sequences(embs1, embs2, similarity_type, temperature, use_random_window=False, use_center_window=False, window_size=0):     # embs1 is U, embs2 is V
    max_num_steps = embs1.shape[0]        # embs1 and embs2 have the same number of steps, N1 = N2

    # if use_random_window or use_center_window:
    #     sim12, _, windowed_embds2 = get_scaled_similarity(embs1, embs2, similarity_type, temperature, use_random_window, use_center_window, window_size)       # sim12:[N1, window_size], selected_indices: [N1, window_size]; windowed_embds2: [N1, window_size, D]
    #     softmaxed_sim12 = F.softmax(sim12, dim=1)         # alpha
    #     alpha = softmaxed_sim12         # [N1, window_size]

    #     # Compute soft-nearest neighbors
    #     nn_embds = torch.bmm(softmaxed_sim12.unsqueeze(1), windowed_embds2).squeeze(1)          # [N1, D], tilda v_i

    #     # Compute similarities between nn_embds and embs1
    #     sim21, selected_indices, _ = get_scaled_similarity(nn_embds, embs1, similarity_type, temperature)        # sim21: [N1, N1], selected_indices: [N1, N1]; windowed_embds1: [N1, N1, D]

    #     logits = sim21
    #     labels = F.one_hot(torch.arange(max_num_steps), num_classes=max_num_steps).float().to(embs1.device)       # [N1, N1], identity matrix

    #     return logits, labels, alpha, selected_indices       # logits before softmax, labels are one-hot

    # Compute similarities between embs1 and embs2
    softmaxed_sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature, use_random_window, use_center_window, window_size)       # [N1, N2]
    # softmaxed_sim_12 = F.softmax(sim_12, dim=1)         # alpha
    alpha = softmaxed_sim_12         # [N1, N2]

    # Compute soft-nearest neighbors
    nn_embs = torch.mm(softmaxed_sim_12, embs2)          # [N1, D], tilda v_i

    # Compute similarities between nn_embs and embs1
    softmaxed_sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature, use_random_window, use_center_window, window_size)        # [N1, N1], beta before softmax

    # logits = sim_21
    labels = F.one_hot(torch.arange(max_num_steps), num_classes=max_num_steps).float().to(embs1.device)       # [N1, N1], identity matrix

    return softmaxed_sim_21, labels, alpha       # logits before softmax, labels are one-hot

def compute_deterministic_alignment_loss(embs,
                                         steps,
                                         seq_lens,
                                         num_steps,
                                         batch_size,
                                         loss_type,
                                         similarity_type,
                                         temperature,
                                         label_smoothing,
                                         variance_lambda,
                                         huber_delta,
                                         normalize_indices,
                                         use_random_window,
                                         use_center_window,
                                         window_size,
                                         use_align_alpha,
                                         align_alpha_strength):
    
    labels_list = []
    softmaxed_logits_list = []
    steps_list = []        # steps of u
    seq_lens_list = []       # sequence lengths of u
    steps_j_list = []       # steps of v
    seq_lens_j_list = []       # sequence lengths of v
    alpha_list = []
    # selected_indices_list = []

    # if use_random_window or use_center_window:
    #     for i in range(batch_size):
    #         for j in range(batch_size):
    #             if i != j:
    #                 softmaxed_logits, labels, alpha, selected_indices = align_pair_of_sequences(
    #                     embs[i],
    #                     embs[j],
    #                     similarity_type,
    #                     temperature,
    #                     use_random_window,
    #                     use_center_window,
    #                     window_size
    #                 )
    #                 softmaxed_logits_list.append(softmaxed_logits)
    #                 labels_list.append(labels)
    #                 alpha_list.append(alpha)
    #                 selected_indices
    # else:
    for i in range(batch_size):
        for j in range(batch_size):
            # Do not align the sequence with itself
            if i != j:
                softmaxed_logits, labels, alpha = align_pair_of_sequences(          # logits is beta before softmax
                    embs[i],
                    embs[j],
                    similarity_type,
                    temperature,
                    use_random_window,
                    use_center_window,
                    window_size
                )
                softmaxed_logits_list.append(softmaxed_logits)       # [T, T]
                labels_list.append(labels)       # [T, T]
                alpha_list.append(alpha)        # [T, T]
                steps_i = steps[i].unsqueeze(0).repeat(num_steps, 1)        # [T, T], every row is the same, representing the step indices of ui
                steps_list.append(steps_i)
                seq_lens_i = seq_lens[i].unsqueeze(0).repeat(num_steps)        # [T], every element is the same, representing the sequence length of ui
                seq_lens_list.append(seq_lens_i)
                steps_j = steps[j].unsqueeze(0).repeat(num_steps, 1)        # [T, T], every row is the same, representing the step indices of vj
                steps_j_list.append(steps_j)
                seq_lens_j = seq_lens[j].unsqueeze(0).repeat(num_steps)        # [T], every element is the same, representing the sequence length of vj
                seq_lens_j_list.append(seq_lens_j)

        softmaxed_logits = torch.cat(softmaxed_logits_list, dim=0)          # [N(N-1)*T, T], N is the batch size
        labels = torch.cat(labels_list, dim=0)          # [N(N-1)*T, T]
        alpha = torch.cat(alpha_list, dim=0)          # [N(N-1)*T, T]
        steps = torch.cat(steps_list, dim=0)           # [N(N-1)*T, T]
        seq_lens = torch.cat(seq_lens_list, dim=0)      # [N(N-1)*T]
        steps_j = torch.cat(steps_j_list, dim=0)           # [N(N-1)*T, T]
        seq_lens_j = torch.cat(seq_lens_j_list, dim=0)      # [N(N-1)*T]

        if loss_type == 'classification':
            loss, align_alpha_loss = classification_loss(softmaxed_logits, labels, label_smoothing)
        elif 'regression' in loss_type:
            loss, align_alpha_loss = regression_loss(
                softmaxed_logits, labels, alpha, num_steps, steps, seq_lens, steps_j, seq_lens_j,
                loss_type, normalize_indices, variance_lambda, huber_delta, use_random_window, window_size, use_align_alpha, align_alpha_strength
            )
        else:
            raise ValueError(f"Unidentified loss_type {loss_type}. Currently supported loss "
                            "types are: regression_mse, regression_huber, classification.")

    return loss, align_alpha_loss

def _align_single_cycle(cycle, embs, cycle_length, num_steps,
                        similarity_type, temperature):
    """Takes a single cycle and returns logits (similarity scores) and labels."""
    device = embs.device

    # Choose a random frame index.
    n_idx = torch.randint(low=0, high=num_steps, size=(1,), device=device)         # a random frame index
    # Create one-hot labels.
    onehot_labels = F.one_hot(n_idx, num_classes=num_steps).float().squeeze(0)       # [t], a random one-hot label

    # Select query features for the first frame in the cycle.
    query_feats = embs[cycle[0], n_idx, :].squeeze(0)  # [D], u_k, k is random

    num_channels = query_feats.shape[-1]
    for c in range(1, cycle_length + 1):
        candidate_feats = embs[cycle[c]]  # [T, D]

        if similarity_type == 'l2':
            # Compute L2 distance.
            query_feats_expanded = query_feats.unsqueeze(0).repeat(num_steps, 1)    # [T, D]
            mean_squared_distance = torch.sum((query_feats_expanded - candidate_feats) ** 2, dim=1)     # [T], squared L2 distance between u_k and every frame in the middle sequence(v) 
            # Convert L2 distance to similarity.
            similarity = -mean_squared_distance
        elif similarity_type == 'cosine':
            # Compute cosine similarity (dot product).
            similarity = candidate_feats @ query_feats  # [T]
        else:
            raise ValueError("similarity_type can either be 'l2' or 'cosine'.")

        # Normalize similarity by the number of channels and temperature.
        similarity = similarity / num_channels
        similarity = similarity / temperature

        # Compute softmax over similarities.
        beta = F.softmax(similarity, dim=0).unsqueeze(1)  # [T, 1]

        # Compute weighted nearest neighbor.
        query_feats = torch.sum(beta * candidate_feats, dim=0)  # [D], v becomes u for the next step

    return similarity, onehot_labels         # similarity is between the last two sequences in a cycle, onehot_labels selects u_k(groundtruth) from the first sequence

def _align(cycles, embs, num_steps, num_cycles, cycle_length,
           similarity_type, temperature):
    """Align by finding cycles in embeddings."""
    logits_list = []
    labels_list = []
    for i in range(num_cycles):
        logits, labels = _align_single_cycle(
            cycles[i],
            embs,
            cycle_length,
            num_steps,
            similarity_type,
            temperature
        )
        logits_list.append(logits)
        labels_list.append(labels)

    logits = torch.stack(logits_list)  # [num_cycles, T], not [num_cycles * T, T]
    labels = torch.stack(labels_list)  # [num_cycles, T]

    return logits, labels

def gen_cycles(num_cycles, batch_size, cycle_length=2):        # if cycle_length=2, then it is a pair of sequences; if cycle_length=3, cycle could be [1, 2, 3, 1]
    # Create a sorted list of indices.
    sorted_idxes = torch.arange(batch_size).unsqueeze(1).repeat(1, num_cycles)
    sorted_idxes = sorted_idxes.T.contiguous().view(-1)  # [num_cycles * batch_size]

    # Shuffle indices.
    shuffled_idxes = sorted_idxes[torch.randperm(sorted_idxes.size(0))].view(num_cycles, batch_size)

    # Select the first 'cycle_length' indices and append the starting index to form a cycle.
    cycles = shuffled_idxes[:, :cycle_length] 
    cycles = torch.cat([cycles, cycles[:, 0:1]], dim=1)         # [num_cycles, cycle_length + 1]

    return cycles

def compute_stochastic_alignment_loss(embs,
                                      steps,
                                      seq_lens,
                                      num_steps,
                                      batch_size,
                                      loss_type,
                                      similarity_type,
                                      num_cycles,
                                      cycle_length,
                                      temperature,
                                      label_smoothing,
                                      variance_lambda,
                                      huber_delta,
                                      normalize_indices,
                                      use_random_window,
                                      use_align_alpha):
    device = embs.device

    # Generate cycles.
    cycles = gen_cycles(num_cycles, batch_size, cycle_length).to(device)

    # Align embeddings using the generated cycles.
    logits, labels = _align(
        cycles,
        embs,
        num_steps,
        num_cycles,
        cycle_length,
        similarity_type,
        temperature
    )

    if loss_type == 'classification':
        loss, align_alpha_loss = classification_loss(logits, labels, label_smoothing)
    elif 'regression' in loss_type:
        steps_selected = steps[cycles[:, 0]]     # [num_cycles, T], sampled steps of u in each cycle
        seq_lens_selected = seq_lens[cycles[:, 0]]       # [num_cycles], sequence lengths of u in each cycle
        loss, align_alpha_loss = regression_loss(
            logits, labels, num_steps, steps_selected, seq_lens_selected,
            loss_type, normalize_indices, variance_lambda, huber_delta, use_random_window, use_align_alpha
        )
    else:
        raise ValueError(f"Unidentified loss type {loss_type}. Currently supported loss "
                         "types are: regression_mse, regression_huber, classification.")

    return loss, align_alpha_loss

def compute_alignment_loss(embs,          # [B, T, D]
                           batch_size,          
                           steps=None,         # [B, T], because of sampling, steps are not necessarily consecutive
                           seq_lens=None,         # sequence lengths are not necessarily equal before sampling
                           stochastic_matching=False,
                           normalize_embeddings=False,         # steps are normalized by sequence lengths
                           loss_type='classification',
                           similarity_type='l2',
                           num_cycles=20,       # number of cycles(a cycle is a pair if cycle_length=2) for each batch. Note that at each cycle, only one frame is sampled
                           cycle_length=2,        # a cycle may contain 2 or more sequences
                           temperature=0.1,
                           label_smoothing=0.1,
                           variance_lambda=0.001,
                           huber_delta=0.1,        
                           normalize_indices=True,
                           use_random_window=False,
                           use_center_window=False,
                           window_size=5,
                           use_align_alpha=False,
                           align_alpha_strength=0.1):
    
    ##############################################################################
    # Checking inputs and setting defaults.
    ##############################################################################

    # Get the number of timesteps in the sequence embeddings.
    num_steps = embs.shape[1]

    # If steps has not been provided assume sampling has been done uniformly.
    if steps is None:
        steps = torch.arange(num_steps, device=embs.device).unsqueeze(0).repeat(batch_size, 1)

    # If seq_lens has not been provided assume is equal to the size of the
    # time axis in the embeddings.
    if seq_lens is None:
        seq_lens = torch.tensor([num_steps]*batch_size, device=embs.device)

    # Check if batch size embs is consistent with provided batch size.
    # print('num_steps:', num_steps)
    # print('steps shape:', steps.shape)
    assert batch_size == embs.shape[0], "Batch size does not match embs shape"
    assert num_steps == steps.shape[1], "num_steps does not match steps shape"
    assert batch_size == steps.shape[0], "Batch size does not match steps shape"

    ##############################################################################
    # Perform alignment and return loss.
    ##############################################################################

    if normalize_embeddings:
        embs = F.normalize(embs, p=2, dim=-1)

    if stochastic_matching:
        loss, align_alpha_loss = compute_stochastic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            num_steps=num_steps,
            batch_size=batch_size,
            loss_type=loss_type,
            similarity_type=similarity_type,
            num_cycles=num_cycles,
            cycle_length=cycle_length,
            temperature=temperature,
            label_smoothing=label_smoothing,
            variance_lambda=variance_lambda,
            huber_delta=huber_delta,
            normalize_indices=normalize_indices,
            use_random_window=use_random_window,
            use_center_window=use_center_window,
            window_size=window_size,
            use_align_alpha=use_align_alpha,
            align_alpha_strength=align_alpha_strength)
    else:
        loss, align_alpha_loss = compute_deterministic_alignment_loss(          # compute loss between all pairs of sequences
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            num_steps=num_steps,
            batch_size=batch_size,
            loss_type=loss_type,
            similarity_type=similarity_type,
            temperature=temperature,
            label_smoothing=label_smoothing,
            variance_lambda=variance_lambda,
            huber_delta=huber_delta,
            normalize_indices=normalize_indices,
            use_random_window=use_random_window,
            use_center_window=use_center_window,
            window_size=window_size,
            use_align_alpha=use_align_alpha,
            align_alpha_strength=align_alpha_strength)

    return loss, align_alpha_loss