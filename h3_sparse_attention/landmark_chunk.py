"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


from collections import defaultdict


import torch


def chunk_frame_lengths(frames, chunk):
    if type(frames) is not int or type(chunk) is not int or min(frames, chunk) < 1:
        raise ValueError('frames and chunk must be positive integers')
    count = max(1, (2 * frames + chunk) // (2 * chunk))
    return (chunk,) * (count - 1) + (frames - chunk * (count - 1),)


def chunk_permutation(samples, *, chunk_frames, grid_shape, fitting_samples=None, **kwargs):
    from .landmark_tree_v2 import _recursive_landmark_tree_v2
    frames, height, width = grid_shape
    lengths = chunk_frame_lengths(frames, chunk_frames)
    tokens = samples.shape[-2]
    if frames * height * width != tokens:
        raise ValueError('latent grid must match samples')
    if kwargs.get('group_size', 1) != 1:
        raise ValueError('chunk splitting currently requires group_size=1')
    if tokens % 64:
        raise ValueError('whole video must contain complete 64-token blocks')
    groups = defaultdict(list)
    start = 0
    for index, length in enumerate(lengths):
        groups[length].append((index, start));start += length * height * width
    batch, _, dim = samples.shape
    completed, tails, stats = {}, {}, []
    for length, members in groups.items():
        n = length * height * width
        if n < 64:
            raise ValueError('each latent chunk must contain at least 64 tokens')
        source = torch.cat([samples[:, offset:offset+n] for _, offset in members], dim=0)
        fitting = None if fitting_samples is None else torch.cat([fitting_samples[:, offset:offset+n] for _, offset in members], dim=0)
        result = _recursive_landmark_tree_v2(source, grid_shape=(length, height, width),
            initial_indices=torch.arange(n, device=samples.device), fitting_samples=fitting,
            **kwargs)
        permutations = result.permutation.reshape(len(members), batch, n)
        for i, (index, offset) in enumerate(members):
            perm = permutations[i] + offset
            completed[index] = perm[:, :result.active_tokens]
            tails[index] = perm[:, result.active_tokens:]
        stats.extend(result.split_stats)
    permutation = torch.cat([completed[i] for i in range(len(lengths))] + [tails[i] for i in range(len(lengths))], dim=1)
    inverse = torch.empty_like(permutation)
    inverse.scatter_(1, permutation, torch.arange(tokens, device=samples.device).expand(batch, -1))
    metadata = dict(mode='chunk',chunk_frames=chunk_frames,latent_frame_lengths=lengths,
        tail_tokens=sum((length*height*width)%64 for length in lengths),
        tail_policy='complete leaves per chunk; excluded tails packed at video end without recursive splitting',
        token_count=tokens)
    return permutation, inverse, tuple(stats), metadata

