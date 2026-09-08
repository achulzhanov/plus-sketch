#!/usr/bin/env python3
"""
sketchtask.py -- dataset loader for plus-sketch.

Replaces llama2.c's tinystories.Task. Same window logic, pointed at
../data/tok{vocab_size}/ where sketchdata.py writes its shards.

Because every drawing is padded to exactly max_seq_len and windows start at
multiples of max_seq_len, each training example is exactly one drawing. The
window is max_seq_len+1 tokens so x and y are both max_seq_len -- the final
target is the BOS of the next drawing, which just teaches "after padding, a
new sketch begins."

shard00 is the validation split, everything else is train.
"""

import glob
import os
import random

import numpy as np
import torch
import torch.distributed as dist

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


class PretokDataset(torch.utils.data.IterableDataset):
    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)

        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
        shards = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        assert shards, f"no .bin shards in {bin_dir} -- run sketchdata.py tokenize"
        shards = shards[1:] if self.split == "train" else shards[:1]
        assert shards, "need at least 2 shards so there is a val split"

        while True:
            rng.shuffle(shards)
            for shard in shards:
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                n = len(m) // self.max_seq_len - 1
                assert n > 0, f"{shard} is too small for max_seq_len={self.max_seq_len}"
                ixs = list(range(n))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    chunk = torch.from_numpy(
                        m[start: start + self.max_seq_len + 1].astype(np.int64))
                    yield chunk[:-1], chunk[1:]


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **kwargs):
        ds = PretokDataset(**kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
        for x, y in dl:
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)
