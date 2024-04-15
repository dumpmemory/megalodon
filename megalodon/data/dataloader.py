from collections import defaultdict
from typing import List, Dict, Iterator

import torch

from .iterator import JSONLIterator
from .tokenizer import Tokenizer
from megalodon.utils import pad


class DataLoader:
    def __init__(
        self,
        tokenizer: Tokenizer,
        path: str,
        batch_size: int,
        world_rank: int,
        world_size: int,
    ):
        self.tokenizer = tokenizer
        self.world_rank = world_rank
        self.world_size = world_size
        self.path = path
        self.batch_size = batch_size
        self.jsonl_iterator = JSONLIterator(
            fpath=path,
            world_rank=world_rank,
            world_size=world_size
        )

    def __iter__(self) -> Iterator[Dict]:
        batch: Dict[str, List] = defaultdict(list)
        curr_bs = 0
        batch_counter = 0
        for example in self.jsonl_iterator:
            text = example['text']
            x = self.tokenizer.encode(text, bos=True, eos=True)
            batch['x'].append(x[:-1])
            batch['y'].append(x[1:])
            curr_bs += 1
            if curr_bs == self.batch_size:
                batch_counter += 1
                yield self.tensorize_batch(batch)
                batch = defaultdict(list)
                curr_bs = 0
        if curr_bs > 0:
            yield self.tensorize_batch(batch)

    def tensorize_batch(self, batch: Dict) -> Dict:
        key_padding = {"x": 0, "y": -100}
        for key, padding in key_padding.items():
            assert key in batch
            batch[key] = self.tensorize(batch[key], padding)
        return batch

    def tensorize(self, batch_tokens: List[List[int]], pad_value: int) -> torch.Tensor:
        batch_max_length = max([len(t) for t in batch_tokens])
        padded_tokens = [
            pad(x, max_length=batch_max_length, value=pad_value, truncating="pre") for x in batch_tokens
        ]
        return torch.tensor(padded_tokens, dtype=torch.long)

    def close(self):
        self.jsonl_iterator.close()
