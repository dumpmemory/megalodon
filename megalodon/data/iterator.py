import json
from logging import getLogger
from typing import Dict, Iterator

logger = getLogger()


class JSONLIterator:
    def __init__(
        self,
        fpath: str,
        world_size: int,
        world_rank: int,
    ):
        assert 0 <= world_rank < world_size, (world_rank, world_size)
        self.f = open(fpath, "r", encoding="utf-8")
        self.fpath = fpath
        self.world_size = world_size
        self.world_rank = world_rank
        self.line_num = 0
        self.iter = iter(self.gen())
        self.iter_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)

    def gen(self) -> Iterator[Dict]:
        for line in self.f:
            self.line_num = self.line_num + 1
            if (self.line_num - 1) % self.world_size == self.world_rank:
                try:
                    sample = json.loads(line)
                except:
                    logger.error(
                        f"Error when trying to decode line {self.line_num} in {self.fpath}"
                    )
                else:
                    yield sample
        self.f.close()

    def close(self):
        self.f.close()
