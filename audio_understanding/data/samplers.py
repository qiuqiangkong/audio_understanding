from __future__ import annotations

import random
from typing import Sized

import numpy as np
from torch.utils.data import Dataset


class InfiniteSampler:
    def __init__(self, dataset: Sized) -> None:
        r"""Randomly sample indexes of a dataset without replacement. Execute
        this process infinitely.
        """

        self.indexes = list(range(len(dataset)))
        random.shuffle(self.indexes)  # self.indexes: [3, 7, 0, ...]
        self.p = 0  # pointer
        
    def __iter__(self) -> int:
        r"""Yield an index."""

        while True:

            if self.p == len(self.indexes):
                
                # Traversed all data. Reshuffle indexes. Reset pointer.
                random.shuffle(self.indexes)
                self.p = 0
                
            index = self.indexes[self.p]
            self.p += 1

            yield index


class MultiDatasetsBatchSampler:
    r"""Multiple datasets sampler. First, sample a dataset. Then, sample all 
    batch indexes from the sampled dataset. This class is useful when collating
    datasets with different keys in a batch. 
    """

    def __init__(
        self, 
        datasets: list[Dataset], 
        batch_size: int,
        weights: None | np.ndarray = None  # Probability of datasets to be sampled
    ) -> None:

        self.batch_size = batch_size
        self.datasets_num = len(datasets)
        
        # Probability of datasets to be sampled
        if weights is None:
            self.weights = [dataset.DURATION for dataset in datasets]
        else:
            self.weights = weights
            
        print("Dataset weights: {}".format(self.weights / np.sum(self.weights)))

        # Indexes of audio files of all datasets
        self.indexes_list = [list(range(len(dataset))) for dataset in datasets]

        for indexes in self.indexes_list:
            random.shuffle(indexes)

        self.datasets_lens = [len(indexes) for indexes in self.indexes_list]
        self.p_list = [0 for _ in range(self.datasets_num)]

    def __iter__(self) -> list[int]:
        r"""Sample a batch of indexes. First, sample a dataset. Then, sample all 
        batch indexes from the sampled dataset."""

        while True:

            # Dataset index
            m = random.choices(
                population=range(self.datasets_num), 
                weights=self.weights
            )[0]

            batch_indexes = []

            for _ in range(self.batch_size):

                if self.p_list[m] == self.datasets_lens[m]:
                
                    # Traversed all data. Reshuffle indexes. Reset pointer.    
                    random.shuffle(self.indexes_list[m])
                    self.p_list[m] = 0

                # Get index
                index = self.indexes_list[m][self.p_list[m]]
                self.p_list[m] += 1

                # Get absolute index
                abs_index = int(np.sum(self.datasets_lens[0 : m])) + index
                batch_indexes.append(abs_index)

            yield batch_indexes