import random

from datasets import SceneData
import utils.dataset_utils as dataset_utils
import numpy as np


class ScenesDataSet:
    def __init__(self, data_list, return_all, min_sample_size=10, max_sample_size=30, phase=None):
        super().__init__()
        self.data_list = data_list
        self.return_all = return_all
        self.min_sample_size = min_sample_size
        self.max_sample_size = max_sample_size
        self.phase = phase

    def __getitem__(self, item):
        current_data = self.data_list[item]
        if self.return_all:
            return current_data
        else:
            if self.max_sample_size > 1.0:
                max_sample = min(self.max_sample_size, len(current_data.y))
                if self.min_sample_size >= max_sample:
                    sample_fraction = max_sample
                else:
                    sample_fraction = np.random.randint(self.min_sample_size, max_sample + 1)
            else:
                if len(current_data.y) < 50:
                    self.max_sample_size = 1.0
                    self.min_sample_size = 0.4

                if self.min_sample_size > 1.0:
                    sample_fraction = int(random.uniform(self.min_sample_size,  min(self.max_sample_size * len(current_data.y),100)))
                else:
                    sample_fraction = int(random.uniform(self.min_sample_size * len(current_data.y),  min(self.max_sample_size * len(current_data.y), 100)))

            counter = 0
            while 1:
                data = SceneData.sample_data(current_data, sample_fraction)
                data._compute_pairwise_data()
                if dataset_utils.is_valid_sample(data, min_pts_per_cam=3, phase=self.phase) or counter > 0:
                    return data
                counter += 1


    def __len__(self):
        return len(self.data_list)


def collate_fn(data):
    """
       default collate function for the dataset
    """
    return data


