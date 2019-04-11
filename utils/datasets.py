import torch


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *data_sets):
        self.data_sets = data_sets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.data_sets)

    def __len__(self):
        return min(len(d) for d in self.data_sets)