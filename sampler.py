"""
Utilities to iterate over data by batch of examples.
build_dataset_iter and IterOnDevice are built very similarily to functions
with the same name in OpenNMT-py (https://github.com/OpenNMT/OpenNMT-py)
"""


from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader

import torch


def build_dataset_iter(dataset, batch_size, vocab_sizes, device=None, is_eval=False):
    """
    Given a dataset and a batch_size, creates a DataLoader that can yields
    batches of examples. These batches are on the correct device.

    :param dataset: the dataset of examples. Should implement datatset[idx]
                    that returns a dict of tensors.
    :param batch_size: number of examples per batch
    :param vocab_sizes: padding is done using vocab_size in this project
    :param device: torch.device object on which to move the batches
    :param is_eval: Shuffle training examples but keep eval in correct order
    :return: A dataloader. Usage: `for batch in dataloader: ...`
    """
    if dataset is None:
        return None

    device = torch.device('cpu') if device is None else device

    collate_fn = build_collate_fn(*vocab_sizes)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=not is_eval,
                        collate_fn=collate_fn, pin_memory=True, drop_last=False)

    return IterOnDevice(loader, device)


class IterOnDevice:
    def __init__(self, iterable, device):
        self.iterable = iterable
        self.device = device

    def __len__(self):
        return len(self.iterable)

    def to_device(self, batch):
        return {
            tname: tensor.to(self.device)
            for tname, tensor in batch.items()
        }

    def __iter__(self):
        for batch in self.iterable:
            yield self.to_device(batch)


def build_collate_fn(word_pad, ent_dist_pad, num_dist_pad):
    """
    a collate_fn is used to merge a number of examples into one batch.
    """
    def collate_fn(batch):
        for example in batch:
            _len = example['lens'].item()
            example['sents'][_len:].fill_(word_pad)
            example['entdists'][_len:].fill_(ent_dist_pad)
            example['numdists'][_len:].fill_(num_dist_pad)

        batch = default_collate(batch)
        max_len = batch['lens'].max()
        return {
            tname: tensor[:, :max_len] if tensor.dim() == 2 else tensor
            for tname, tensor in batch.items()
        }

    return collate_fn
