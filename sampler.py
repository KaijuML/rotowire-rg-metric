from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader


def build_dataset_iter(dataset, batch_size, vocab_sizes, is_eval=False):
    if dataset is None:
        return None

    collate_fn = build_collate_fn(*vocab_sizes)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=not is_eval,
                        collate_fn=collate_fn, pin_memory=True, drop_last=False)

    return loader


def build_collate_fn(word_pad, ent_dist_pad, num_dist_pad):
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
