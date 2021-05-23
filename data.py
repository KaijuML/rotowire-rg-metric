from torch.utils.data import Dataset as PytorchDataset

import numpy as np
import torch
import h5py


def prep_data(train_filename, eval_filename=None, is_test=False, is_just_eval=False):
    kwargs = {'do_train': True, 'do_test' if is_test else 'do_val': True}
    datasets = load_datasets(train_filename, **kwargs)

    train = datasets.pop('tr')
    val = datasets[next(iter(datasets))]

    test = None
    if eval_filename is not None:
        test = load_datasets(eval_filename, do_test=True)

    min_entdist = min(train['entdists'].min(), val['entdists'].min())
    min_numdist = min(train['numdists'].min(), val['numdists'].min())

    train.shift_dists(min_entdist=min_entdist, min_numdist=min_numdist)
    val.shift_dists(min_entdist=min_entdist, min_numdist=min_numdist)

    if test is not None:
        test.clamp_dist(min_entdist=min_entdist, min_numdist=min_numdist,
                        max_entdist=train['entdists'].max(),
                        max_numdist=train['numdists'].max())
        test.shift_dists(min_entdist=min_entdist, min_numdist=min_numdist)

    nlabels = train['labels'].max().item() + 1
    ent_dist_pad = train['entdists'].max() + 1
    num_dist_pad = train['numdists'].max() + 1
    word_pad = train['sents'].max() + 1

    datasets = [train, val, test]
    min_dists = [min_entdist, min_numdist]
    paddings = [word_pad, ent_dist_pad, num_dist_pad]
    return datasets, min_dists, paddings, nlabels


def load_datasets(filename, do_train=False, do_val=False, do_test=False):
    sets = set()
    if do_train: sets.add('tr')
    if do_val: sets.add('val')
    if do_test: sets.add('test')

    if not len(sets):
        raise RuntimeError('This function was asked to load no data!')

    return make_datasets(filename, sets)


def make_datasets(h5_filename, sets=None):
    sets = sets if sets is not None else ['tr']
    assert all(dname in {'tr', 'val', 'test'} for dname in sets)

    file = h5py.File(h5_filename, mode="r")

    datasets = {dname: dict() for dname in {'tr', 'val', 'test'}}
    for dname, dvals in file.items():
        for prefix, dataset in datasets.items():
            if dname.startswith(prefix):
                dataset[dname[len(prefix):]] = torch.tensor(np.array(dvals))
                break
        else:
            assert dname == 'boxrestartidxs' and 'test' in datasets, dname
            datasets['val'] = torch.tensor(np.array(dvals))

    file.close()
    return {
        dname: __datasets[dname](**datasets[dname])
        for dname in sets
    }


class Dataset(PytorchDataset):
    def __init__(self, entdists, labels, lens, numdists, sents):
        self._len = entdists.size(0)
        assert all(tensor.size(0) == len(self)
                   for tensor in [entdists, labels, lens, numdists, sents])

        self.entdists = entdists
        self.labels = labels
        self.lens = lens
        self.numdists = numdists
        self.sents = sents

    def shift_dists(self, min_entdist, min_numdist):
        self.entdists.add_(-min_entdist)
        self.numdists.add_(-min_numdist)

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return {
                'sents': self.sents[item],
                'entdists': self.entdists[item],
                'numdists': self.numdists[item],
                'lens': self.lens[item],
                'labels': self.labels[item],
            }

        if (ret := getattr(self, item, None)) is not None:
            return ret

        raise AttributeError(f'{self.__class__}')

    def __len__(self):
        return self._len

    def __repr__(self):
        return f'{self.__class__.__name__.title()}(n_examples={len(self)})'


class EvaluationDataset(Dataset):
    def __init__(self, entdists, labels, lens, numdists, sents, boxrestartidxs=None):
        super().__init__(entdists, labels, lens, numdists, sents)
        self.boxrestartidxs = boxrestartidxs

        self.labelnums = self.labels[:, -1]
        self.labels = self.labels[:, :-1]

    def clamp_dists(self, min_entdist, max_entdist, min_numdist, max_numdist):
        self.entdists.clamp_(min_entdist, max_entdist)
        self.numdists.clamp_(min_numdist, max_numdist)

    def __getitem__(self, item):
        ret = super().__getitem__(item)

        if isinstance(item, (int, slice)):
            ret['labelnums'] = self.labelnums[item]
            if self.boxrestartidxs is not None:
                ret['boxrestartidxs'] = self.boxrestartidxs[item]
        return ret


__datasets = {
    'tr': Dataset,
    'val': EvaluationDataset,
    'test': EvaluationDataset,
}
