from log_utils import logger as default_logger

import torch
import tqdm
import os


class Inference:
    def __init__(self, vocab_prefix, min_entdist, min_numdist,
                 average_func='arithmetic',
                 ignore_idx=None, logger=None):
        self.ignore_idx = ignore_idx
        self.average_func = average_func

        self.min_entdist, self.min_numdist = min_entdist, min_numdist

        self.words_vocab = self.read_vocab(f'{vocab_prefix}.dict')
        self.labels_vocab = self.read_vocab(f'{vocab_prefix}.labels')

        if self.ignore_idx:
            assert self.labels_vocab[self.ignore_idx] == "NONE"

        self.logger = logger if logger is not None else default_logger

    @staticmethod
    def read_vocab(filename):
        dict_ = dict()
        with open(filename, mode="r", encoding='utf8') as f:
            for line in f:
                if line := line.strip():
                    value, key = line.split()
                    dict_[int(key)] = value
        return dict_

    @staticmethod
    def extract_boxscore_restarts(dataloader):
        boxrestartidxs = getattr(dataloader.dataset, 'boxrestartidxs', None)
        if boxrestartidxs is None:
            return None

        boxscore_restarts = dict()
        for idx in boxrestartidxs:
            if idx not in boxscore_restarts:
                boxscore_restarts[int(idx)] = 0
            boxscore_restarts[int(idx)] += 1

        return boxscore_restarts

    def idxstostring(self, t):
        strtbl = []
        keys = self.words_vocab.keys()
        for i in range(len(t)):
            key = int(t[i])
            if key in keys:
                strtbl.append(self.words_vocab[key])
        return ' '.join(strtbl)

    def get_args(self, sent, ent_dists, num_dists):
        entwrds, numwrds = [], []
        for i in range(sent.size(0)):
            if ent_dists[i] + self.min_entdist == 0:
                entwrds.append(sent[i])
            if num_dists[i] + self.min_numdist == 0:
                numwrds.append(sent[i])
        return self.idxstostring(entwrds), self.idxstostring(numwrds)

    def run(self, dataloader, models, tuple_filename):

        if os.path.exists(tuple_filename):
            self.logger.warn(f'Overwriting {tuple_filename}')
        with open(tuple_filename, mode="w", encoding='utf8') as f:
            pass  # touch

        [model.eval() for model in models]

        boxscore_restarts = self.extract_boxscore_restarts(dataloader)

        correct, total = 0, 0
        candNum = 0
        seen = {}
        ndupcorrects = 0
        nduptotal = 0

        for batch in tqdm.tqdm(dataloader, desc="Running inference"):

            batch_size = batch['sents'].size(0)

            # compute the prediction for each model individualy
            preds = [
                model([batch['sents'], batch['entdists'], batch['numdists']])
                for model in models
            ]

            if self.average_func == 'geometric':
                softmax = torch.nn.functional.log_softmax
            else:
                softmax = torch.nn.functional.softmax

            # Ensemble scores
            preds = sum(softmax(prd, dim=1) for prd in preds)

            # We create a tensor with 1 for every correct label, zero otherwise
            g_one_hot = torch.zeros(batch_size, preds.size(1))
            numpreds = 0

            preds = preds.argmax(dim=1)

            iterable = zip(preds, batch['labels'], batch['labelnums'])
            for idx, (pred, labels, labelnum) in enumerate(iterable):
                if pred != self.ignore_idx:
                    g_one_hot[idx].index_fill_(0, labels[0:labelnum], 1)
                    numpreds += 1

            # we gather the 0-or-1 across predicitons
            g_correct_buf = torch.gather(g_one_hot, 1, preds.unsqueeze(1))

            iterable = zip(preds,
                           batch['sents'],
                           batch['entdists'],
                           batch['numdists'])

            with open(tuple_filename, mode="a", encoding='utf8') as tupfile:
                for idx, (pred, sent, ent_dist, num_dist) in enumerate(iterable):
                    candNum = candNum + 1
                    if boxscore_restarts and candNum in boxscore_restarts:
                        for space_num in range(boxscore_restarts[candNum]):
                            tupfile.write("\n")
                        boxscore_restarts.pop(candNum)
                        seen = dict()

                    if pred != self.ignore_idx:

                        entarg, numarg = self.get_args(sent, ent_dist, num_dist)
                        predkey = entarg + numarg + self.labels_vocab[int(pred)]
                        tupfile.write(entarg + '|' + numarg + '|' + self.labels_vocab[int(pred)])
                        seen_tag = predkey in seen.keys()
                        if g_correct_buf[idx, 0] > 0:
                            tupfile.write('|RIGHT')
                            if seen_tag:
                                ndupcorrects = ndupcorrects + 1
                        else:
                            tupfile.write('|WRONG')

                        if seen_tag:
                            nduptotal = nduptotal + 1
                        tupfile.write('\n')
                        seen[predkey] = True

                correct = correct + g_correct_buf.sum()
                total = total + numpreds

        with open(tuple_filename, mode="a", encoding='utf8') as tupfile:
            for k, v in boxscore_restarts.items():
                for p in range(v):
                    tupfile.write("\n")

        acc = correct / total
        self.logger.info("prec {}".format(acc.item()))
        self.logger.info("nodup prec {}".format((correct - ndupcorrects) / (total - nduptotal)))
        self.logger.info("total correct {}".format(correct.item()))
        self.logger.info("nodup correct {}".format(correct - ndupcorrects))
        return acc
