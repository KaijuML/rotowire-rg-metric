from models import RecurrentRgModel
import torch
import tqdm
import os


class MultilabelCrossEntropyLoss(torch.nn.Module):
    """
    Similar to NLLLoss, minimizes the log proba of the target class
    In the multilabel cases, probas of correct classes are summed.
    """

    def __init__(self, reduction=True):
        super(MultilabelCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.tol = 1e-6

    def forward(self, prd, tgt):
        """
        :param prd: [batch_size, nlabels] the tensor of prediction logits
        :param tgt: [batch_size, X + 1] where X is the max number of labels in
                    the entire datasets. The last columns indicates how many
                    labels are correct
        :return: the loss, averaged over batch dim if self.reduction is True
        """
        assert prd.size(0) == tgt.size(0)  # same batch_size

        num_correct_labels = tgt[:, -1]
        labels = tgt[:, :-1]

        loss = torch.tensor(0.0).to(prd.device)
        for _prd, _tgt, _n in zip(prd, labels, num_correct_labels):

            # Get sum of probas over all correct labels
            _loss = _prd.index_select(0, index=_tgt[:_n]).sum()

            # Add its log to current running loss
            loss -= torch.log(_loss + self.tol)

        if self.reduction:
            loss = loss / prd.size(0)

        return loss


class Trainer:
    def __init__(self,
                 paddings,
                 logger,
                 save_directory=None,
                 max_grad_norm=5,
                 ignore_idx=None):

        self.logger = logger

        self.word_pad, self.ent_dist_pad, self.num_dist_pad = paddings

        self.save_directory = save_directory
        if self.save_directory and not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        self.ignore_idx = ignore_idx

        self.criterion = MultilabelCrossEntropyLoss()
        self.max_grad_norm = max_grad_norm

    def compute_multilabel_acc(self, model, dataloader, break_after=None, log=True):

        model.eval()

        correct, total, ignored = 0, 0, 0
        nonnolabel = 0

        break_after = break_after or len(dataloader)
        for bidx, batch in tqdm.tqdm(enumerate(dataloader),
                                     total=break_after,
                                     desc="Computing accuracy"):

            # Usefull to run only a few iterations when debugging
            if bidx >= break_after:
                break

            batch_size = batch['sents'].size(0)

            with torch.no_grad():
                preds = model([batch['sents'], batch['entdists'], batch['numdists']])

            nonnolabel = nonnolabel + batch["labels"][:, 0].ne(self.ignore_idx).sum()
            g_one_hot = torch.zeros(batch_size, preds.size(1), device=model.device)
            preds = preds.argmax(dim=1)

            numpreds = 0

            iterable = zip(preds, batch["labels"], batch["labelnums"])
            for idx, (pred, labels, labelnum) in enumerate(iterable):
                if pred != self.ignore_idx:
                    g_one_hot[idx].index_fill_(0, labels[0:labelnum], 1)
                    numpreds = numpreds + 1

            g_correct_buf = torch.gather(g_one_hot, 1, preds.unsqueeze(1))
            correct = correct + g_correct_buf.sum()
            total = total + numpreds
            ignored = ignored + batch_size - numpreds

        accuracy = correct / total
        recall = correct / nonnolabel

        if log:
            self.logger.info(f"recall: {recall.item():.3f}%")
            self.logger.info(f"ignored: {ignored/(ignored+total):.3f}%")

        return accuracy, recall

    def run_one_epoch(self, model, dataloader, learning_rate, epoch):

        running_loss = 0.0
        model.train()

        with torch.no_grad():
            model[0][0].weight[self.word_pad].zero_()
            model[0][1].weight[self.ent_dist_pad].zero_()
            model[0][2].weight[self.num_dist_pad].zero_()

        with tqdm.tqdm(total=len(dataloader)) as progressbar:
            for step, batch in enumerate(dataloader, 1):
                model.zero_grad()

                prd = model([batch['sents'], batch['entdists'], batch['numdists']])
                tgt = batch['labels']

                loss = self.criterion(prd, tgt)
                loss.backward()

                with torch.no_grad():
                    running_loss += loss.item()

                    if isinstance(model, RecurrentRgModel):
                        model[0][0].weight.grad[self.word_pad].zero_()
                        model[0][1].weight.grad[self.ent_dist_pad].zero_()
                        model[0][2].weight.grad[self.num_dist_pad].zero_()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       self.max_grad_norm, 2)

                    for p in model.parameters():
                        p.add_(-learning_rate * p.grad)

                    # optimizer.step()
                    model[0][0].weight[self.word_pad].zero_()
                    model[0][1].weight[self.ent_dist_pad].zero_()
                    model[0][2].weight[self.num_dist_pad].zero_()

                progressbar.update(1)

                new_desc = f"Training (Epoch={epoch}, avg_loss={(running_loss/step):.3f})"
                progressbar.set_description(new_desc, refresh=True)

        return running_loss

    def train(self, model, loaders, n_epochs=1, lr=0.1, lr_decay=1):

        train_dataloader, val_dataloader, _ = loaders

        self.logger.info('Running 3 validation batches before training to catch bugs')
        _ = self.compute_multilabel_acc(model, val_dataloader,
                                        break_after=3, log=False)

        prev_loss = float('inf')

        with tqdm.tqdm(total=n_epochs) as progressbar:

            progressbar.set_description(f'Training total (acc=..., rec=...)',
                                        refresh=True)

            for epoch in range(1, n_epochs+1):

                trainloss = self.run_one_epoch(model, train_dataloader, lr, epoch)
                trainloss = trainloss / len(train_dataloader)

                accuracy, recall = self.compute_multilabel_acc(model, val_dataloader, log=False)

                _ = model.save(self.save_directory, epoch, accuracy, recall)

                evalloss = -accuracy
                if evalloss >= prev_loss and lr > 1e-4:
                    lr *= lr_decay
                prev_loss = evalloss

                progressbar.update(1)

                new_desc = f'Training total (acc={accuracy:.3f}, rec={recall:.3f})'
                progressbar.set_description(new_desc, refresh=True)
