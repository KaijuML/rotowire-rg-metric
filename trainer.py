import torch
import tqdm


class MultilabelCrossEntropyLoss(torch.nn.Module):
    """
    Similar to NLLLoss, minimizes the log proba of the target class
    In the multilabel cases, probas of correct classes are summed.
    """

    def __init__(self, ignore_idx=None, reduction=True):
        super(MultilabelCrossEntropyLoss, self).__init__()
        self.ignore_idx = ignore_idx
        self.reduction = reduction
        self.tol = 1e-5

    def forward(self, prd, tgt):
        """
        :param prd: [batch_size, nlabels] the tensor of prediction logits
        :param tgt: [batch_size, X + 1] where X is the max number of labels in
                    the entire datasets. The last columns indicates how many
                    labels are correct
        :return: the loss, averaged over batch dim if self.reduction is True
        """
        assert prd.size(0) == tgt.size(0)

        num_correct_labels = tgt[:, -1:]

        # identify items with no correct label, or only the ignore_idx
        if self.ignore_idx is not None:
            num_correct_labels -= tgt[:, :-1].eq(self.ignore_idx).sum(dim=1, keepdim=True)
        index = num_correct_labels.squeeze(1).gt(0).nonzero().squeeze(1).long()

        # remove such items
        prd = prd.index_select(dim=0, index=index)
        tgt = tgt.index_select(dim=0, index=index)

        # replace ignore_idx by -1 (which is padding in this code)
        labels = tgt[:, :-1]
        if self.ignore_idx is not None:
            labels = torch.where(labels.eq(self.ignore_idx),
                                 torch.full(labels.size(), -1),
                                 labels)

        # Compute the softmax over logits
        prd = torch.nn.functional.softmax(prd, dim=1)

        loss = torch.tensor(0.0).to(prd.device)
        for _prd, _tgt in zip(prd, labels):

            # Get sum of probas over all correct labels
            index = (~_tgt.eq(-1)).nonzero().squeeze(1)
            _loss = _prd.index_select(0, index=index).sum()

            # Add its log to current running loss
            loss -= torch.log(_loss + self.tol)

        if self.reduction:
            loss = loss / prd.size(0)

        return loss


class Trainer:
    def __init__(self, logger, save_directory, max_grad_norm=None, ignore_idx=None):
        self.logger = logger

        self.save_directory = save_directory

        self.ignore_idx = ignore_idx

        self.max_grad_norm = max_grad_norm
        self.optimizer = None

        self.criterion = MultilabelCrossEntropyLoss()

    def run_one_epoch(self, model, dataloader, epoch):
        running_loss = 0

        model.train()
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch}"):
            self.optimizer.zero_grad()
            prd = model([batch['sents'], batch['entdists'], batch['numdists']])
            tgt = batch['labels']

            loss = self.criterion(prd, tgt)
            loss.backward()

            running_loss += loss.item()

            if (clip := self.max_grad_norm) is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            self.optimizer.step()

        return running_loss

    def compute_multilabel_acc(self, model, dataloader):

        correct, total, ignored = 0, 0, 0
        nonnolabel = 0

        for batch in tqdm.tqdm(dataloader, desc="Computing accuracy"):
            batch_size = batch['sents'].size(0)

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

        self.logger.info(f"recall: {recall.item():.3f}%")
        self.logger.info(f"ignored: {ignored/(ignored+total):.3f}%")

        return accuracy, recall

    def train(self, model, loaders, n_epochs=1, lr=0.1, lr_decay=1):
        model.count_parameters(self.logger.info)

        train_dataloader, val_dataloader, test_dataloader = loaders

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        prev_loss = float('inf')
        for epoch in range(1, n_epochs + 1):
            self.logger.info(f"Epoch {epoch} ({lr=})")

            trainloss = self.run_one_epoch(model, train_dataloader, epoch)
            trainloss = trainloss.item() / len(train_dataloader)

            self.logger.info(f"train loss: {trainloss:.5f}")

            accuracy, recall = self.compute_multilabel_acc(model, val_dataloader)
            self.logger.info(f"acc:{accuracy.item()}")

            filename = model.save(self.save_directory, epoch, accuracy, recall)
            self.logger.info(f"saving current checkpoint to {filename}")

            evalloss = -accuracy
            if evalloss >= prev_loss and lr > 1e-4:
                lr *= lr_decay
            prev_loss = evalloss
