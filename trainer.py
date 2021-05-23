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
    def __init__(self, logger, save_directory, max_grad_norm=None):
        self.logger = logger

        self.save_directory = save_directory

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

    def train(self, model, loaders, n_epochs=1, lr=0.1):
        model.count_parameters(self.logger.info)

        train_dataloader, val_dataloader, test_dataloader = loaders

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_acc = 0
        for epoch in range(1, n_epochs + 1):
            self.logger.info(f"Epoch {epoch} ({lr=})")

            trainloss = self.run_one_epoch(model, train_dataloader, epoch)
            trainloss = trainloss.item() / len(train_dataloader)

            self.logger.info(f"train loss: {trainloss:.5f}")

            evalloss = .0
            # acc, rec = get_multilabel_acc(model, valbatches, opt.ignore_idx)
            # logger.info("acc:{}".format(acc.item()))

            filename = model.save(self.save_directory, epoch, trainloss, evalloss)
            self.logger.info(f"saving current checkpoint to {filename}")

            # valloss = -acc
            # if valloss >= prev_loss and opt.lr > 0.0001:
            #     opt.lr = opt.lr * opt.lr_decay
            # prev_loss = valloss
