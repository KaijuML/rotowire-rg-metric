import torch
import os


class JointEmbeddings(torch.nn.Module):
    def __init__(self, vocab_sizes, emb_sizes):
        super().__init__()
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(vsize, esize)
            for vsize, esize in zip(vocab_sizes, emb_sizes)
        ])

    def forward(self, inputs):
        return torch.cat([
            emb(_input) for emb, _input in zip(self.embeddings, inputs)
        ], dim=2)

    @property
    def emb_dim(self):
        return sum(module.embedding_dim for module in self.embeddings)

    def __getitem__(self, item):
        return self.embeddings[item]


class RgModel(torch.nn.Module):
    """
    Base Rg model class, to be slightly modifed depending on which backbone
    architecture you want to use. Like in the original code, LSTM and Conv
    have been implemented further down this file.
    """
    def __init__(self, vocab_sizes, emb_sizes):
        super().__init__()

        self.hparams = None
        self.embeddings = JointEmbeddings(vocab_sizes, emb_sizes)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def __getitem__(self, item):
        if item != 0:
            raise ValueError('Can only get embeddings (index=0) from a RgModel')
        return self.embeddings

    def uniform_initialization(self):
        for mod in self.modules():
            if hasattr(mod, "weight"):
                torch.nn.init.uniform_(mod.weight, -.1, .1)
            if hasattr(mod, "bias") and isinstance(mod.bias, torch.Tensor):
                torch.nn.init.uniform_(mod.bias, -.1, .1)

    def store_hparams(self, **kwargs):
        self.hparams = kwargs

    def save(self, directory, epoch, accuracy, recall):
        filename = f'{self.class_name.lower()}.epoch_{epoch}'
        filename += f'.acc={accuracy:.3f}.rec={recall:.3f}.pt'
        filename = os.path.join(directory, filename)

        torch.save({
            "model_class": self.class_name,
            "model_hparams": self.hparams,
            "model_state_dict": self.state_dict(),

            "epoch": epoch,
            "accuracy": accuracy,
            "recall": recall
        }, filename)

        return filename

    @staticmethod
    def from_file(filename):
        ckpt = torch.load(filename)
        model = available_models[ckpt['model_class']](**ckpt['model_hparams'])
        model.load_state_dict(ckpt['model_state_dict'])
        return model

    @property
    def emb_dim(self):
        # return sum(module.embedding_dim for module in self.embeddings)
        return self.embeddings.emb_dim

    def count_parameters(self, log=print, module_names=None):
        """
        Count number of parameters in model (& print with `log` callback).
        If module names is specified, separate counts per module
        Returns:
            int or Union[int]: total number of parameters OR nb per module
        """
        if module_names is None:
            count = sum(p.nelement() for p in self.parameters())
            if callable(log):
                log(f'Total number of parameters: {count:,}')
            return count

        counts = [0 for _ in module_names] + [0]
        for pname, param in self.named_parameters():
            for idx, mname in enumerate(module_names):
                if mname in pname:
                    counts[idx] += param.nelement()
                    break
            else:
                counts[-1] += param.nelement()

        if callable(log):
            for mname, count in zip(module_names, counts):
                log(f'{mname}: {count:,} params.')
            if counts[-1] != 0:
                log(f'Unattributed: {counts[-1]:,} params.')
            log(f'Total number of parameters: {sum(counts):,}')

        if counts[-1] == 0: counts.pop(-1)
        return counts


class RecurrentRgModel(RgModel):

    class_name = 'lstm'

    def __init__(self, vocab_sizes, emb_sizes, hidden_dim, nlabels, dropout=0):
        super().__init__(vocab_sizes, emb_sizes)
        self.store_hparams(vocab_sizes=vocab_sizes,
                           emb_sizes=emb_sizes,
                           hidden_dim=hidden_dim,
                           nlabels=nlabels,
                           dropout=dropout)

        self.rnn = torch.nn.LSTM(self.emb_dim, self.emb_dim, 1, bidirectional=True)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(2 * self.emb_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, nlabels)
        )

        self.uniform_initialization()

    def forward(self, inputs):
        embedded_inputs = self.embeddings(inputs).transpose(0, 1).contiguous()
        outputs = torch.max(self.rnn(embedded_inputs)[0], dim=0)[0]
        return torch.nn.functional.softmax(self.linear(outputs), dim=-1)


class ConvRgModel(RgModel):

    class_name = 'conv'

    def __init__(self, vocab_sizes, emb_sizes, num_filters, hidden_dim, nlabels, dropout=.0):
        super().__init__(vocab_sizes, emb_sizes)
        self.store_hparams(vocab_sizes=vocab_sizes,
                           emb_sizes=emb_sizes,
                           num_filters=num_filters,
                           hidden_dim=hidden_dim,
                           nlabels=nlabels,
                           dropout=dropout)

        kernel_widths = [2, 3, 5]
        self.convolutions = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(self.emb_dim, num_filters, kwidth, padding=kwidth - 1),
                torch.nn.ReLU()
            )
            for kwidth in kernel_widths
        ])

        self.linear = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(3 * num_filters, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, nlabels)
        )

        self.uniform_initialization()

    def run_convolutions(self, inputs):
        return torch.cat([
            torch.max(conv(inputs), dim=2)[0]
            for conv in self.convolutions
        ], dim=1)

    def forward(self, inputs):
        embedded_inputs = self.embeddings(inputs).transpose(1, 2).contiguous()
        outputs = self.run_convolutions(embedded_inputs)
        return torch.nn.functional.softmax(self.linear(outputs), dim=-1)


class Ensemble(torch.nn.Module):
    def __init__(self, models, average_func='arithmetic'):
        super().__init__()
        self.average_func = average_func
        self.models = torch.nn.ModuleList(models)

    def forward(self, inputs):

        # compute the prediction for each model individualy
        preds = [model(inputs) for model in self.models]

        if self.average_func == 'geometric':
            softmax = torch.nn.functional.log_softmax
        else:
            softmax = torch.nn.functional.softmax

        # Ensemble scores
        return sum(softmax(prd, dim=1) for prd in preds)


available_models = {
    'lstm': RecurrentRgModel,
    'conv': ConvRgModel
}
