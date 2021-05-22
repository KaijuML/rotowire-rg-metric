import torch


class RgModel(torch.nn.Module):
    """
    Base Rg model class, to be slightly modifed depending on which backbone
    architecture you want to use. Like in the original code, LSTM and Conv
    have been implemented further down this file.
    """
    def __init__(self, vocab_sizes, emb_sizes):
        super().__init__()

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(vsize, esize)
            for vsize, esize in zip(vocab_sizes, emb_sizes)
        ])

    @property
    def emb_dim(self):
        return sum(module.embedding_dim for module in self.embeddings)

    def run_embeddings(self, inputs):
        if len(inputs) != len(self.embeddings):
            raise RuntimeError('Wrong number of embeddings: '
                               f'{len(inputs)=} vs {len(self.embeddings)=}')
        return torch.cat([
            emb(_input) for emb, _input in zip(self.embeddings, inputs)
        ], dim=2)


class RecurrentRgModel(RgModel):
    def __init__(self, vocab_sizes, emb_sizes, hidden_dim, nlabels, dropout=0):
        super().__init__(vocab_sizes, emb_sizes)

        self.rnn = torch.nn.LSTM(self.emb_dim, self.emb_dim, 1, bidirectional=True)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(2 * self.emb_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, nlabels)
        )

    def forward(self, inputs):
        embedded_inputs = self.run_embeddings(inputs)
        embedded_inputs = embedded_inputs.transpose(0, 1).contiguous()

        outputs = self.rnn(embedded_inputs)[0].max(0)[0]

        return self.linear(outputs)


class ConvRgModel(RgModel):
    def __init__(self, vocab_sizes, emb_sizes, num_filters, hidden_dim, nlabels, dropout=0):
        super().__init__(vocab_sizes, emb_sizes)

        kernel_widths = [2, 3, 5]
        self.convolutions = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(self.emb_dim, num_filters, kwidth, padding=kwidth - 1),
                torch.nn.ReLU()
            )
            for kwidth in kernel_widths
        ])

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(3 * num_filters, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, nlabels)
        )

    def run_convolutions(self, inputs):

        return torch.cat([
            torch.max(conv(inputs), dim=2)[0]
            for conv in self.convolutions
        ], dim=1)

    def forward(self, inputs):
        embedded_inputs = self.run_embeddings(inputs)
        embedded_inputs = embedded_inputs.transpose(1, 2).contiguous()

        outputs = self.run_convolutions(embedded_inputs)

        return self.linear(outputs)
