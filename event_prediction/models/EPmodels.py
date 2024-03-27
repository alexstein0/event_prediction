import torch.nn as nn
import torch

# ideas adapted from fata-trans


class RowEncoder(nn.Module):
    """This is my user defined encoder to move a row into embedding space before doing autoregressive step"""
    def __init__(self, n_cols: int, vocab_size: int, hidden_size: int, col_hidden_size: int,
                 nheads: int = 8, nlayers: int = 1):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, col_hidden_size)

        enc = nn.TransformerEncoderLayer(d_model=col_hidden_size, nhead=nheads, dim_feedforward=col_hidden_size)
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)

        self.linear = nn.Linear(col_hidden_size * n_cols, hidden_size)
        # self.hidden_size = hidden_size
        # self.col_hidden_size = col_hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embeddings(input_ids)
        embeds_shape = list(embedded.size())

        embedded = embedded.view([-1] + embeds_shape[-2:])
        embedded = embedded.permute(1, 0, 2)
        embedded = self.encoder(embedded)
        embedded = embedded.permute(1, 0, 2)
        embedded = embedded.contiguous().view(embeds_shape[0:2] + [-1])

        embedded = self.linear(embedded)

        return embedded


class HierarchicalModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids: torch):
        return self.decoder(self.encoder(input_ids))
