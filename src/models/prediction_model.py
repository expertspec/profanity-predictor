import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM_attention, self).__init__()

        # Define the LSTM layers
        self.lstm1 = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.attention1 = Attention(hidden_dim * 2, batch_first=True)
        self.attention2 = Attention(hidden_dim * 2, batch_first=True)
        # Define the output layer
        self.fc = nn.Linear(hidden_dim * 4, output_dim)

        self.softmax = nn.Softmax()

    def forward(self, x):
        # Pass the input sequence through the LSTM layer
        x = torch.Tensor(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.dropout(x)
        out1, (h_n, c_n) = self.lstm1(x)
        x, _ = self.attention1(out1, 64)
        out2, (h_n, c_n) = self.lstm2(out1)
        y, _ = self.attention2(out2, 64)

        try:
            out = torch.cat([x, y], dim=1)
        # for prediction when batch size is 1
        except IndexError:
            out = torch.cat([x, y], dim=0)
        out = self.fc(out)
        out = self.softmax(out)

        return out


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(
            torch.Tensor(1, hidden_size), requires_grad=True
        )

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(
            inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1)
        )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True)

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions
