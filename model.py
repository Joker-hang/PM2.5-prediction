import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GRU(nn.Module):
    def __init__(self, n_val, window, hidRNN):
        super(GRU, self).__init__()
        self.use_cuda = True
        self.P = window  # 输入窗口大小
        self.m = n_val  # 列数，变量数
        self.hidR = hidRNN
        self.GRU = nn.GRU(self.m, self.hidR)

        self.linear = nn.Linear(self.hidR, self.m)

    def forward(self, x):
        # x: [batch, window, n_val]
        #         batch_size = x.shape[0]
        #         x_flat = x.view(batch_size, -1)
        x1 = x.permute(1, 0, 2).contiguous()  # x1: [window, batch, n_val]
        _, h = self.GRU(x1)  # r: [1, batch, hidRNN]
        h = torch.squeeze(h, 0)  # r: [batch, hidRNN]
        res = self.linear(h)  # res: [batch, n_val]
        return res


class LSTM(nn.Module):
    def __init__(self, n_val, window, hidRNN):
        super(LSTM, self).__init__()
        self.use_cuda = True
        self.P = window  # 输入窗口大小
        self.m = n_val  # 列数，变量数
        self.hidR = hidRNN
        self.LSTM = nn.LSTM(self.m, self.hidR)

        self.linear = nn.Linear(self.hidR, self.m)

    def forward(self, x):
        # x: [batch, window, n_val]
        #         batch_size = x.shape[0]
        #         x_flat = x.view(batch_size, -1)
        x1 = x.permute(1, 0, 2).contiguous()  # x1: [window, batch, n_val]
        _, h = self.LSTM(x1)  # r: [1, batch, hidRNN]
        h = torch.squeeze(h, 0)  # r: [batch, hidRNN]
        res = self.linear(h)  # res: [batch, n_val]
        return res


class RNN(nn.Module):
    def __init__(self, n_val, window, hidRNN):
        super(RNN, self).__init__()
        self.use_cuda = True
        self.P = window  # 输入窗口大小
        self.m = n_val  # 列数，变量数
        self.hidR = hidRNN
        self.RNN = nn.RNN(self.m, self.hidR)

        self.linear = nn.Linear(self.hidR, self.m)

    def forward(self, x):
        # x: [batch, window, n_val]
        #         batch_size = x.shape[0]
        #         x_flat = x.view(batch_size, -1)
        x1 = x.permute(1, 0, 2).contiguous()  # x1: [window, batch, n_val]
        _, h = self.RNN(x1)  # r: [1, batch, hidRNN]
        h = torch.squeeze(h, 0)  # r: [batch, hidRNN]
        res = self.linear(h)  # res: [batch, n_val]
        return res


class BiGRU(nn.Module):
    def __init__(self, n_val, window, hidRNN):
        super(BiGRU, self).__init__()
        self.use_cuda = True
        self.P = window  # 输入窗口大小
        self.m = n_val  # 列数，变量数
        self.hidR = hidRNN
        self.GRU = nn.GRU(self.m, self.hidR, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(self.hidR * 2, self.m)

    def forward(self, x):
        # x: [batch, window, n_val]
        #         batch_size = x.shape[0]
        #         x_flat = x.view(batch_size, -1)
        # x1 = x.permute(1, 0, 2).contiguous()  # x1: [window, batch, n_val]
        out, h = self.GRU(x)  # r: [1, batch, hidRNN]
        h = torch.squeeze(h, 0)  # r: [batch, hidRNN]
        res = self.linear(out)  # res: [batch, n_val]
        return res[:, -1, :]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        position_embedding = self.pe[:x.size(0), :]
        input = x + position_embedding
        return input


class TransAm(nn.Module):
    def __init__(self, input_size, feature_size=128, num_layers=2, dropout=0.3):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.change_layer = nn.Linear(input_size, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.change_layer(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask