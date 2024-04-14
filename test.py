from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
import pandas as pd
import numpy as np
import torch
import math
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from model import GRU, BiGRU
import matplotlib.pyplot as plt
import seaborn as sns


# MAPE和SMAPE需要自己实现
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


def FD(x, y):
    distence = np.sqrt(np.power(x[:, 0] - y[:, 0], 2) + np.power(x[:, 1] - y[:, 1], 2))
    return max(distence)


def AED(x, y):
    distence = np.sqrt(np.power(x[:, 0] - y[:, 0], 2) + np.power(x[:, 1] - y[:, 1], 2))
    return np.mean(distence)


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack([item[0] for item in data])  # 1 is feature size
    target = torch.stack([item[1] for item in data])
    return input, target


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
    def __init__(self, input_size, feature_size, num_layers=2, dropout=0.3):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.change_layer = nn.Linear(input_size, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, input_size)
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


np.set_printoptions(suppress=True)
input_window = 5  # number of input steps
output_window = 1  # number of prediction steps, in this model it's fixed to one
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model_BiGRU.pth').to(device)

# 预测过程
model.eval()

uscole = [1]
K = 3

data = np.array(pd.read_csv(r'./data/NANCHANG.csv', usecols=uscole))[14000:, :]
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data).reshape(-1, len(uscole))

train_data = create_inout_sequences(data, input_window).to(device)
batch_size = 32
result = torch.Tensor()
target = torch.Tensor()
for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
    datas, targets = get_batch(train_data, i, batch_size)
    output = model(datas)
    result = torch.cat((result, output.cpu()), dim=0)
    target = torch.cat((target, targets[:, -1, :].cpu()), dim=0)
    # print(output)
# a = target[:, 2].reshape(-1, 1)
# result = torch.cat((result, a), 1)
result = scaler.inverse_transform(result.detach().reshape(-1, 1))
target = scaler.inverse_transform(target.detach())

sns.set_style('darkgrid')
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 设置字体大小为12
plt.rcParams['font.size'] = 14
plt.figure(figsize=(10, 6))
plt.plot(result, color='b', alpha=0.8, linewidth=2)
plt.plot(target, color='r', alpha=0.8, linewidth=2)
plt.legend(['KNN-LSTM', 'True'])
plt.savefig('1.jpg', dpi=300)
plt.show()
#
# RMSE
print('RMSE: %.4f' % np.sqrt(mean_squared_error(result, target)))  # 2.847304489713536
# MAE
print('MAE: %.4f' % mean_absolute_error(result, target))
# MAPE
print('MAPE: %.4f' % mape(result, target))  # 76.07142857142858，即76%
# SMAPE
print('SMAPE: %.4f' % smape(result, target))  # 57.76942355889724，即58%
