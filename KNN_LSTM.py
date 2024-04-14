import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import math
from matplotlib import pyplot
from scipy.spatial.distance import cdist
from model import GRU, BiGRU, TransAm

# torch.manual_seed(0)
# np.random.seed(0)
np.set_printoptions(suppress=True)


def create_inout_sequences_knn(input_data, tw):
    inout_seq = []
    out_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        distances = cdist(train_seq, train_seq, 'euclidean')
        max_indices = np.argsort(-distances, axis=1)[:, :2]
        train_sep_near_1 = []
        train_sep_near_2 = []
        for j in range(train_seq.shape[0]):
            train_sep_near_1.append(train_seq[max_indices[j][0]] * distances[j][max_indices[j][0]])
            train_sep_near_2.append(train_seq[max_indices[j][1]] * distances[j][max_indices[j][1]])
        train_sep_near_1 = np.array(train_sep_near_1)
        train_sep_near_2 = np.array(train_sep_near_2)
        train_seq = np.concatenate((train_seq, train_sep_near_1, train_sep_near_2), axis=1)
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, np.concatenate((train_label, train_label, train_label), axis=1)))
    return torch.FloatTensor(inout_seq)


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_data(filepath, uscols=None):
    if uscols is None:
        uscols = [0]
    data = np.array(pd.read_csv(filepath, usecols=uscols))[:, :]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data).reshape(-1, len(uscols))

    train_data = data[:int(data.shape[0] * 0.8)]
    test_data = data[int(data.shape[0] * 0.8):]

    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment..
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]
    # todo: fix hack? -> didn't think this through, looks like the last n sequences are to short, so I just remove
    #  them. Hackety Hack..

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]  # todo: fix hack?

    return train_sequence.to(device), test_data.to(device)


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack([item[0] for item in data])  # 1 is feature size
    target = torch.stack([item[1] for item in data])
    return input, target


def train(train_data):
    model.train()  # Turn on the train mode \o/
    total_loss = 0.
    total = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        # data, targets = np.squeeze(data), np.squeeze(targets)
        optimizer.zero_grad()
        output = model(data)
        tar = targets[:, -1, 0].unsqueeze(-1)
        loss = criterion(output, tar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total += loss.item()
        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(epoch, batch, len(train_data) // batch_size,
                                                      scheduler.get_last_lr()[0], elapsed * 1000 / log_interval,
                                                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    file_write_obj = open("loss_BiGRU.txt", 'a')
    file_write_obj.writelines(str(total / len(train_data) * batch_size))
    file_write_obj.write('\n')
    file_write_obj.close()


def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy() -> no need to detach stuff.
    len(test_result)

    pyplot.plot(test_result, color="red")
    pyplot.plot(truth[:500], color="blue")
    pyplot.plot(test_result - truth, color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-epoch%d.png' % epoch)
    pyplot.close()

    return total_loss / i


# predict the next n steps based on the input data
# def predict_future(eval_model, data_source, steps):
#     eval_model.eval()
#     total_loss = 0.
#     test_result = torch.Tensor(0)
#     truth = torch.Tensor(0)
#     data, _ = get_batch(data_source, 0, 1)
#     with torch.no_grad():
#         for i in range(0, steps):
#             output = eval_model(data[-input_window:])
#             data = torch.cat((data, output[-1:]))
#
#     data = data.cpu().view(-1)
#
#     # I used this plot to visualize if the model pics up any long therm struccture within the data.
#     pyplot.plot(data, color="red")
#     pyplot.plot(data[:input_window], color="blue")
#     pyplot.grid(True, which='both')
#     pyplot.axhline(y=0, color='k')
#     pyplot.savefig('graph/transformer-future%d.png' % steps)
#     pyplot.close()
#
#
# def evaluate(eval_model, data_source):
#     eval_model.eval()  # Turn on the evaluation mode
#     total_loss = 0.
#     eval_batch_size = 32
#     with torch.no_grad():
#         for i in range(0, len(data_source) - 1, eval_batch_size):
#             data, targets = get_batch(data_source, i, eval_batch_size)
#             # data, targets = np.squeeze(data), np.squeeze(targets)[:, :, :2]
#             output = eval_model(data)
#             total_loss += len(data[0]) * criterion(output, targets[:, -1, 0].unsqueeze(-1)).cpu().item()
#     return total_loss / len(data_source)


filepath = r'./data/NANCHANG.csv'

input_window = 5  # number of input steps
output_window = 1  # number of prediction steps, in this model it's fixed to one
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
uscole = [1]
K = 3
train_data, val_data = get_data(filepath, uscole)

model = BiGRU(n_val=len(uscole), window=input_window, hidRNN=batch_size).to(device)

criterion = nn.MSELoss()
lr = 0.0001
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.98)

best_val_loss = float("inf")
epochs = 500  # The number of epochs
best_model = None
train_loss = []

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)

    # if epoch % 10 == 0:
    #     val_loss = plot_and_loss(model, val_data, epoch)
    #     predict_future(model, val_data, 200)
    # else:
    #     val_loss = evaluate(model, val_data)
    val_loss = evaluate(model, val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
            time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        torch.save(model, r'model_BiGRU.pth')

    scheduler.step()
