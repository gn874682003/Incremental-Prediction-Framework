import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from torch.optim import lr_scheduler
import numpy as np
import torch
import os
from torch import Tensor

LR = 0.001

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=20, verbose=False, delta=0.1):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

def test(X_Test, Y_Test, rnn, type, data):
    eval_loss = 0
    loss_func = nn.L1Loss()
    pred_y = []
    true_y = []
    if type == 1:
        Metric = []
        for x, y in zip(X_Test, Y_Test):
            output, prediction = rnn(x, data)
            prediction = output
            # prediction = get_att_dis(prediction, data['0'])
            for line1, line2 in zip(y.numpy().tolist()[0], prediction):
                true_y.append(line1)
                pred_y.append(line2)
        Metric.append(accuracy_score(true_y, pred_y))
    else:
        Metric = []
        represents = []
        for x, y in zip(X_Test, Y_Test):
            output, represent = rnn(x, data)#
            output = output.view(output.size(0),output.size(1))
            # represents.append(represent)
            # 记录误差
            loss = loss_func(output, y)
            eval_loss += loss.item()
            for line1, line2 in zip(y.numpy().tolist()[0], output.detach().numpy().tolist()[0]):
                true_y.append(line1)
                pred_y.append(line2)
        Metric.append(mean_absolute_error(true_y, pred_y))
    return Metric, eval_loss#len(true_y), represents

def trian(Train,Test_X, Test_Y ,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None,isEarly=0):
    # LR = 0.001
    save_path = '../Save/multiFea/'  # 当前目录下
    early_stopping = EarlyStopping(save_path)
    train_loss = 0
    if method == 'rnn':
        method = RNN(input_size, hidden_size, num_layers, data)
    elif method == 'rnnm':
        method = RNNM(input_size, hidden_size, num_layers, data)
    # else:
    #     LR = 0.01
    optimizer = torch.optim.Adam(method.parameters(), lr=LR)
    # 1.学习率下降：每过n个epoch，学习率乘以0.1
    scheduler = lr_scheduler.StepLR(optimizer, 100, 0.1)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for j, (x, y, l) in enumerate(Train):
            ty = nn.utils.rnn.pack_padded_sequence(y, l, batch_first=True)
            output, represents = method(x, data)#
            py = nn.utils.rnn.pack_padded_sequence(output, l, batch_first=True)
            optimizer.zero_grad()
            if type == 1:
                temp = []
                for line in ty.data:
                    temp.append(data['0'][int(line)])
                ty = torch.FloatTensor(temp)
                loss = loss_func(py.data, ty)
            else:
                loss = loss_func(py.data.view(py.data.size(0)), ty.data)
            loss.backward()
            # 2.梯度剪裁
            nn.utils.clip_grad_norm_(method.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            # 记录误差
            train_loss += loss.item()
        scheduler.step()
        if isEarly == 1:
            Metric, eval_loss = test(Test_X, Test_Y, method, type, data, input_size)
            # 早停止
            early_stopping(eval_loss, method)
            # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练
    if isEarly == 0:
        Metric, count = test(Test_X, Test_Y, method, type, data)
    print(i, Metric)
    return method, Metric#, count

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data):
        super(RNN, self).__init__()
        self.input_size = input_size
        if input_size == -1:  # 全部拼接，索引编码
            input_size = len(data['index'][0])
        elif input_size == -2:  # 全部拼接，CBOW和向量编码
            input_size = 0
            for i in range(len(data['index'][0])):
                if data['state'][0][i] == 0:  # 活动
                    input_size = 1
                    if '0' in data.keys():
                        self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                        input_size = self.embed1.embedding_dim
                        if type == 1:
                            output_size = input_size
                elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                    if str(data['index'][0][i]) in data.keys():
                        setattr(self, 'embed' + str(i + 1),
                                nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                        input_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                    else:
                        input_size += 1
                elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                    input_size += 1
        elif input_size == -3:  # 全部拼接，活动CBOW和其他索引编码
            input_size = 0
            for i in range(len(data['index'][0])):
                if data['state'][0][i] == 0:  # 活动
                    input_size = 1
                    if '0' in data.keys():
                        self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                        input_size = self.embed1.embedding_dim+len(data['index'][0])-1
        elif input_size == 0:
            self.embed = nn.Embedding.from_pretrained(torch.tensor(data['0']))
            input_size = self.embed.embedding_dim
        hidden_size = input_size
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size*2,
            num_layers=num_layers,
            dropout=0.2,  # 3.丢弃层
            batch_first=True
        )
        # self.out = nn.Linear(hidden_size, 1)
        self.out1 = nn.Linear(hidden_size*2, hidden_size)
        self.out2 = nn.Linear(hidden_size, 1)#int(hidden_size/2)
        # 正则初始化
        nn.init.orthogonal(self.rnn.weight_ih_l0)
        nn.init.orthogonal(self.rnn.weight_hh_l0)
        # nn.init.zeros_(self.rnn.bias_ih_l0)
        # nn.init.zeros_(self.rnn.bias_hh_l0)

    def forward(self, x, data):
        if self.input_size == -1:  # 全部拼接，索引编码
            r_out = self.rnn(x)
        elif self.input_size == -2:  # 全部拼接，CBOW和向量编码
            for i in range(len(data['index'][0])):
                if data['state'][0][i] == 0:  # 活动
                    xi = x[:, :, i].view(x.size(0), x.size(1))
                    if str(i) in data.keys():
                        xi = torch.LongTensor(np.array(xi))
                        input = self.__getattr__('embed' + str(i + 1))(xi)
                elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                    if str(data['index'][0][i]) in data.keys():
                        xi = x[:, :, i].view(x.size(0), x.size(1))
                        xi = torch.LongTensor(np.array(xi))
                        xi = self.__getattr__('embed' + str(i + 1))(xi)
                    else:
                        xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    input = torch.cat([input.detach(), xi.detach()], dim=2)
                elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                    xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    input = torch.cat([input.detach(), xi.detach()], dim=2)
            r_out = self.rnn(input)
        elif self.input_size == -3:  # 全部拼接，活动CBOW和其他索引编码
            for i in range(len(data['index'][0])):
                if data['state'][0][i] == 0:  # 活动
                    xi = x[:, :, i].view(x.size(0), x.size(1))
                    if str(i) in data.keys():
                        xi = torch.LongTensor(np.array(xi))
                        input = self.__getattr__('embed' + str(i + 1))(xi)
                else:  # 其他
                    xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    input = torch.cat([input.detach(), xi.detach()], dim=2)
            r_out = self.rnn(input)
        elif self.input_size == 1:  # 仅活动，索引编码
            r_out = self.rnn(x[:, :, 0:1])
        elif self.input_size == 0:  # 仅活动，CBOW编码
            xi = torch.tensor(x[:, :, 0], dtype=torch.int64)
            xi = self.embed(xi)
            r_out = self.rnn(xi)
        else:  # 仅活动，One-hot编码
            ohx = np.eye(self.input_size)[torch.tensor(x[:, :, 0], dtype=torch.int64)]
            ohx = torch.tensor(ohx, dtype=torch.float32)
            if ohx.shape.__len__() == 1:
                ohx = ohx.view(1, 1, -1)
            r_out = self.rnn(ohx)
        # outs = self.out(r_out[0])  # F.relu()
        outs1 = self.out1(r_out[0])
        outs = self.out2(outs1)
        return outs.view(-1, x.size(1)), r_out[1]

class RNNM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data):
        super(RNNM, self).__init__()
        self.input_size = input_size
        einput_size = 0
        if input_size == -1:  # 活动与其他特征分开输入
            for i in range(len(data['index'][0])):
                if data['state'][0][i] == 0:  # 活动
                    input_size = 1
                    if '0' in data.keys():
                        self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                        input_size = self.embed1.embedding_dim
                        if type == 1:
                            output_size = input_size
                elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                    if str(data['index'][0][i]) in data.keys():
                        setattr(self, 'embed' + str(i + 1),
                                nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                        einput_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                    else:
                        einput_size += 1
                elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                    einput_size += 1
        elif input_size == -2:  # 前缀特征与其他特征分开输入
            for i in range(len(data['index'][0])):
                if data['state'][0][i] == 0:  # 活动
                    input_size = 1
                    if '0' in data.keys():
                        self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                        input_size = self.embed1.embedding_dim
                        if type == 1:
                            output_size = input_size
                elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                    if str(data['index'][0][i]) in data.keys():
                        setattr(self, 'embed' + str(i + 1),
                                nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                        if i in data['prefix'][0]:
                            input_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                        else:
                            einput_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                    else:
                        if i in data['prefix'][0]:
                            input_size += 1
                        else:
                            einput_size += 1
                elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                    if i in data['prefix'][0]:
                        input_size += 1
                    else:
                        einput_size += 1
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,  # 3.丢弃层
            batch_first=True
        )

        self.out = nn.Linear(einput_size, hidden_size)
        self.out1 = nn.Linear(2 * hidden_size, hidden_size)
        self.out2 = nn.Linear(hidden_size, 1)
        # 正则初始化
        nn.init.orthogonal(self.rnn.weight_ih_l0)
        nn.init.orthogonal(self.rnn.weight_hh_l0)
        # nn.init.zeros_(self.rnn.bias_ih_l0)
        # nn.init.zeros_(self.rnn.bias_hh_l0)

    def forward(self, x, data):
        if self.input_size == -1:  # 全部拼接，索引编码
            einput = []
            for i in range(len(data['index'][0])):
                if data['state'][0][i] == 0:  # 活动
                    xi = x[:, :, i].view(x.size(0), x.size(1))
                    if str(i) in data.keys():
                        xi = torch.LongTensor(np.array(xi))
                        input = self.__getattr__('embed' + str(i + 1))(xi)
                elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                    if str(data['index'][0][i]) in data.keys():
                        xi = x[:, :, i].view(x.size(0), x.size(1))
                        xi = torch.LongTensor(np.array(xi))
                        xi = self.__getattr__('embed' + str(i + 1))(xi)
                    else:
                        xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    if einput == []:
                        einput = xi
                    else:
                        einput = torch.cat([einput.detach(), xi.detach()], dim=2)
                elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                    xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    if einput == []:
                        einput = xi
                    else:
                        einput = torch.cat([einput.detach(), xi.detach()], dim=2)
            r_out = self.rnn(input)
            # einput = torch.cat([r_out[0].detach(), einput.detach()], dim=2)
        elif self.input_size == -2:  # 全部拼接，CBOW和向量编码
            einput = []
            for i in range(len(data['index'][0])):
                if data['state'][0][i] == 0:  # 活动
                    xi = x[:, :, i].view(x.size(0), x.size(1))
                    if str(i) in data.keys():
                        xi = torch.LongTensor(np.array(xi))
                        input = self.__getattr__('embed' + str(i + 1))(xi)
                elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                    if str(data['index'][0][i]) in data.keys():
                        xi = x[:, :, i].view(x.size(0), x.size(1))
                        xi = torch.LongTensor(np.array(xi))
                        xi = self.__getattr__('embed' + str(i + 1))(xi)
                    else:
                        xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    if i in data['prefix'][0]:
                        input = torch.cat([input.detach(), xi.detach()], dim=2)
                    else:
                        if einput == []:
                            einput = xi
                        else:
                            einput = torch.cat([einput.detach(), xi.detach()], dim=2)
                elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                    xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    if i in data['prefix'][0]:
                        input = torch.cat([input.detach(), xi.detach()], dim=2)
                    else:
                        if einput == []:
                            einput = xi
                        else:
                            einput = torch.cat([einput.detach(), xi.detach()], dim=2)
            r_out = self.rnn(input)
        outse = self.out(einput)
        einput = torch.cat([r_out[0].detach(), outse.detach()], dim=2)
        outs1 = self.out1(einput)
        outs = self.out2(outs1)
        return outs.view(-1, x.size(1)),einput

def testT(X_Test, Y_Test, model, type, data):
    eval_loss = 0
    loss_func = nn.L1Loss()
    pred_y = []
    true_y = []
    Metric = []
    abs_error = []
    represents = []
    for x, y in zip(X_Test, Y_Test):
        batch = x.shape[0]
        length = x.shape[1]
        PosEncode = torch.zeros(batch, length)
        for ii in range(batch):
            for jj in range(length):
                PosEncode[ii][jj] = jj
        prediction, represent = model(x, PosEncode, data)
        represents.append(represent)
        prediction = prediction.view(prediction.shape[0], prediction.shape[1])
        # 记录误差
        loss = loss_func(prediction, y)
        eval_loss += loss.item()
        for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):
            true_y.append(line1)
            pred_y.append(line2)
    Metric.append(mean_absolute_error(true_y, pred_y))
    return Metric, len(pred_y), represents

def trianT(Train, Test_X, Test_Y,epoch,type,mcl,method=None,data=None):
    save_path = '../Save/multiFea/'  # 当前目录下
    early_stopping = EarlyStopping(save_path)
    train_loss = 0
    if isinstance(method, str):
        n = Transformer(mcl, data)
    else:
        n = method
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    # 学习率下降：每过n个epoch，学习率乘以0.1
    scheduler = lr_scheduler.StepLR(optimizer, 100, 0.1)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        scheduler.step()
        for j, (x, y, l) in enumerate(Train):
            batch = x.shape[0]
            length = x.shape[1]
            PosEncode = torch.zeros(batch, length)
            for ii in range(batch):
                for jj in range(length):
                    PosEncode[ii][jj] = jj
            padding = torch.zeros(batch, length)
            for ii in range(batch):
                for jj in range(l[ii]):
                    padding[ii][jj] = 1
            output, _ = n(x, PosEncode, data)
            py = output.view(output.shape[0], output.shape[1])
            optimizer.zero_grad()
            loss = loss_func(py, y)
            loss.backward()
            optimizer.step()
            # 记录误差
            train_loss += loss.item()
    Metric, count, _ = testT(Test_X, Test_Y, n, type, data)
        # 早停止
        # early_stopping(eval_loss, n)
        # 达到早停止条件时，early_stop会被置为True
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break  # 跳出迭代，结束训练
    # print(i, Metric, count)
    return n, Metric, count

class Transformer(nn.Module):
    def __init__(self, pos_dim, data, hidden=128, nhead=8, num_layers=3, dim_feedforward=64, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=True, norm_first=True) -> None:
        super(Transformer, self).__init__()
        self.dim = 0
        for i in range(len(data['index'][0])):
            if data['state'][0][i] < 3:  # 分类
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    self.dim += torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    self.dim += 1
            else:  # 数值
                self.dim += 1
        m = self.dim % nhead
        if m > 0:
            self.dim = self.dim + nhead - m
        self.pos = nn.Embedding(pos_dim, self.dim)
        encoder_layer = nn.TransformerEncoderLayer(self.dim, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first)
        encoder_norm = nn.LayerNorm(self.dim, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.pooling = nn.AdaptiveAvgPool1d(self.dim)
        self.fnn1 = nn.Linear(self.dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fnn2 = nn.Linear(hidden, 1)

    def forward(self, x, PosEncode, data):
        for i in range(len(data['index'][0])):
            if data['state'][0][i] == 0:  # 活动
                xc = x[:, :, i].view(x.size(0), x.size(1))
                if str(i) in data.keys():
                    xc = torch.LongTensor(np.array(xc))
                    inputc = self.__getattr__('embed' + str(i + 1))(xc)
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                if str(data['index'][0][i]) in data.keys():
                    xc = x[:, :, i].view(x.size(0), x.size(1))
                    xc = torch.LongTensor(np.array(xc))
                    xc = self.__getattr__('embed' + str(i + 1))(xc)
                else:
                    xc = x[:, :, i].view(x.size(0), x.size(1), 1)
                inputc = torch.cat([inputc.detach(), xc.detach()], dim=2)
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                xc = x[:, :, i].view(x.size(0), x.size(1), 1)
                inputc = torch.cat([inputc.detach(), xc.detach()], dim=2)
        m = inputc.size(2)
        if m != self.dim:
            xc = torch.zeros(inputc.size(0), inputc.size(1), self.dim - m)
            inputc = torch.cat([inputc.detach(), xc.detach()], dim=2)
        pos = self.pos(PosEncode.long())
        memory = self.encoder(inputc + pos)
        out1 = self.pooling(memory)
        out2 = self.fnn1(out1)
        output = self.dropout(out2)
        output = self.fnn2(output)  # F.relu()
        return output, memory[:, -1, :]

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
