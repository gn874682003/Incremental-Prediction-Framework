import copy
import math
import random

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from typing import Optional, Any, Union, Callable
from torch import Tensor

# Hyper Parameters
INPUT_SIZE = 5          # rnn input size / image width
LR = 0.001               # learning rate

def get_att_dis(target, behaviored):#计算余弦相似度
    attention_distribution = []
    result = []
    for j in range(target.shape[1]):
        for i in range(behaviored.shape[0]):
            attention_score = torch.cosine_similarity(target[0,j,:].view(1, -1), torch.FloatTensor(behaviored[i]).view(1, -1))  # 计算每一个元素与给定元素的余弦相似度
            attention_distribution.append(attention_score)
        result.append(int(torch.argmax(torch.Tensor(attention_distribution))))
        attention_distribution = []
    return result

def viewResult(X_Test, Y_Test, rnn, type, data):
    for x, y in zip(X_Test, Y_Test):
        output, prediction = rnn(x, data)
        print(x)
        for j in range(prediction.size(1)):
            yi = 0
            for line, label in zip(output, data['name']):
                plt.scatter(line.view(line.size(1))[j].detach().numpy(), yi)
                plt.annotate(label, xy=(line.view(line.size(1))[j].detach().numpy(), yi), xytext=(5, 2),
                             textcoords='offset points', ha='right', va='bottom')
                yi += 1
            plt.vlines(y[0][j],0,yi-1)
            y_major_locator = MultipleLocator(1)
            ax = plt.gca()
            ax.yaxis.set_major_locator(y_major_locator)
            plt.xlabel('Time(day)')
            plt.ylabel('Feature Number')
            plt.show()

def viewResultD(X_Test, Y_Test, rnn, type, data, ConvertReflact,attribute):
    for x, y in zip(X_Test, Y_Test):
        output, prediction = rnn(x, data)
        print(x)
        for j in range(prediction.size(1)):
            yi = 0
            for line,i in zip(output,range(len(output))):
                if data['state'][0][i]<3:
                    print(x[0][0][i])
                    a = int(x[0][0][i])
                    label = ConvertReflact[attribute.index(data['index'][0][i]+3)][int(x[0][0][i])]
                else:
                    label = x[0][0][i]
                plt.scatter(line.view(line.size(1))[j].detach().numpy(), yi)
                plt.annotate(label, xy=(line.view(line.size(1))[j].detach().numpy(), yi), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
                yi += 1
            plt.vlines(y[0], 0, yi-1)
            y_major_locator = MultipleLocator(1)
            ax = plt.gca()
            ax.yaxis.set_major_locator(y_major_locator)
            plt.yticks(range(len(output)), data['name'])
            plt.xlabel('Time(day)')
            plt.ylabel('Feature Number')
            plt.show()

def TestMetric(X_Test, Y_Test, rnn, type, data, flag, input_size):
    pred_y = []
    true_y = []
    if type == 1:
        Metric = []
        for i in range(input_size):#len(data['index'][0])
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = get_att_dis(prediction, data['0'])
                for line1, line2 in zip(y.numpy().tolist()[0], prediction):
                    true_y.append(line1)
                    pred_y.append(line2)
            Metric.append(accuracy_score(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    else:
        Metric = []
        for i in range(input_size):
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = prediction.view(prediction.size(0),prediction.size(1))
                for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):
                    true_y.append(line1)
                    pred_y.append(line2)#[-1]
            Metric.append(mean_absolute_error(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    # if type == 1:
    #     for x, y in zip(X_Test, Y_Test):
    #         prediction = rnn(x)  # rnn output
    #         if prediction.size()[-1] == 1:
    #             true_y.append(np.round(prediction.detach().numpy().tolist()[0]))
    #             pred_y.append(y.numpy().tolist()[0])
    #         else:
    #             for line1, line2 in zip(y.numpy().tolist()[0], np.round(prediction.detach().numpy()).tolist()[0]):
    #                 true_y.append(line1)
    #                 pred_y.append(line2)
    #     Metric = accuracy_score(true_y, pred_y)
    # else:
    #     for x, y in zip(X_Test, Y_Test):
    #         prediction = rnn(x)  # rnn output
    #         for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):
    #             true_y.append(line1)
    #             pred_y.append(line2)
    #     Metric = mean_absolute_error(true_y, pred_y)
    return Metric

def TestMetricB(X_Test, Y_Test, rnn, type):
    pred_y = []
    true_y = []
    if type == 1:
        for x, y in zip(X_Test, Y_Test):
            prediction = rnn(x)  # rnn output
            for line1, line2 in zip(y.numpy().tolist()[0], np.round(prediction.detach().numpy()).tolist()[0]):#[0]
                true_y.append(line1)
                pred_y.append(line2)
        Metric = accuracy_score(true_y, pred_y)
    else:
        for x, y in zip(X_Test, Y_Test):
            prediction = rnn(x)
            for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):#[0]
                true_y.append(line1)
                pred_y.append(line2)#[-1]
        Metric = mean_absolute_error(true_y, pred_y)
    return Metric

def TestMetricD(X_Test, Y_Test, rnn, type, data, flag, input_size):
    pred_y = []
    true_y = []
    if type == 1:
        Metric = []
        for i in range(input_size):#len(data['index'][0])
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = get_att_dis(prediction, data['0'])
                true_y.append(y.numpy().tolist()[0])
                pred_y.append(prediction)
            Metric.append(accuracy_score(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    else:
        Metric = []
        for i in range(input_size):
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = prediction.view(prediction.size(0),prediction.size(1))
                true_y.append(y.numpy().tolist()[0])
                pred_y.append(prediction.detach().numpy().tolist()[0][-1])#
            Metric.append(mean_absolute_error(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    return Metric

def TestMetricN(X_Test, Y_Test, rnn, type, data, flag, input_size):
    pred_y = []
    true_y = []
    if type == 1:
        Metric = []
        for i in range(input_size):#len(data['index'][0])
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = get_att_dis(prediction, data['0'])
                true_y.append(y.numpy().tolist()[0])
                pred_y.append(prediction)
            Metric.append(accuracy_score(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    else:
        Metric = []
        for i in range(input_size):
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output = rnn(x, data)
                true_y.append(y.numpy().tolist()[0])
                pred_y.append(output.detach().numpy().tolist()[0][-1])#
            Metric.append(mean_absolute_error(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    return Metric

def LSTMNest(Train_X,Train_Y,Test_X,Test_Y,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    n = NNN(input_size, hidden_size, num_layers, data, type)
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for x, y in zip(Train_X, Train_Y):
            output = n(x, data)
            optimizer.zero_grad()
            if type == 1:
                temp = []
                for line in y:
                    temp.append(data['0'][int(line)])
                ty = torch.FloatTensor(temp)
                loss = loss_func(output, ty)
            else:
                loss = loss_func(output[:,0], y)
            loss.backward()
            optimizer.step()
        if (i+1) % 10 == 0:
            Metric = TestMetricN(Test_X, Test_Y, n, type, data,1,input_size)
        else:
            Metric = TestMetricN(Test_X, Test_Y, n, type, data,0,input_size)
        print(Metric)
        if i == 0:
            BestResult = Metric[-1]
            BestAll = Metric[-1]
            BestModel = copy.deepcopy(n)
            BestModelAll = copy.deepcopy(n)
        elif type == 1:
            if Metric[-1] > BestResult:
                BestResult = Metric[-1]
                BestModel = copy.deepcopy(n)
            if max(Metric) > BestAll:
                BestAll = max(Metric)
                BestModelAll = copy.deepcopy(n)
        elif type == 2:
            if Metric[-1] < BestResult:
                BestResult = Metric[-1]
                BestModel = copy.deepcopy(n)
            if min(Metric) < BestAll:
                BestAll = min(Metric)
                BestModelAll = copy.deepcopy(n)
    return BestResult, BestModel, BestAll, BestModelAll

def LSTMDiff(Train_X,Train_Y,Test_X,Test_Y,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    if method == 'inn':
        n = INN(input_size, hidden_size, num_layers, data, type)
    elif method == 'inn2':
        n = INN2(input_size, hidden_size, num_layers, data, type)
    print(n)
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for k in range(input_size):
            for x, y in zip(Train_X, Train_Y):
                output, prediction = n(x, data)
                optimizer.zero_grad()
                if type == 1:
                    temp = []
                    for line in y:
                        temp.append(data['0'][int(line)])
                    ty = torch.FloatTensor(temp)
                    loss = loss_func(prediction, ty)
                else:
                    loss = loss_func(output[k][:,-1,0], y)
                loss.backward()
                optimizer.step()
        if (i+1) % 10 == 0:
            Metric = TestMetricD(Test_X, Test_Y, n, type, data,1,input_size)
        else:
            Metric = TestMetricD(Test_X, Test_Y, n, type, data,0,input_size)
        print(Metric)
        if i == 0:
            BestResult = Metric[-1]
            BestAll = Metric[-1]
            BestModel = copy.deepcopy(n)
            BestModelAll = copy.deepcopy(n)
        elif type == 1:
            if Metric[-1] > BestResult:
                BestResult = Metric[-1]
                BestModel = copy.deepcopy(n)
            if max(Metric) > BestAll:
                BestAll = max(Metric)
                BestModelAll = copy.deepcopy(n)
        elif type == 2:
            if Metric[-1] < BestResult:
                BestResult = Metric[-1]
                BestModel = copy.deepcopy(n)
            if min(Metric) < BestAll:
                BestAll = min(Metric)
                BestModelAll = copy.deepcopy(n)
    return BestResult, BestModel, BestAll, BestModelAll

def LSTMNew(Train,Test_X,Test_Y,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    if isinstance(method, str):
        if method == 'inn':
            n = INN(input_size, hidden_size, num_layers, data, type)
        elif method == 'inn2':
            n = INN2(input_size, hidden_size, num_layers, data, type)
    else:
        n = method
    print(n)
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for k in range(input_size):#6
            for j, (x, y, l) in enumerate(Train):
                ty = nn.utils.rnn.pack_padded_sequence(y, l, batch_first=True)
                output, prediction = n(x, data)
                py = nn.utils.rnn.pack_padded_sequence(output[k], l, batch_first=True)
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
                optimizer.step()
        if (i+1) % 10 == 0:
            Metric = TestMetric(Test_X, Test_Y, n, type, data,1,input_size)
            # print(Metric)
        else:
            Metric = TestMetric(Test_X, Test_Y, n, type, data,0,input_size)
        print(Metric)
        if i == 0:
            BestResult = Metric[-1]
            BestAll = Metric[-1]
            BestModel = copy.deepcopy(n)
            BestModelAll = copy.deepcopy(n)
        elif type == 1 and Metric[-1] > BestResult:
            BestResult = Metric[-1]
            BestModel = copy.deepcopy(n)
        elif type == 2 and Metric[-1] < BestResult:
            BestResult = Metric[-1]
            BestModel = copy.deepcopy(n)
        elif type == 1 and max(Metric) > BestAll:
            BestAll = max(Metric)
            BestModelAll = copy.deepcopy(n)
        elif type == 2 and min(Metric) < BestAll:
            BestAll = min(Metric)
            BestModelAll = copy.deepcopy(n)
    return BestResult, BestModel, BestAll, BestModelAll

def LSTM(X,Y,X_Test,Y_Test, Train_L, Test_L,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    if method == 'inn':
        n = INN(input_size, hidden_size, num_layers, data, type)
    elif method == 'inn2':
        n = INN2(input_size, hidden_size, num_layers, data, type)
    print(n)
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)  # optimize all cnn parameters
    # loss_func = nn.MSELoss()  # the target label is not one-hotted.L1Loss()
    loss_func = nn.L1Loss()  # the target label is not one-hotted.L1Loss()
    for i in range(epoch):
        for k in range(input_size):
            for x, y, l in zip(X, Y, Train_L):
                if type == 1:
                    ty, temp = [], []
                    for line1 in y:
                        for line2 in line1:
                            temp.append(data['0'][int(line2)])
                        ty.append(temp)
                        temp = []
                    y = torch.FloatTensor(ty)
                else:
                    y = y.view(y.size(0), y.size(1), 1)
                # for k in range(start,input_size):
                output, prediction = n(x, data, l)#
                optimizer.zero_grad()  # clear gradients for this training step
                # for preI in output:#[0:j]
                    # preI = output[j]
                loss = loss_func(output[k], y)  # cross entropy loss[:,-1]
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
        if (i+1) % 10 == 0:
            Metric = TestMetric(X_Test, Y_Test, n, type, data,1)
        else:
            Metric = TestMetric(X_Test, Y_Test, n, type, data,0)
        print(Metric)
        if i == 0:
            BestResult = Metric[-1]
            BestModel = n
        elif type == 1 and Metric[-1] > BestResult:
            BestResult = Metric[-1]
            BestModel = n
        elif type == 2 and Metric[-1] < BestResult:
            BestResult = Metric[-1]
            BestModel = n
    return BestResult, BestModel

def baseLine(X,Y,X_Test,Y_Test,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None):
    if method == 'fnn':
        n = FNN(input_size,hidden_size)
    elif method == 'rnn':
        n = RNN(input_size, hidden_size, num_layers)
    elif method == 'OLSTM':#倪老师
        embedding = nn.Embedding(15, 4, padding_idx=14)
        n = OLSTM(4, 5, 1, embedding=embedding)
    print(n)
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)  # optimize all cnn parameters
    # loss_func = nn.MSELoss()  # the target label is not one-hotted.L1Loss()
    loss_func = nn.L1Loss()  # the target label is not one-hotted.L1Loss()
    for i in range(epoch):
        for x, y in zip(X, Y):
            prediction = n(x)#
            optimizer.zero_grad()  # clear gradients for this training step
            loss = loss_func(prediction, y)  # cross entropy loss[:,-1]
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
        # if i % 49 == 0:
        Metric = TestMetricB(X_Test, Y_Test, n, type)
        print(Metric)#
        if i == 0:
            BestResult = Metric
            BestModel = n
        elif type == 1 and Metric > BestResult:
            BestResult = Metric
            BestModel = n
        elif type == 2 and Metric < BestResult:
            BestResult = Metric
            BestModel = n
    return BestResult, BestModel

def trianMT(Train,Test_X, Test_Y ,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    if method == 'MultiTaskNN':
        n = MultiNN(input_size, hidden_size, num_layers, data, type)
    elif method == 'MultiTaskNN3':
        n = MultiNN3(input_size, hidden_size, num_layers, data, type)
    else:
        n = method
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for k in range(input_size):
            for j, (x, y1, y2, y3, l) in enumerate(Train):

                output, prediction = n(x, data)
                ty1 = nn.utils.rnn.pack_padded_sequence(y1, l, batch_first=True)
                ty2 = nn.utils.rnn.pack_padded_sequence(y2, l, batch_first=True)
                ty3 = nn.utils.rnn.pack_padded_sequence(y3, l, batch_first=True)
                py1 = nn.utils.rnn.pack_padded_sequence(prediction[0], l, batch_first=True)
                py2 = nn.utils.rnn.pack_padded_sequence(prediction[1], l, batch_first=True)
                py3 = nn.utils.rnn.pack_padded_sequence(prediction[2], l, batch_first=True)
                optimizer.zero_grad()
                temp = []
                for line in ty2.data:
                    temp.append(data['0'][int(line)])
                ty2 = torch.FloatTensor(temp)
                loss = loss_func(py1.data.view(py1.data.size(0)), ty1.data) \
                       + loss_func(py2.data, ty2.data) \
                       + loss_func(py3.data.view(py3.data.size(0)), ty3.data)
                loss.backward()
                optimizer.step()
    Metric = testMT(Test_X, Test_Y, n, type, data, 1, input_size)
    return n, Metric

def trian(Train,Test_X, Test_Y ,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    if isinstance(method, str):
        if method == 'inn':
            n = INN(input_size, hidden_size, num_layers, data, type)
        elif method == 'inn2':
            n = INN2(input_size, hidden_size, num_layers, data, type)
        elif method == 'DNN1':
            n = DNN1(input_size, hidden_size, num_layers, data, type)
            input_size = 1
        elif method == 'DNN2':
            n = DNN2(input_size, hidden_size, num_layers, data, type)
            input_size = 1
        elif method == 'DNN3':
            n = DNN3(input_size, hidden_size, num_layers, data, type)
            input_size = 1
    else:
        n = method
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for k in range(input_size):
            for j, (x, y, l) in enumerate(Train):
                ty = nn.utils.rnn.pack_padded_sequence(y, l, batch_first=True)
                output, prediction = n(x, data)
                py = nn.utils.rnn.pack_padded_sequence(output[k], l, batch_first=True)
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
                optimizer.step()
    Metric = test(Test_X, Test_Y, n, type, data, 1, input_size)
    return n, Metric

def trianD(Train_X,Train_Y,Test_X,Test_Y,epoch,type,input_size=2,hidden_size=32,num_layers=1,method=None,data=None):
    if isinstance(method, str):
        if method == 'inn':
            n = INN(input_size, hidden_size, num_layers, data, type)
        elif method == 'inn2':
            n = INN2(input_size, hidden_size, num_layers, data, type)
    else:
        n = method
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for k in range(input_size):
            for x, y in zip(Train_X, Train_Y):
                output, prediction = n(x, data)
                optimizer.zero_grad()
                if type == 1:
                    temp = []
                    for line in y:
                        temp.append(data['0'][int(line)])
                    ty = torch.FloatTensor(temp)
                    loss = loss_func(prediction, ty)
                else:
                    loss = loss_func(output[k][:,-1,0], y)
                loss.backward()
                optimizer.step()
    Metric, abs_error = test(Test_X, Test_Y, n, type, data, 1, input_size)
    return n, Metric, abs_error

def trianT(Train, Test_X, Test_Y,epoch,type,dim,hidden_size,num_layers,nhead,method=None,data=None):
    if isinstance(method, str):
        n = Transformer(dim, nhead, num_layers, hidden_size)
    else:
        n = method
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    if dim == data['0'].shape[1]:
        emb = data['0']
    else:
        emb = nn.Embedding(data['0'].shape[0], dim).weight.data.numpy()
    for i in range(epoch):
        for j, (x, y, l) in enumerate(Train):
            batch = x.shape[0]
            length = x.shape[1]
            InputEmb = torch.zeros(batch, length, dim)
            for ii in range(batch):
                for jj in range(length):
                    InputEmb[ii][jj] = torch.from_numpy(emb[int(x[ii][jj])])
            PosEncode = torch.zeros(batch, length, dim)
            for ii in range(batch):
                for jj in range(length):
                    for kk in range(dim):
                        if jj % 2 == 0:
                            PosEncode[ii][jj][kk] = math.sin(jj/math.exp(-2*(kk+1)/(dim-1)*math.log(10000)))
                        else:
                            PosEncode[ii][jj][kk] = math.cos(jj/math.exp(-2*(kk+1)/(dim-1)*math.log(10000)))
            # mask = torch.ones(length, length)
            # for ii in range(length-1):#
            #     for jj in range(length-ii-1):#ii+1
            #         mask[ii][jj] = 0
            mask = n.generate_square_subsequent_mask(length)
            # mask = mask.repeat(batch*nhead, 1, 1)
            padding = torch.zeros(batch, length)
            for ii in range(batch):
                for jj in range(l[ii]):
                    padding[ii][jj] = 1
            output = n(InputEmb+PosEncode, mask, padding)#
            ty = nn.utils.rnn.pack_padded_sequence(y, l, batch_first=True)
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
            optimizer.step()
    Metric, abs_error = testT(Test_X, Test_Y, n, type, data, 1, dim)
    return n, Metric, abs_error

def trianPT(Train, Test_X, Test_Y,epoch,type,dim,mcl,method=None,data=None):
    if isinstance(method, str):
        emb = nn.Embedding(data['0'].shape[0], dim)
        pos = nn.Embedding(mcl, dim)
        n = ProTransformer(emb, pos)
    else:
        n = method
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
        for j, (x, y, l) in enumerate(Train):
            batch = x.shape[0]
            length = x.shape[1]
            x1 = torch.zeros(batch, length)
            x2 = torch.zeros(batch, length, 2)
            PosEncode = torch.zeros(batch, length)
            for ii in range(batch):
                for jj in range(length):
                    PosEncode[ii][jj] = jj
                    x1[ii][jj] = x[ii][jj][0]
                    x2[ii][jj] = x[ii][jj][1:2]
            mask = n.generate_square_subsequent_mask(length)
            padding = torch.zeros(batch, length)
            for ii in range(batch):
                for jj in range(l[ii]):
                    padding[ii][jj] = 1
            output = n(x1, x2, PosEncode)
            py = output.view(output.shape[0], output.shape[1])
            optimizer.zero_grad()
            loss = loss_func(py, y)
            loss.backward()
            optimizer.step()
    Metric, abs_error = testPT(Test_X, Test_Y, n, type, data, 1)
    return n, Metric, abs_error

def trianPT1(Train, Test_X, Test_Y,epoch,type,input_size,mcl,method=None,data=None):
    if isinstance(method, str):
        n = ProTransformer1(input_size, mcl, data)
    else:
        n = method
    optimizer = torch.optim.Adam(n.parameters(), lr=LR)
    loss_func = nn.L1Loss()
    for i in range(epoch):
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
            output = n(x, PosEncode, data)
            py = output.view(output.shape[0], output.shape[1])
            optimizer.zero_grad()
            loss = loss_func(py, y)
            loss.backward()
            optimizer.step()
    Metric, abs_error = testPT1(Test_X, Test_Y, n, type, data, 1)
    return n, Metric, abs_error

def testMT(X_Test, Y_Test, rnn, type, data, flag, input_size):
    pred_y1 = []
    true_y1 = []
    pred_y2 = []
    true_y2 = []
    pred_y3 = []
    true_y3 = []
    Metric = []
    for x, y1, y2, y3 in zip(X_Test, Y_Test[0], Y_Test[1], Y_Test[2]):
        output, prediction = rnn(x, data)
        # 剩余时间测试结果
        prediction[0] = prediction[0].view(prediction[0].size(0), prediction[0].size(1))
        for line1, line2 in zip(y1.numpy().tolist()[0], prediction[0].detach().numpy().tolist()[0]):
            true_y1.append(line1)
            pred_y1.append(line2)
        # 下一事件测试结果
        prediction[1] = get_att_dis(prediction[1], data['0'])
        for line1, line2 in zip(y2.numpy().tolist()[0], prediction[1]):
            true_y2.append(line1)
            pred_y2.append(line2)
        # 持续时间测试结果
        prediction[2] = prediction[2].view(prediction[2].size(0), prediction[2].size(1))
        for line1, line2 in zip(y3.numpy().tolist()[0], prediction[2].detach().numpy().tolist()[0]):
            true_y3.append(line1)
            pred_y3.append(line2)
    Metric.append(mean_absolute_error(true_y1, pred_y1))
    Metric.append(accuracy_score(true_y2, pred_y2))
    Metric.append(mean_absolute_error(true_y3, pred_y3))
    return Metric

def test(X_Test, Y_Test, rnn, type, data, flag, input_size):
    pred_y = []
    true_y = []
    if type == 1:
        Metric = []
        for i in range(input_size):#len(data['index'][0])
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = get_att_dis(prediction, data['0'])
                for line1, line2 in zip(y.numpy().tolist()[0], prediction):
                    true_y.append(line1)
                    pred_y.append(line2)
            Metric.append(accuracy_score(true_y, pred_y))
            if flag == 0:
                break
            pred_y = []
            true_y = []
    else:
        Metric = []
        abs_error = []
        for i in range(input_size):
            if flag == 0:
                i = -1
            for x, y in zip(X_Test, Y_Test):
                output, prediction = rnn(x, data)
                prediction = output[i]
                prediction = prediction.view(prediction.size(0),prediction.size(1))
                for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):
                    true_y.append(line1)
                    pred_y.append(line2)#[-1]
            Metric.append(mean_absolute_error(true_y, pred_y))
            abs_error.append(abs(np.array(true_y) - np.array(pred_y)).tolist())
            if flag == 0:
                break
            pred_y = []
            true_y = []
    return Metric

def testT(X_Test, Y_Test, model, type, data, flag, dim):
    pred_y = []
    true_y = []
    if type == 1:
        Metric = []
        if flag == 0:
            i = -1
        for x, y in zip(X_Test, Y_Test):
            output, prediction = model(x, data)
            prediction = output[i]
            prediction = get_att_dis(prediction, data['0'])
            for line1, line2 in zip(y.numpy().tolist()[0], prediction):
                true_y.append(line1)
                pred_y.append(line2)
        Metric.append(accuracy_score(true_y, pred_y))
    else:
        Metric = []
        abs_error = []
        if dim == data['0'].shape[1]:
            emb = data['0']
        else:
            emb = nn.Embedding(data['0'].shape[0], dim).weight.data.numpy()
        for x, y in zip(X_Test, Y_Test):
            batch = x.shape[0]
            length = x.shape[1]
            InputEmb = torch.zeros(batch, length, dim)
            for ii in range(batch):
                for jj in range(length):
                    InputEmb[ii][jj] = torch.from_numpy(emb[int(x[ii][jj])])
            PosEncode = torch.zeros(batch, length, dim)
            for ii in range(batch):
                for jj in range(length):
                    for kk in range(dim):
                        if jj % 2 == 0:
                            PosEncode[ii][jj][kk] = math.sin(jj / math.exp(-2 * (kk + 1) / (dim - 1) * math.log(10000)))
                        else:
                            PosEncode[ii][jj][kk] = math.cos(jj / math.exp(-2 * (kk + 1) / (dim - 1) * math.log(10000)))
            prediction = model(InputEmb + PosEncode)
            for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):
                true_y.append(line1)
                pred_y.append(line2)
        Metric.append(mean_absolute_error(true_y, pred_y))
        abs_error.append(abs(np.array(true_y) - np.array(pred_y)).tolist())
    return Metric, abs_error

def testPT(X_Test, Y_Test, model, type, data, flag):
    pred_y = []
    true_y = []
    Metric = []
    abs_error = []
    for x, y in zip(X_Test, Y_Test):
        batch = x.shape[0]
        length = x.shape[1]
        x1 = torch.zeros(batch, length)
        x2 = torch.zeros(batch, length, 2)
        PosEncode = torch.zeros(batch, length)
        for ii in range(batch):
            for jj in range(length):
                PosEncode[ii][jj] = jj
                x1[ii][jj] = x[ii][jj][0]
                x2[ii][jj] = x[ii][jj][1:2]
        prediction = model(x1, x2, PosEncode)
        prediction = prediction.view(prediction.shape[0], prediction.shape[1])
        for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):
            true_y.append(line1)
            pred_y.append(line2)
    Metric.append(mean_absolute_error(true_y, pred_y))
    abs_error.append(abs(np.array(true_y) - np.array(pred_y)).tolist())
    return Metric, abs_error

def testPT1(X_Test, Y_Test, model, type, data, flag):
    pred_y = []
    true_y = []
    Metric = []
    abs_error = []
    for x, y in zip(X_Test, Y_Test):
        batch = x.shape[0]
        length = x.shape[1]
        PosEncode = torch.zeros(batch, length)
        for ii in range(batch):
            for jj in range(length):
                PosEncode[ii][jj] = jj
        prediction = model(x, PosEncode, data)
        prediction = prediction.view(prediction.shape[0], prediction.shape[1])
        for line1, line2 in zip(y.numpy().tolist()[0], prediction.detach().numpy().tolist()[0]):
            true_y.append(line1)
            pred_y.append(line2)
    Metric.append(mean_absolute_error(true_y, pred_y))
    abs_error.append(abs(np.array(true_y) - np.array(pred_y)).tolist())
    return Metric, abs_error
#特征按时序输入至循环神经网络中
class NNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data,type):
        super(NNN, self).__init__()
        self.rnn1 = nn.LSTM(1, 8, num_layers, batch_first=True)
        self.rnn2 = nn.LSTM(8, 128, num_layers, batch_first=True)
        self.fnn1 = nn.Linear(128, 64)
        self.fnn2 = nn.Linear(64, 1)

    def forward(self, x, data):
        nn_out1 = np.zeros([x.size(0),x.size(1),8])
        for i in range(x.size(0)):
            xi = self.rnn1(x[i,:,:].view(x.size(1),x.size(2),1))
            nn_out1[i] = xi[1][0].detach().numpy()
        nn_out2 = self.rnn2(torch.FloatTensor(nn_out1))
        nn_out3 = self.fnn1(nn_out2[0][:,-1,:])
        output = self.fnn2(nn_out3)
        return output

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,  # rnn hidden unit
            num_layers=num_layers,  # number of rnn layer
            batch_first=True,
        )
        self.out = nn.Linear(hidden_size, 1)
        self.out1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.out2 = nn.Linear(int(hidden_size/2), 1)

    def forward(self, x):
        r_out = self.rnn(x)
        outs = self.out(F.relu(r_out[0]))
        # outs1 = self.out1(F.relu(r_out[0]))
        # outs = self.out2(F.relu(outs1))
        return outs.view(-1, x.size(1))#[:,-1]
#可解释分层模型（特征全部当做动态处理）
class INN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data,type):
        super(INN, self).__init__()
        self.input_size = input_size
        if type == 2:
            output_size = 1
        for i in range(input_size):#6
            if data['state'][0][i] == 0: # 活动
                in_size = 1
                if '0' in data.keys():
                    self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                    in_size = self.embed1.embedding_dim
                    if type == 1:
                        output_size = in_size
                self.rnn1 = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
                self.fnn1 = nn.Linear(hidden_size, int(hidden_size / 2))
                self.out1 = nn.Linear(int(hidden_size / 2), output_size)
                in_size = output_size
            # elif data['state'][0][i] == 1: #静态分类
            #     if str(i) in data.keys():
            #         setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(i)])))
            #         in_size += torch.tensor(data[str(i)]).size(1)
            #     else:
            #         in_size += 1
            #     setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size, in_size*2))
            #     # setattr(self, 'fnnt' + str(i + 1), nn.Linear(in_size*4, in_size*2))
            #     setattr(self, 'out' + str(i + 1), nn.Linear(in_size*2, output_size))
            #     in_size = output_size
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1: #动态分类
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    in_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    in_size += 1
                    setattr(self, 'fnnf' + str(i + 1), nn.Linear(in_size, 8))
                    in_size = 8
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=in_size*2,
                    num_layers=num_layers, batch_first=True))
                setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size*2, in_size))
                setattr(self, 'out' + str(i + 1), nn.Linear(in_size, output_size))
                in_size = output_size
            # elif data['state'][0][i] == 3: #静态数值
            #     in_size += 1
            #     setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size, in_size*4))
            #     setattr(self, 'fnnt' + str(i + 1), nn.Linear(in_size*4, in_size * 2))
            #     setattr(self, 'out' + str(i + 1), nn.Linear(in_size*2, output_size))
            #     in_size = output_size
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3: #动态数值
                in_size += 1
                setattr(self, 'fnnf' + str(i + 1), nn.Linear(in_size, 8))
                in_size = 8
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=in_size*2,
                    num_layers=num_layers, batch_first=True))
                setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size*2, in_size))
                setattr(self, 'out' + str(i + 1), nn.Linear(in_size, output_size))
                in_size = output_size

    def forward(self, x, data):
        output = []
        for i in range(self.input_size):#6:#len(data['state'][0])
            if data['state'][0][i] == 0: # 活动
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(i) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(xi)
                nn_out2 = self.__getattr__('fnn' + str(i + 1))(nn_out1[0])
                nn_out = self.__getattr__('out' + str(i + 1))(nn_out2)
                output.append(nn_out)
            # elif data['state'][0][i] == 1:  # 静态分类
            #     xi = x[:, :, i].view(x.size(0), x.size(1))
            #     if str(i) in data.keys():
            #         xi = torch.LongTensor(np.array(xi))
            #         xi = self.__getattr__('embed' + str(i + 1))(xi)
            #         x2 = torch.cat([nn_out.detach(), xi], dim=2)
            #     else:
            #         x2 = torch.cat([nn_out.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
            #     nn_out1 = self.__getattr__('fnn' + str(i + 1))(x2)
            #     # nn_out2 = self.__getattr__('fnnt' + str(i + 1))(nn_out1)
            #     nn_out = self.__getattr__('out' + str(i + 1))(nn_out1)
            #     output.append(nn_out)
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 动态分类
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(data['index'][0][i]) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                    x2 = torch.cat([nn_out.detach(), xi], dim=2)
                else:
                    x2 = torch.cat([nn_out.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                    x2 = self.__getattr__('fnnf' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(x2)
                nn_out2 = self.__getattr__('fnn' + str(i + 1))(nn_out1[0])
                nn_out = self.__getattr__('out' + str(i + 1))(nn_out2)
                output.append(nn_out)
            # elif data['state'][0][i] == 3:  # 静态数值
            #     xi = x[:, :, i].view(x.size(0), x.size(1))
            #     x2 = torch.cat([nn_out.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
            #     nn_out1 = self.__getattr__('fnn' + str(i + 1))(x2)
            #     nn_out2 = self.__getattr__('fnnt' + str(i + 1))(nn_out1)
            #     nn_out = self.__getattr__('out' + str(i + 1))(nn_out1)
            #     output.append(nn_out)
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 动态数值
                xi = x[:, :, i].view(x.size(0), x.size(1))
                x2 = torch.cat([nn_out.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                x2 = self.__getattr__('fnnf' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(x2)
                nn_out2 = self.__getattr__('fnn' + str(i + 1))(nn_out1[0])
                nn_out = self.__getattr__('out' + str(i + 1))(nn_out2)
                output.append(nn_out)
        return output, nn_out#.view(x.size(0), x.size(1))
#可解释分层模型（特征动静态分别处理）
class INN2(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data,type):
        super(INN2, self).__init__()
        if type == 2:
            output_size = 1
        for i in range(input_size):
            if data['state'][0][i] == 0: # 活动
                in_size = 1
                if '0' in data.keys():
                    self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                    in_size = self.embed1.embedding_dim
                    if type == 1:
                        output_size = in_size
                self.rnn1 = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
                self.fnn1 = nn.Linear(hidden_size, int(hidden_size / 2))
                self.out1 = nn.Linear(int(hidden_size / 2), output_size)
                in_size = int(hidden_size / 2)
            elif data['state'][0][i] == 1: #静态分类
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    in_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    in_size += 1
                setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size, hidden_size))
                setattr(self, 'fnns' + str(i + 1), nn.Linear(hidden_size, int(hidden_size / 2)))
                setattr(self, 'out' + str(i + 1), nn.Linear(int(hidden_size / 2), output_size))
                in_size = int(hidden_size / 2)
            elif data['state'][0][i] == 2: #动态分类 or data['state'][0][i] == 1
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    in_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    in_size += 1
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=hidden_size,
                    num_layers=num_layers, batch_first=True))
                setattr(self, 'fnn' + str(i + 1), nn.Linear(hidden_size, int(hidden_size/2)))
                setattr(self, 'out' + str(i + 1), nn.Linear(int(hidden_size/2), output_size))
                in_size = int(hidden_size / 2)
            elif data['state'][0][i] == 3: #静态数值
                in_size += 1
                setattr(self, 'fnn' + str(i + 1), nn.Linear(in_size, hidden_size))
                setattr(self, 'fnns' + str(i + 1), nn.Linear(hidden_size, int(hidden_size / 2)))
                setattr(self, 'out' + str(i + 1), nn.Linear(int(hidden_size / 2), output_size))
                in_size = int(hidden_size / 2)
            elif data['state'][0][i] == 4: #动态数值 or data['state'][0][i] == 3
                in_size += 1
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=hidden_size,
                    num_layers=num_layers, batch_first=True))
                setattr(self, 'fnn' + str(i + 1), nn.Linear(hidden_size, int(hidden_size/2)))
                setattr(self, 'out' + str(i + 1), nn.Linear(int(hidden_size/2), output_size))
                in_size = int(hidden_size / 2)

    def forward(self, x, data):
        output = []
        for i in range(len(data['state'][0])):
            if data['state'][0][i] == 0: # 活动
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(i) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                nn_out = self.__getattr__('rnn' + str(i + 1))(xi)
                nn_out1 = self.__getattr__('fnn' + str(i + 1))(nn_out[0])
                nn_out2 = self.__getattr__('out' + str(i + 1))(nn_out1)
                output.append(nn_out2)
            elif data['state'][0][i] == 1:  # 静态分类
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(data['index'][0][i]) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                    x2 = torch.cat([nn_out1.detach(), xi], dim=2)
                else:
                    x2 = torch.cat([nn_out1.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                nn_out = self.__getattr__('fnn' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('fnns' + str(i + 1))(nn_out)
                nn_out2 = self.__getattr__('out' + str(i + 1))(nn_out1)
                output.append(nn_out2)
            elif data['state'][0][i] == 2:  # 动态分类 or data['state'][0][i] == 1
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(data['index'][0][i]) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                    x2 = torch.cat([nn_out1.detach(), xi], dim=2)
                else:
                    x2 = torch.cat([nn_out1.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                nn_out = self.__getattr__('rnn' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('fnn' + str(i + 1))(nn_out[0])
                nn_out2 = self.__getattr__('out' + str(i + 1))(nn_out1)
                output.append(nn_out2)
            elif data['state'][0][i] == 3:  # 静态数值
                xi = x[:, :, i].view(x.size(0), x.size(1))
                x2 = torch.cat([nn_out1.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                nn_out = self.__getattr__('fnn' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('fnns' + str(i + 1))(nn_out)
                nn_out2 = self.__getattr__('out' + str(i + 1))(nn_out1)
                output.append(nn_out2)
            elif data['state'][0][i] == 4:  # 动态数值 or data['state'][0][i] == 3
                xi = x[:, :, i].view(x.size(0), x.size(1))
                x2 = torch.cat([nn_out1.detach(), xi.view(x.size(0), x.size(1),1)], dim=2)
                nn_out = self.__getattr__('rnn' + str(i + 1))(x2)
                nn_out1 = self.__getattr__('fnn' + str(i + 1))(nn_out[0])
                nn_out2 = self.__getattr__('out' + str(i + 1))(nn_out1)
                output.append(nn_out2)
        return output, nn_out#.view(x.size(0), x.size(1))

class FNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(FNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.out2 = nn.Linear(int(hidden_size/2), 1)

    def forward(self, x):
        h_out = self.hidden(x)
        outs1 = self.out1(torch.sigmoid(h_out))
        outs = self.out2(torch.sigmoid(outs1))
        return outs.view(-1, x.size(1))[:,-1]

class OLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_size, batch_size=20, n_layer=1, dropout=0, embedding = None):
        super(OLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout,
                           num_layers=self.n_layer, bidirectional=False)
        self.out = nn.Linear(hidden_dim, out_size)

    def forward(self, X):
        X = X.view(X.size(0),X.size(1)).long()
        input = self.embedding(X)
        input = input.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.rnn(input)
        hn = output[-1]
        output = self.out(hn)
        return output

class ProTransformer(nn.Module):
    def __init__(self, emb, pos, dim=36, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=True, norm_first=True) -> None:
        super(ProTransformer, self).__init__()
        self.emb = emb
        self.pos = pos
        encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first)
        encoder_norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.pooling = nn.AdaptiveAvgPool1d(dim)
        self.fnn1 = nn.Linear(2, 32)
        self.dropout1 = nn.Dropout(dropout)
        self.fnn2 = nn.Linear(dim+32, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fnn3 = nn.Linear(128, 1)

    def forward(self, x1, x2, PosEncode):
        emb = self.emb(x1.long())
        pos = self.pos(PosEncode.long())
        memory = self.encoder(emb+pos)
        out1 = self.pooling(memory)
        out2 = self.fnn1(F.relu(x2))
        output = torch.cat([out1, out2], dim=2)
        output = self.dropout1(output)
        output = self.fnn2(F.relu(output))
        output = self.dropout2(output)
        output = self.fnn3(output)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

class ProTransformer0(nn.Module):
    def __init__(self, emb, pos, dim=36, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=True, norm_first=True) -> None:
        super(ProTransformer0, self).__init__()
        self.emb = emb
        self.pos = pos
        encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first)
        encoder_norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.pooling = nn.AdaptiveAvgPool1d(dim)
        self.fnn2 = nn.Linear(dim, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fnn3 = nn.Linear(128, 1)

    def forward(self, x1, x2, PosEncode):
        emb = self.emb(x1.long())
        pos = self.pos(PosEncode.long())
        memory = self.encoder(emb+pos)
        out = self.pooling(memory)
        output = self.fnn2(F.relu(out))
        output = self.dropout2(output)
        output = self.fnn3(output)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

class ProTransformer1(nn.Module):
    def __init__(self, input_size, pos_dim, data, dim=36, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=True, norm_first=True) -> None:
        super(ProTransformer1, self).__init__()
        self.input_size = input_size
        self.dimc = 0
        self.dimd = 0
        for i in range(input_size):
            if data['state'][0][i] < 3: # 分类
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    self.dimc += torch.tensor(data[str(data['index'][0][i])]).size(1)
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3: #数值
                self.dimd += 1
        self.pos = nn.Embedding(pos_dim, self.dimc)
        encoder_layer = nn.TransformerEncoderLayer(self.dimc, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first)
        encoder_norm = nn.LayerNorm(self.dimc, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.pooling = nn.AdaptiveAvgPool1d(self.dimc)
        self.fnn1 = nn.Linear(self.dimd, 32)
        self.dropout1 = nn.Dropout(dropout)
        self.fnn2 = nn.Linear(self.dimc + 32, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fnn3 = nn.Linear(128, 1)

    def forward(self, x, PosEncode, data):
        inputd = None
        for i in range(self.input_size):
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
                inputc = torch.cat([inputc.detach(), xc.detach()], dim=2)
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                xd = x[:, :, i].view(x.size(0), x.size(1), 1)
                if inputd == None:
                    inputd = xd
                else:
                    inputd = torch.cat([inputd.detach(), xd.detach()], dim=2)
        pos = self.pos(PosEncode.long())
        memory = self.encoder(inputc + pos)
        out1 = self.pooling(memory)
        out2 = self.fnn1(F.relu(inputd))
        output = torch.cat([out1, out2], dim=2)
        output = self.dropout1(output)
        output = self.fnn2(F.relu(output))
        output = self.dropout2(output)
        output = self.fnn3(output)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

class Transformer(nn.Module):
    def __init__(self, dim=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=True, norm_first=True) -> None:
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first)
        encoder_norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        # self.pooling = nn.AdaptiveAvgPool1d(dim)
        # self.fnn1 = nn.Linear(dim, dim_feedforward)
        self.decoder = nn.Linear(dim, 1)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # output = self.pooling(memory)
        # output = self.dropout1(output)
        # output = self.fnn1(F.relu(memory))
        # output = self.dropout2(output)
        output = self.decoder(memory)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
#不分层直接输入所有特征，特征分-合训练
class DNN1(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data, type):
        super(DNN1, self).__init__()
        self.input_size = input_size
        output_size = 1
        for i in range(input_size):
            if data['state'][0][i] == 0: # 活动
                if '0' in data.keys():
                    self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                    in_size = self.embed1.embedding_dim
                    if type == 1:
                        output_size = in_size
                self.rnn1 = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1: #分类
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    in_size = torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    in_size = 8
                    setattr(self, 'fnnf' + str(i + 1), nn.Linear(1, in_size))
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=in_size*2,
                    num_layers=num_layers, batch_first=True))
                hidden_size += in_size * 2
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3: #数值
                in_size = 8
                setattr(self, 'fnnf' + str(i + 1), nn.Linear(1, in_size))
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=in_size*2,
                    num_layers=num_layers, batch_first=True))
                hidden_size += in_size*2
        self.rnnA = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fnn = nn.Linear(hidden_size, int(hidden_size / 2))
        self.out = nn.Linear(int(hidden_size / 2), output_size)

    def forward(self, x, data):
        output = []
        for i in range(self.input_size):
            if data['state'][0][i] == 0: # 活动
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(i) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                nn_out = self.__getattr__('rnn' + str(i + 1))(xi)[0]
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                if str(data['index'][0][i]) in data.keys():
                    xi = x[:, :, i].view(x.size(0), x.size(1))
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                else:
                    xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    xi = self.__getattr__('fnnf' + str(i + 1))(xi)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(xi)[0]
                nn_out = torch.cat([nn_out.detach(), nn_out1.detach()], dim=2)
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                xi = self.__getattr__('fnnf' + str(i + 1))(xi)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(xi)[0]
                nn_out = torch.cat([nn_out.detach(), nn_out1.detach()], dim=2)
        rnnA_out = self.rnnA(nn_out)[0]
        fnn_out = self.fnn(rnnA_out)
        out = self.out(fnn_out)
        output.append(out)
        return output, out
#不分层直接输入所有特征，特征合训练
class DNN2(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data, type):
        super(DNN2, self).__init__()
        self.input_size = input_size
        output_size = 1
        for i in range(input_size):
            if data['state'][0][i] == 0: # 活动
                in_size = 1
                if '0' in data.keys():
                    self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                    in_size = self.embed1.embedding_dim
                    if type == 1:
                        output_size = in_size
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1: #分类
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    in_size += torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    in_size += 1
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3: #数值
                in_size += 1
        self.rnn = nn.LSTM(input_size=in_size, hidden_size=in_size*2, num_layers=num_layers, batch_first=True)
        self.fnn = nn.Linear(in_size*2, in_size)
        self.out = nn.Linear(in_size, output_size)

    def forward(self, x, data):
        output = []
        for i in range(self.input_size):
            if data['state'][0][i] == 0: # 活动
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

        # inputR = input.chunk(input.size(1), 1)
        # flag = 0
        # for ir in inputR:
        #     if flag == 0:
        #         rnn_out, (hn, cn) = self.rnn(ir)#[0]
        #         flag = 1
        #     else:
        #         rnn_out, (hn, cn) = self.rnn(ir, (hn, cn))
        #     print(torch.tanh(cn))

        rnn_out = self.rnn(input)[0]
        fnn_out = self.fnn(rnn_out)
        out = self.out(fnn_out)
        output.append(out)
        return output, out
#不分层直接输入所有特征，特征合训练（数值处理）
class DNN3(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data, type):
        super(DNN3, self).__init__()
        self.input_size = input_size
        output_size = 1
        hidden_size = 0
        for i in range(input_size):
            if data['state'][0][i] == 0: # 活动
                in_size = 1
                if '0' in data.keys():
                    self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                    in_size = self.embed1.embedding_dim
                    if type == 1:
                        output_size = in_size
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1: #分类
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    in_size = torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    in_size = 8
                    setattr(self, 'fnnf' + str(i + 1), nn.Linear(1, in_size))
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3: #数值
                in_size = 8
                setattr(self, 'fnnf' + str(i + 1), nn.Linear(1, in_size))
            hidden_size += in_size
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size*2, num_layers=num_layers, batch_first=True)
        self.fnn = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, data):
        output = []
        for i in range(self.input_size):
            if data['state'][0][i] == 0: # 活动
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
                    xi = self.__getattr__('fnnf' + str(i + 1))(xi)
                input = torch.cat([input.detach(), xi.detach()], dim=2)
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                xi = self.__getattr__('fnnf' + str(i + 1))(xi)
                input = torch.cat([input.detach(), xi.detach()], dim=2)
        rnn_out = self.rnn(input)[0]
        fnn_out = self.fnn(rnn_out)
        out = self.out(fnn_out)
        output.append(out)
        return output, out

class MultiNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data, type):
        super(MultiNN, self).__init__()
        self.input_size = input_size
        output_size = 1
        for i in range(input_size):
            if data['state'][0][i] == 0: # 活动
                if '0' in data.keys():
                    self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                    in_size = self.embed1.embedding_dim
                    if type == 1:
                        output_size = in_size
                self.rnn1 = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1: #分类
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    in_size = torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    in_size = 8
                    setattr(self, 'fnnf' + str(i + 1), nn.Linear(1, in_size))
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=in_size*2,
                    num_layers=num_layers, batch_first=True))
                hidden_size += in_size * 2
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3: #数值
                in_size = 8
                setattr(self, 'fnnf' + str(i + 1), nn.Linear(1, in_size))
                setattr(self, 'rnn' + str(i + 1), nn.LSTM(input_size=in_size, hidden_size=in_size*2,
                    num_layers=num_layers, batch_first=True))
                hidden_size += in_size*2
        self.rnnA = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fnnR = nn.Linear(hidden_size, int(hidden_size / 2))
        self.outR = nn.Linear(int(hidden_size / 2), 1)
        self.fnnE = nn.Linear(hidden_size, int(hidden_size / 2))
        self.outE = nn.Linear(int(hidden_size / 2), output_size)
        self.fnnD = nn.Linear(hidden_size, int(hidden_size / 2))
        self.outD = nn.Linear(int(hidden_size / 2), 1)

    def forward(self, x, data):
        output = []
        for i in range(self.input_size):
            if data['state'][0][i] == 0: # 活动
                xi = x[:, :, i].view(x.size(0), x.size(1))
                if str(i) in data.keys():
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                nn_out = self.__getattr__('rnn' + str(i + 1))(xi)[0]
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                if str(data['index'][0][i]) in data.keys():
                    xi = x[:, :, i].view(x.size(0), x.size(1))
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                else:
                    xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    xi = self.__getattr__('fnnf' + str(i + 1))(xi)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(xi)[0]
                nn_out = torch.cat([nn_out.detach(), nn_out1.detach()], dim=2)
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                xi = self.__getattr__('fnnf' + str(i + 1))(xi)
                nn_out1 = self.__getattr__('rnn' + str(i + 1))(xi)[0]
                nn_out = torch.cat([nn_out.detach(), nn_out1.detach()], dim=2)
        rnnA_out = self.rnnA(nn_out)[0]
        fnn_outR = self.fnnR(rnnA_out)
        outR = self.outR(fnn_outR)
        fnn_outE = self.fnnE(rnnA_out)
        outE = self.outE(fnn_outE)
        fnn_outD = self.fnnD(rnnA_out)
        outD = self.outD(fnn_outD)
        output = torch.cat([outR.detach(), outE.detach(), outD.detach()], dim=2)
        return output, [outR, outE, outD]

class MultiNN3(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,data, type):
        super(MultiNN3, self).__init__()
        self.input_size = input_size
        for i in range(input_size):
            if data['state'][0][i] == 0: # 活动
                if '0' in data.keys():
                    self.embed1 = nn.Embedding.from_pretrained(torch.tensor(data['0']))
                    in_size = self.embed1.embedding_dim
                    output_size = in_size
                    hidden_size = output_size
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1: #分类
                if str(data['index'][0][i]) in data.keys():
                    setattr(self, 'embed' + str(i + 1), nn.Embedding.from_pretrained(torch.tensor(data[str(data['index'][0][i])])))
                    in_size = torch.tensor(data[str(data['index'][0][i])]).size(1)
                else:
                    in_size = 8
                    setattr(self, 'fnnf' + str(i + 1), nn.Linear(1, in_size))
                hidden_size += in_size
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3: #数值
                in_size = 8
                setattr(self, 'fnnf' + str(i + 1), nn.Linear(1, in_size))
                hidden_size += in_size
        self.rnnA = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size*2, num_layers=num_layers, batch_first=True)
        self.fnnR = nn.Linear(hidden_size*2, hidden_size)
        self.outR = nn.Linear(hidden_size, 1)
        self.fnnE = nn.Linear(hidden_size*2, hidden_size)
        self.outE = nn.Linear(hidden_size, output_size)
        self.fnnD = nn.Linear(hidden_size*2, hidden_size)
        self.outD = nn.Linear(hidden_size, 1)

    def forward(self, x, data):
        for i in range(self.input_size):
            if data['state'][0][i] == 0: # 活动
                x1 = x[:, :, i].view(x.size(0), x.size(1))
                if str(i) in data.keys():
                    x1 = torch.LongTensor(np.array(x1))
                    x1 = self.__getattr__('embed' + str(i + 1))(x1)
            elif data['state'][0][i] == 2 or data['state'][0][i] == 1:  # 分类
                if str(data['index'][0][i]) in data.keys():
                    xi = x[:, :, i].view(x.size(0), x.size(1))
                    xi = torch.LongTensor(np.array(xi))
                    xi = self.__getattr__('embed' + str(i + 1))(xi)
                else:
                    xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                    xi = self.__getattr__('fnnf' + str(i + 1))(xi)
                x1 = torch.cat([x1, xi], dim=2)
            elif data['state'][0][i] == 4 or data['state'][0][i] == 3:  # 数值
                xi = x[:, :, i].view(x.size(0), x.size(1), 1)
                xi = self.__getattr__('fnnf' + str(i + 1))(xi)
                x1 = torch.cat([x1, xi], dim=2)
        rnnA_out = self.rnnA(x1)[0]
        fnn_outR = self.fnnR(rnnA_out)
        outR = self.outR(fnn_outR)
        fnn_outE = self.fnnE(rnnA_out)
        outE = self.outE(fnn_outE)
        fnn_outD = self.fnnD(rnnA_out)
        outD = self.outD(fnn_outD)
        output = torch.cat([outR.detach(), outE.detach(), outD.detach()], dim=2)
        return output, [outR, outE, outD]

if __name__ == "__main__":
    transformer_model = Transformer(dim=20, nhead=4, num_layers=2)
    src = torch.rand((10, 100, 20))
    src_mask = torch.rand((400, 10, 10))
    src_key_padding_mask = torch.ones((100, 10))
    out = transformer_model(src, src_mask, src_key_padding_mask)
    # transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
    # src = torch.rand((10, 32, 512))
    # tgt = torch.rand((20, 32, 512))
    # out = transformer_model(src, tgt)