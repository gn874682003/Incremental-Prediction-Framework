import math
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from BPP_Frame.Log.DataRecord import DataRecord as DR
import BPP_Frame.Log.LogConvert as LC
import BPP_Frame.Feature.FeatureSel as FS
import BPP_Frame.Feature.CompareMethod as FC
import BPP_Frame.Log.LogAnalysis as LA
import BPP_Frame.Log.DivideData as DD
import BPP_Frame.Log.Prefix as P
import BPP_Frame.Method.My.multiModel as M
import BPP_Frame.Method.My.Model as M0
import BPP_Frame.Code.word2vec as w2v
import BPP_Frame.Method.AETS.multiset as multiset
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import  LabelBinarizer
import copy
import pandas as pd
#'CoSeLoG',[3,4,5,6,7,8,9],'BPIC2017',[3,4,5,7,9,11,12,14,15,16]
EL = ['BPIC2017','CoSeLoG','sepsis','BPIC2015_3','hd','BPIC2012','Production_Data','BPIC2015_1','BPIC2015_2','BPIC2015_4','BPIC2015_5']#,'RRT','hospital','sepsis'
Att = [[3,4,5,7,9,11,13,14],[3,4,5,6,7,8,9],
       [3,5,6,7,8,9,10,11,12,14,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
       [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[3,4,5,6,7,8,9,10,11,12,13],[3,5],[3,4,5,6,7,8,9,10,11,12],
       [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
       [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
       # [3,5,6,7,9,10,11,12,13],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
for eventlog,attribute in zip(EL,Att):
    print(eventlog)
    # 属性转换
    DR.Convert, DR.header, DR.ConvertReflact, maxA, maxR = LC.LogC(eventlog, attribute)#需转换属性值的下标
    actSum = len(DR.ConvertReflact[0])
    # 特征类别编号
    # 0 活动；1 分类静态特征；2 分类动态特征；3 数值静态特征；4 数值动态特征
    DR.State = []
    DR.State.append(0)
    for i in range(4, len(DR.Convert[0])-9):
        if i in attribute:
            DR.State.append(1)
        else:
            DR.State.append(3)
    for i in range(6):
        DR.State.append(3)
    # 数据集划分
    DR.Train, DR.Test, DR.AllData = DD.DiviData(DR.Convert, DR.State)
    # 特征选择比较方法一
    # FC.RFECV(DR.Train, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
    # DR.Train, DR.Val = train_test_split(DR.Train, test_size=0.2, random_state=20)#20
    # 特征选择比较方法二
    # an, aii = FC.NullImportance() # aii = [13, 0, 7, 8, 10] [0, 12, 2, 3, 11]
    # FS.TestK(DR.Train, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))], aii)

    # FR = FS.FTTree(DR.Train, DR.Val, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
    FR = FS.LightGBMNew(DR.Train, DR.Train, DR.Train, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])

    # FR = FS.LightGBM(DR.Train,  DR.Val, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])

    # #
    # # # DR.Test1, DR.Test2 = train_test_split(DR.TrainAll, test_size=0.5, random_state=20)
    # # # DR.Test3, DR.Test4 = train_test_split(DR.Test2, test_size=0.5, random_state=20)
    # # # DR.Test1, DR.Test2 = train_test_split(DR.Test1, test_size=0.5, random_state=20)
    # # # DR.Train1 = DR.Test2+DR.Test3+DR.Test4
    # # # DR.Train2 = DR.Test1+DR.Test3+DR.Test4
    # # # DR.Train3 = DR.Test1+DR.Test2+DR.Test4
    # # # DR.Train4 = DR.Test1+DR.Test2+DR.Test3
    # # # FR1 = FS.LightGBMNew(DR.Train1, DR.Test1, DR.TrainAll, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
    # # # FR2 = FS.LightGBMNew(DR.Train2, DR.Test2, DR.TrainAll, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
    # # # FR3 = FS.LightGBMNew(DR.Train3, DR.Test3, DR.TrainAll, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
    # # # FR4 = FS.LightGBMNew(DR.Train4, DR.Test4, DR.TrainAll, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
    # # # ai = FR1[2][0:math.ceil(len(FR1[2])*0.6)]
    # # # for i in FR2[2][0:math.ceil(len(FR2[2])*0.6)]:
    # # #     if i not in ai:
    # # #         ai.append(i)
    # # # for i in FR3[2][0:math.ceil(len(FR3[2])*0.6)]:
    # # #     if i not in ai:
    # # #         ai.append(i)
    # # # for i in FR4[2][0:math.ceil(len(FR4[2])*0.6)]:
    # # #     if i not in ai:
    # # #         ai.append(i)
    # # # # ai = []
    # # # # for i in list(set(FR1[2]+FR2[2]+FR3[2]+FR4[2])):
    # # # #     num = 0
    # # # #     if i in FR1[2]:
    # # # #         num += 1
    # # # #     if i in FR2[2]:
    # # # #        num += 1
    # # # #     if i in FR3[2]:
    # # # #         num += 1
    # # # #     if i in FR4[2]:
    # # # #         num += 1
    # # # #     if num > 2:
    # # # #         ai.append(i)
    # # # print(ai)
    # # # FR = FS.TestK(DR.TrainAll, DR.Test, DR.header,[attribute[i]-3 for i in range(len(attribute))], ai)
    # # # # print(FE)
    # # # # print(FD)
    # # # # print(FR)
    # # # # FE = [0.47114093959731546, ['Activity', 'Qty Completed', 'hour'], [0, 6, 15]]
    # # # # FD = [1.2374524083469405, ['Activity'], [0]]
    # # # # FR = [8.978115615700178, ['Activity', 'month', 'Part Desc.', 'Worker ID', 'day', 'hour'], [0, 12, 3, 4, 13, 15]]
    # # # state = [j for i, j in zip(DR.State,range(len(DR.State))) if i == 2 or i == 4]
    # # # PR = FS.PrefixLightGBM(DR.Train, DR.Val, DR.Test, DR.header, state, [attribute[i] - 3 for i in range(len(attribute))], FR)#FE, FD,
    # # 活动编码 训练
    # DR.Train_XA, DR.Train_YA = P.cutPrefixBy(DR.Train, [0], label=-3, batchSize=20, LEN=3)  # [FR[2][0]]
    # EmbA, ACCE = w2v.word2vec(DR.Train_XA, DR.Train_YA, DR.ConvertReflact)
    # # dataFE = {'0': EmbA.detach().numpy(), 'name': FE[1], 'index': FE[2], 'state': [DR.State[i] for i in FE[2]],
    # #           'result': FE[0], 'prefix': PE}
    # # dataFD = {'0': EmbA.detach().numpy(), 'name': FD[1], 'index': FD[2], 'state': [DR.State[i] for i in FD[2]],
    # #           'result': FD[0], 'prefix': PD}
    # dataFR = {'0': EmbA.detach().numpy(), 'name': FR[1], 'index': FR[2], 'state': [DR.State[i] for i in FR[2]],
    #           'result': FR[0]}#, 'prefix': PR
    # # 其他分类特征编码 随机初始化Embding
    # for i in range(1, len(DR.Train[0][0]) - 3):
    #     if i + 3 in attribute:
    #         if len(DR.ConvertReflact[attribute.index(i + 3)]) > 5:
    #             eim = 5
    #             olen = len(DR.ConvertReflact[attribute.index(i + 3)])
    #             while olen > 20:
    #                 olen /= 4
    #                 eim += 5
    #             EmbS = nn.Embedding(len(DR.ConvertReflact[attribute.index(i + 3)]), eim)
    #             # X_tsne = TSNE(n_components=2, learning_rate=0.1).fit_transform(EmbS.weight.detach().numpy())
    #             # for j in range(len(X_tsne)):
    #             #     plt.scatter(X_tsne[j, 0], X_tsne[j, 1])
    #             # plt.xlabel('x')
    #             # plt.ylabel('y')
    #             # plt.title(DR.header[i])
    #             # plt.show()
    #             # if i in dataFE['index']:
    #             #     dataFE[str(i)] = EmbS.weight.detach().numpy()
    #             # if i in dataFD['index']:
    #             #     dataFD[str(i)] = EmbS.weight.detach().numpy()
    #             if i in dataFR['index']:
    #                 dataFR[str(i)] = EmbS.weight.detach().numpy()

    # 保存编码
    # dataNameFE = '../Save/multiFea/preFE' + eventlog + '.mat'
    # dataNameFD = '../Save/multiFea/preFD' + eventlog + '.mat'
    dataNameFR = '../Save/multiFea/' + eventlog + 'Z.mat'
    # scio.savemat(dataNameFE, dataFE)
    # scio.savemat(dataNameFD, dataFD)
    # scio.savemat(dataNameFR, dataFR)
    # 读取编码
    # dataFE = scio.loadmat(dataNameFE)
    # dataFD = scio.loadmat(dataNameFD)
    dataFR = scio.loadmat(dataNameFR)
    # FS.TestLGBM(DR.Train, DR.Test, dataFE, dataFD, dataFR)

    preType = 2  # 预测类型 1分类 2回归
    task = -1  # 预测任务 -1剩余时间 -2下一事件时间 -3下一事件
    epoch = 300
    batchSize = 100
    LEN = 10
    hiddenSize = 32
    numLayer = 1
    feature = dataFR['index'][-1]

    DR.Train_batch = P.NoFill(DR.Train, feature, task, batchSize)
    DR.Val_X, DR.Val_Y = P.changeLen(DR.Test, feature, task, 1)
    DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, feature, task, 1)

    # 仅活动，索引编码
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, 1, hiddenSize, 1, 'rnn', dataFR)
    # Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
    # print(Metric, '仅活动，索引编码')
    # 仅活动，One-hot编码
    # act = list(DR.ConvertReflact[0].keys())
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, len(act), hiddenSize, 1, 'rnn', dataFR)
    # Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
    # print(Metric, '仅活动，One-hot编码')
    # # 仅活动，CBOW编码
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, 0, hiddenSize, 1, 'rnn', dataFR)
    # Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
    # print(Metric, '仅活动，CBOW编码')
    # 全部拼接，索引编码
    # timeS = time.time()
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, -1, hiddenSize, 1, 'rnn', dataFR)
    # Metric, eval_loss = M.test( DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
    # timeE = time.time()
    # print(Metric, timeE-timeS,'s,全部拼接，索引编码')
    # 全部拼接，向量编码
    # timeS = time.time()
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, -2, hiddenSize, numLayer, 'rnn', dataFR)
    # Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], preType, dataFR)
    # timeE = time.time()
    # print(Metric, timeE - timeS, 's,全部拼接，向量编码')
    # # 全部拼接，活动CBOW，其他索引编码
    # timeS = time.time()
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, -3, hiddenSize, numLayer, 'rnn', dataFR)
    # Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], preType, dataFR)
    # timeE = time.time()
    # print(Metric, timeE - timeS, 's,全部拼接，混合编码')
    # 构造可解释训练集，全部拼接，向量编码
    # EachTrain = []
    # for i in range(len(feature)):
    #     Train = copy.deepcopy(DR.Train)
    #     for line1 in Train:
    #         lineT = []
    #         for line2 in line1:
    #             for j in range(len(line2)-3):
    #                 if j not in feature[:i + 1]:
    #                     line2[j] = 0
    #             lineT.append(line2)
    #         EachTrain.append(lineT)
    # DR.Train_batch = P.NoFill(EachTrain, feature, task, batchSize)
    # # DR.Val_X, DR.Val_Y = P.changeLen(DR.Val.copy(), feature, task, 1)
    # timeS = time.time()
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, -2, hiddenSize, 1, 'rnn', dataFR)
    # for i in range(len(feature)):
    #     EachTest = []
    #     Test = copy.deepcopy(DR.Test)
    #     for line1 in Test:
    #         lineT = []
    #         for line2 in line1:
    #             for j in range(len(line2)-3):
    #                 if j not in feature[:i + 1]:
    #                     line2[j] = 0
    #             lineT.append(line2)
    #         EachTest.append(lineT)
    #     DR.Test_X, DR.Test_Y = P.changeLen(EachTest, feature, task, 1)
    #     Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
    #     timeE = time.time()
    #     print(Metric, timeE - timeS, 's,全部拼接，向量编码，可解释训练集')


    # 活动与其他特征分开输入
    # timeS = time.time()
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, -1, 32, 1, 'rnnm', dataFR)
    # Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], preType, dataFR)
    # timeE = time.time()
    # print(Metric, timeE-timeS, 's,多特征，活动与其他特征分开输入---------------------------------------------')
    # 前缀特征与其他特征分开输入
    # timeS = time.time()
    # MR = M.trian(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, -2, 32, 1, 'rnnm', dataFR)
    # Metric, eval_loss = M.test(DR.Test_X, DR.Test_Y, MR[0], type, dataFR, len(feature))
    # timeE = time.time()
    # print(Metric, timeE - timeS, 's,多特征，前缀特征与其他特征分开输入--------------------------------------- ')

    # Transformer 全部拼接
    # max_case_length = LA.GeneralIndicator(DR, DR.AllData)
    # timeS = time.time()
    # MR = M.trianT(DR.Train_batch, DR.Val_X, DR.Val_Y, epoch, preType, max_case_length, 'tran', dataFR)
    # Metric, eval_loss,_ = M.testT(DR.Test_X, DR.Test_Y, MR[0], type, dataFR)
    # timeE = time.time()
    # print(Metric, timeE - timeS, 's,Transformer 全部拼接')

    # Transformer 原方法
    # timeS = time.time()
    # MR = M0.trianPT(DR.Train_batch, DR.Val_X, DR.Val_Y, 100, preType, 36, max_case_length, 'tran', dataFR)
    # Metric, eval_loss = M0.testPT(DR.Test_X, DR.Test_Y, MR[0], type, dataFR, 1)
    # timeE = time.time()
    # print(Metric, timeE - timeS, 's,Transformer 原方法')

    # AutoEncoder 全部拼接
    # timeS = time.time()
    # MR = multiset.train(DR.AllData, DR.Train, DR.Val, 'multiset', feature, isMul=1)
    # Metric, count = multiset.test(DR.AllData, DR.Test, MR[0], MR[2], 'multiset', feature, isMul=1)
    # timeE = time.time()
    # print(Metric, timeE - timeS, 's, AutoEncoder 全部拼接')

    # 原AutoEncoder
    # timeS = time.time()
    # MR = multiset.train(DR.AllData, DR.Train, DR.Val, 'multiset', feature)
    # Metric, count = multiset.test(DR.AllData, DR.Test, MR[0], MR[2], 'multiset', feature)
    # timeE = time.time()
    # print(Metric, timeE - timeS, 's, AutoEncoder 原方法')