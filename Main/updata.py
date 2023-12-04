import math
import random
import time
import warnings
warnings.filterwarnings('ignore')
import BPP_Frame.Log.DataRecord as DR
import BPP_Frame.Log.Prefix as P
import BPP_Frame.Method.My.multiModel as M
import BPP_Frame.Method.AETS.multiset as multiset
import copy
import torch
import scipy.io as scio
from scipy.stats import pearsonr
from skmultiflow.drift_detection.adwin import ADWIN
import numpy as np
import matplotlib.pyplot as plt
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.drift_detection import KSWIN
from skmultiflow.drift_detection import PageHinkley
import BPP_Frame.Feature.FeatureSel as FS
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


preType = 2 #预测类型 1分类 2回归
task = -1 #预测任务 -1剩余时间 -2下一事件时间 -3下一事件
batchSize = 100

def updateAuto0(event_list, MR, method, dataFR, attribute, Buckets, Time, mcl = None, model = None, hisdata = 1):
    dt = -1
    feature = dataFR['index'][-1]
    # Train_X, Train_Y = P.changeLen(DR.Train.copy(), feature, task, 1)
    # 按月分割represents，画图分析
    # Train_X = []
    # Train_Y = []
    # MonthTrace = {}
    # MonthCenter = {}
    # MonthTrace[str(DR.Train[0][-1][-4])+str(DR.Train[0][-1][-8])] = []
    # for trace in DR.Train:
    #     if str(trace[-1][-4]) + str(trace[-1][-8]) not in MonthTrace.keys():
    #         if method == 'NoFill':
    #             MAE, countg, represents = M.test(Train_X, Train_Y, MR[-1], preType, dataFR, len(dataFR['index'][-1]))
    #             X = torch.zeros(len(represents), represents[0][0].shape[2])
    #             for i in range(len(represents)):
    #                 X[i, :] = represents[i][0].reshape(represents[0][0].shape[2])
    #             X[i, :] = represents[i][0].reshape(represents[0][0].shape[2])
    #         elif method == 'Transformer':
    #             MAE, countg, represents = M.testT(Train_X, Train_Y, MR[-1], preType, dataFR)
    #             X = torch.zeros(len(represents), represents[0].shape[1])
    #             for i in range(len(represents)):
    #                 X[i, :] = represents[i].reshape(represents[0].shape[1])
    #             X[i, :] = represents[i].reshape(represents[0].shape[1])
    #         Train_X = []
    #         Train_Y = []
    #         MonthTrace[str(trace[-1][-4]) + str(trace[-1][-8])] = []
    #         X = X.detach().numpy()
    #         center = [np.mean(X[:,i]) for i in range(X.shape[1])]
    #         MonthCenter[MonthKey] = center
    #     MonthKey = str(trace[-1][-4]) + str(trace[-1][-8])
    #     MonthTrace[str(trace[-1][-4]) + str(trace[-1][-8])].append(trace)
    #     traX = []
    #     traY = []
    #     for line in trace:
    #         temp = []
    #         for i in feature:
    #             temp.append(line[i])
    #         traX.append(temp)
    #     traY.append(line[task])
    #     Train_X.append(torch.Tensor([traX]))
    #     Train_Y.append(torch.Tensor([traY]))
    # if Train_X != []:
    #     if method == 'NoFill':
    #         MAE, countg, represents = M.test(Train_X, Train_Y, MR[-1], preType, dataFR, len(dataFR['index'][-1]))
    #         X = torch.zeros(len(represents), represents[0][0].shape[2])
    #         for i in range(len(represents)):
    #             X[i, :] = represents[i][0].reshape(represents[0][0].shape[2])
    #         X[i, :] = represents[i][0].reshape(represents[0][0].shape[2])
    #     elif method == 'Transformer':
    #         MAE, countg, represents = M.testT(Train_X, Train_Y, MR[-1], preType, dataFR)
    #         X = torch.zeros(len(represents), represents[0].shape[1])
    #         for i in range(len(represents)):
    #             X[i, :] = represents[i].reshape(represents[0].shape[1])
    #         X[i, :] = represents[i].reshape(represents[0].shape[1])
    #     X = X.detach().numpy()
    #     center = [np.mean(X[:, i]) for i in range(X.shape[1])]
    #     MonthCenter[MonthKey] = center
    # while 1:
    #     x = []
    #     y = []
    #     y2 = []
    #     temp = []
    #     count = 0
    #     for center in MonthCenter.keys():
    #         if temp == []:
    #             temp = center
    #             last = center
    #         x.append(center)
    #         a = np.linalg.norm(np.array(MonthCenter[temp]) - np.array(MonthCenter[center]))
    #         a2 = np.linalg.norm(np.array(MonthCenter[last]) - np.array(MonthCenter[center]))
    #         y.append(a)
    #         y2.append(a2)
    #         if a > max(y2):
    #             count += 1
    #         else:
    #             count = 0
    #         last = center
    #     plt.plot(x, y, ls='-')
    #     plt.plot(x, y2, ls='-')
    #     plt.show()
    #     if count > 11:
    #         MonthCenter.pop(temp)
    #         MonthTrace.pop(temp)
    #     else:
    #         break

    # variant = {}
    # num = 0
    # for trace in DR.Train:
    #     Train_X = []
    #     Train_Y = []
    #     for line in trace:
    #         temp = []
    #         for i in feature:
    #             temp.append(line[i])
    #         Train_X.append(temp)
    #     Train_Y.append(line[task])
    #     Train_X = torch.Tensor([[Train_X]])
    #     Train_Y = torch.Tensor([[Train_Y]])
    #     if method == 'NoFill':
    #         _, _, represents = M.test(Train_X, Train_Y, MR[-1], preType, dataFR, len(dataFR['index'][-1]))
    #         represents = represents[0][0].reshape(represents[0][0].shape[2]).detach().numpy()
    #     elif method == 'Transformer':
    #         _, _, represents = M.testT(Train_X, Train_Y, MR[-1], preType, dataFR)
    #         represents = represents[0].reshape(represents[0].shape[1]).detach().numpy()
    #     activity = ''
    #     for act in trace:
    #         activity = activity + str(act[0]) + ' '
    #     if activity not in variant.keys():
    #         variant[activity] = [represents]
    #     else:
    #         max = 0
    #         for rep in variant[activity]:
    #             similar = pearsonr(rep, represents)
    #             if similar[0] > max:
    #                 max = similar[0]
    #                 if max >= 0.99999999:
    #                     num += 1
    #                     break
    #         if max < 0.99999999:
    #             variant[activity].append(represents)
    # 轨迹向量表示
    # Train_X, Train_Y = P.changeLen(DR.Train, dataFR['index'][-1], task, 1)
    # if method == 'NoFill':
    #     _, _, represents = M.test(Train_X, Train_Y, MR[-1], preType, dataFR, len(dataFR['index'][-1]))
    #     X = torch.zeros(len(represents), represents[0][0].shape[2])
    #     for i in range(len(represents)):
    #         X[i,:] = represents[i][0].reshape(represents[0][0].shape[2])
    # elif method == 'Transformer':
    #     _, _, represents = M.testT(Train_X, Train_Y, MR[-1], preType, dataFR)
    # # 质心计算
    # X = X.detach().numpy()
    # center = [np.mean(X[:,i]) for i in range(X.shape[1])]
    # # 两时间间隔的质心相似度计算
    # a = pearsonr(center, center)  # 计算原中心点与新中心点的相似度
    # # 均值漂移聚类
    # from numpy import unique
    # from numpy import where
    # from sklearn.cluster import MeanShift
    # from matplotlib import pyplot
    # # 定义数据集
    #
    # # 定义模型
    # modelCluster = MeanShift(cluster_all=False,max_iter=500)
    # # 模型拟合与聚类预测
    # yhat = modelCluster.fit_predict(X)
    # # 检索唯一群集
    # clusters = unique(yhat)
    # # 为每个群集的样本创建散点图
    # for cluster in clusters:
    #     # 获取此群集的示例的行索引
    #     row_ix = where(yhat == cluster)
    #     # 创建这些样本的散布
    #     pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # # 绘制散点图
    # pyplot.show()
    # 新增与历史不同的轨迹判断
    count = 0

    MAEsum = [0 for x in feature]
    BucketTest = [[] for x in Buckets]
    BucketAllTest = [[] for x in Buckets]
    BucketTrain = copy.deepcopy(DR.BucketTrain)
    MAEList = []
    MAEListI = []
    MAES = 0
    countS = 0
    flag = 0
    timeS = time.time()
    adwin = KSWIN(alpha=0.001)#window_size=1000,stat_size=300
    # adwin = ADWIN()
    # adwin = PageHinkley()
    # 计算历史质心
    Train_X = []
    Train_Y = []
    for trace in DR.Train:
        traX = []
        traY = []
        for line in trace:
            temp = []
            for i in feature:
                temp.append(line[i])
            traX.append(temp)
        traY.append(line[task])
        Train_X.append(torch.Tensor([traX]))
        Train_Y.append(torch.Tensor([traY]))
    if method == 'NoFill':
        MAE, countg, represents = M.test(Train_X, Train_Y, MR[-1], preType, dataFR, len(dataFR['index'][-1]))
        X = torch.zeros(len(represents), represents[0][0].shape[2])#1000
        for i in range(len(X)):
            X[i, :] = represents[i][0].reshape(represents[0][0].shape[2])
        X[i, :] = represents[i][0].reshape(represents[0][0].shape[2])
        X = X.detach().numpy()
    elif method == 'multiset':
        X = multiset.represent(DR.AllData, DR.Train, 'multiset', dataFR['index'][-1], MR[-1], event_list)
    elif method == 'Transformer':
        MAE, countg, represents = M.testT(Train_X, Train_Y, MR[-1], preType, dataFR)
        X = torch.zeros(len(represents), represents[0].shape[1])
        for i in range(len(X)):
            X[i, :] = represents[i].reshape(represents[0].shape[1])
        X[i, :] = represents[i].reshape(represents[0].shape[1])
        X = X.detach().numpy()
    center = [np.mean(X[:, i]) for i in range(X.shape[1])]
    # 计算质心与各轨迹表示的距离，添加至adwin
    for ri in range(len(X)):
        a = np.linalg.norm(np.array(center) - np.array(X[ri,:]))
        adwin.add_element(a)

    # for train in DR.Train:
    #     Train_X = []
    #     Train_Y = []
    #     for line in train:
    #         temp = []
    #         for j in feature:
    #             temp.append(line[j])
    #         Train_X.append(temp)
    #         Train_Y.append(line[task])
    #     Train_X = torch.Tensor([[Train_X]])
    #     Train_Y = torch.Tensor([[Train_Y]])
    #     if method == 'NoFill':
    #         MAE, countg, _ = M.test(Train_X, Train_Y, MR[-1], preType, dataFR, len(dataFR['index'][-1]))
    #     elif method == 'Transformer':
    #         MAE, countg, _ = M.testT(Train_X, Train_Y, MR[-1], preType, dataFR)
    #     adwin.add_element(MAE[0])
    for test in DR.Test:
        for i in range(len(Buckets)):
            if test[dt][-8] in Buckets[i][0] and test[dt][-7] in Buckets[i][1] and test[dt][-6] in Buckets[i][2]:
                # if len(BucketTest[i]) >= 100:#len(BucketTest[i]) >= Time and
                #     flag = 1
                if adwin.detected_change():# and len(BucketAllTest[i]) > 30
                    flag = 1
                    # print(len(BucketAllTest[i]))
                if flag == 1:
                    Test_X, Test_Y = P.changeLen(BucketAllTest[i], feature, task, 1)
                    if method == 'NoFill':
                        MAEO, _, represents = M.test(Test_X, Test_Y, MR[-1], preType, dataFR, len(feature))
                        MAE, countg, represents = M.test(Test_X, Test_Y, MR[i], preType, dataFR, len(feature))
                        count += countg
                        MAEsum = [MAEsum[j] + MAE[j] * countg for j in range(len(MAE))]
                    elif method == 'multiset':
                        if len(Buckets) == 1:
                            MAEO, _ = multiset.test(DR.AllData, BucketAllTest[i], MR[0], MR[-1], 'multiset', feature, isMul=1)
                            MAE, countg = multiset.test(DR.AllData, BucketAllTest[i], MR[1], MR[-1], 'multiset', feature, isMul=1)
                        else:
                            MAEO = 0
                            MAE, countg = multiset.testBucket(DR.AllData, BucketAllTest[i], MR[-2], MR[i], 'multiset', feature, isMul=1)
                        MAE = [MAE.__float__()]
                        MAEO = [MAEO.__float__()]
                        count += countg
                        MAEsum = [MAEsum[0] + MAE[0] * countg]
                    elif method == 'Transformer':
                        MAEO, _, _ = M.testT(Test_X, Test_Y, MR[-1], preType, dataFR)
                        MAE, countg, _ = M.testT(Test_X, Test_Y, MR[i], preType, dataFR)
                        count += countg
                        MAEsum = [MAEsum[i] + MAE[i] * countg for i in range(len(MAE))]
                    MAEListI.append(MAE)
                    MAEList.append(MAEO)
                    if hisdata == 1:
                        # 1.随机抽取已训练过的数据集
                        TrainNum = len(BucketTrain[i])
                        a = random.sample(range(0, TrainNum), math.ceil(0.2 * TrainNum))
                        Tra = [BucketTrain[i][j] for j in a]
                        Tra.extend(BucketTest[i])
                    elif hisdata == 2:
                        # 2.所有数据
                        Tra = BucketTrain[i]
                        Tra.extend(BucketTest[i])
                    elif hisdata == 3:
                        # 3.不添加原始训练集
                        Tra = BucketTest[i]
                    # historyTrain = []
                    # for mt in MonthTrace:
                    #     historyTrain.extend(MonthTrace[mt])
                    # # Tra.extend(historyTrain) # 2.全部数据
                    #
                    # a = random.sample(range(0, len(historyTrain)), len(BucketAllTest[i]))
                    # Tra = [historyTrain[j] for j in a] # 3.历史数据随机抽取
                    # Tra.extend(BucketAllTest[i])  # 1.仅用新数据
                    # 不添加原始训练集
                    # Tra = BucketTest[i]
                    Train_batch = P.NoFill(Tra, feature, task, batchSize)
                    if method == 'NoFill':
                        MRU = M.trian(Train_batch, Test_X, Test_Y, 20, preType, -2, 32, 1, MR[i], dataFR, isEarly=0)
                        MR[i] = MRU[0]
                    elif method == 'multiset':
                        if len(Buckets) == 1:
                            MRU = multiset.update(DR.AllData, Tra, MR[1], MR[-1], 'multiset', feature, isMul=1)  # 长度分桶
                            MR[-1] = MRU[2]
                            MR[1] = MRU[0]
                        else:
                            MRU = multiset.trainBucket(DR.AllData, Tra, BucketTest[i], method, feature, MR[-2], MR[-1], isMul=1)  # 周期分桶
                            MR[i] = MRU[0]
                    elif method == 'Transformer':
                        MRU = M.trianT(Train_batch, Test_X, Test_Y, 20, preType, mcl, MR[i], dataFR)
                        MR[i] = MRU[0]
                    BucketTrain[i].extend(BucketAllTest[i])
                    BucketTest[i] = []
                    BucketAllTest[i] = []
                    flag = 0
                    adwin.reset()
                    # 添加历史轨迹
                    # for train in BucketTrain[i]:
                    #     Train_X = []
                    #     Train_Y = []
                    #     for line in train:
                    #         temp = []
                    #         for j in feature:
                    #             temp.append(line[j])
                    #         Train_X.append(temp)
                    #         Train_Y.append(line[task])
                    #     Train_X = torch.Tensor([[Train_X]])
                    #     Train_Y = torch.Tensor([[Train_Y]])
                    #     if method == 'NoFill':
                    #         MAE, countg, _ = M.test(Train_X, Train_Y, MR[i], preType, dataFR, len(dataFR['index'][-1]))
                    #     elif method == 'Transformer':
                    #         MAE, countg, _ = M.testT(Train_X, Train_Y, MR[i], preType, dataFR)
                    #     adwin.add_element(MAE[0])
                    # 计算历史质心
                    Train_X = []
                    Train_Y = []
                    for trace in BucketTrain[i]:#DR.Train
                        traX = []
                        traY = []
                        for line in trace:
                            temp = []
                            for j in feature:
                                temp.append(line[j])
                            traX.append(temp)
                        traY.append(line[task])
                        Train_X.append(torch.Tensor([traX]))
                        Train_Y.append(torch.Tensor([traY]))
                    if method == 'NoFill':
                        MAE, countg, represents = M.test(Train_X, Train_Y, MR[i], preType, dataFR, len(dataFR['index'][-1]))
                        X = torch.zeros(len(represents), represents[0][0].shape[2])
                        for j in range(len(X)):
                            X[j, :] = represents[j][0].reshape(represents[0][0].shape[2])
                        X[j, :] = represents[j][0].reshape(represents[0][0].shape[2])
                        X = X.detach().numpy()
                    elif method == 'multiset':
                        X = multiset.represent(DR.AllData, BucketTrain[i], 'multiset', dataFR['index'][-1], MR[-1], event_list)
                    elif method == 'Transformer':
                        MAE, countg, represents = M.testT(Train_X, Train_Y, MR[i], preType, dataFR)
                        X = torch.zeros(len(represents), represents[0].shape[1])
                        for j in range(len(X)):
                            X[j, :] = represents[j].reshape(represents[0].shape[1])
                        X[j, :] = represents[j].reshape(represents[0].shape[1])
                        X = X.detach().numpy()
                    center = [np.mean(X[:, j]) for j in range(X.shape[1])]
                    # 计算质心与各轨迹表示的距离，添加至adwin
                    for ri in range(len(X)):
                        a = np.linalg.norm(np.array(center) - np.array(X[ri, :]))
                        adwin.add_element(a)
                Train_X = []
                Train_Y = []
                for line in test:
                    temp = []
                    for j in feature:
                        temp.append(line[j])
                    Train_X.append(temp)
                    Train_Y.append(line[task])
                Train_X = torch.Tensor([[Train_X]])
                Train_Y = torch.Tensor([[Train_Y]])
                if method == 'NoFill':
                    MAE, countg, represents = M.test(Train_X, Train_Y, MR[i], preType, dataFR, len(dataFR['index'][-1]))
                    represents = represents[0][0].reshape(represents[0][0].shape[2]).detach().numpy()
                elif method == 'multiset':
                    represents = multiset.represent(DR.AllData, [test], 'multiset', dataFR['index'][-1], MR[-1], event_list)
                elif method == 'Transformer':
                    MAE, countg, represents = M.testT(Train_X, Train_Y, MR[i], preType, dataFR)
                    represents = represents[0].reshape(represents[0].shape[1]).detach().numpy()
                d2c = np.linalg.norm(np.array(represents) - np.array(center))
                adwin.add_element(d2c)
                # adwin.add_element(MAE[0])
                # if MAET < MAE:
                #     print(MAE)
                # MAES += countg * MAE[0]
                # countS += countg
                # if MAES/countS > MAET:
                #     flag = 1
                # BucketTest[i].append(test)
                BucketAllTest[i].append(test)
                # 判断是否为新轨迹
                # Train_X = []
                # Train_Y = []
                # for line in test:
                #     temp = []
                #     for j in feature:
                #         temp.append(line[j])
                #     Train_X.append(temp)
                # Train_Y.append(line[task])
                # Train_X = torch.Tensor([[Train_X]])
                # Train_Y = torch.Tensor([[Train_Y]])
                # if method == 'NoFill':
                #     _, _, represents = M.test(Train_X, Train_Y, MR[-1], preType, dataFR, len(dataFR['index'][-1]))
                #     represents = represents[0][0].reshape(represents[0][0].shape[2]).detach().numpy()
                # elif method == 'Transformer':
                #     _, _, represents = M.testT(Train_X, Train_Y, MR[-1], preType, dataFR)
                #     represents = represents[0].reshape(represents[0].shape[1]).detach().numpy()
                # activity = ''
                # for act in test:
                #     activity = activity + str(act[0]) + ' '
                # BucketAllTest[i].append(test)
                # if activity not in variant.keys():
                #     variant[activity] = [represents]
                #     BucketTest[i].append(test)
                # else:
                #     for rep in variant[activity]:
                #         similar = pearsonr(rep, represents)
                #         if similar[0] >= 0.99999999:
                #             break
                #     if similar[0] < 0.99999999:
                #         variant[activity].append(represents)
                #         BucketTest[i].append(test)

                # flagN = 0
                # for act in test:
                #     if flagN == 1:
                #         break
                #     activity = activity + str(act[0]) + ' '
                #     if act[0] not in activites.keys():
                #         flagN = 1
                #     else:
                #         for feaNum, state in zip(dataFR['index'][0][1:], dataFR['state'][0][1:]):
                #             if state < 3:
                #                 if act[feaNum] not in activites[act[0]][feaNum]:
                #                     flagN = 1
                #                     break
                #             else:
                #                 if act[feaNum] > activites[act[0]][feaNum - 1][1] or act[feaNum] < activites[act[0]][feaNum - 1][0]:
                #                     flagN = 1
                #                     break
                # if flagN == 0:
                #     if activity not in variant.keys():
                #         flagN = 1
                #     elif test[0][-1] > variant[activity][1] or test[0][-1] < variant[activity][0]:
                #         flagN = 1
                # if flagN == 1:
                #     BucketTest[i].append(test)
                # BucketAllTest[i].append(test)
    timeE = time.time()
    print('更新集:', hisdata, '增量:represents，MAE:', [MAEsum[j] / count for j in range(len(MAEsum))], timeE - timeS, 's')
    return MAEList, MAEListI

def code(FR, dataFR, attribute):
    for i in range(1, len(DR.Train[0][0]) - 3):
        if i + 3 in attribute:
            if len(DR.ConvertReflact[attribute.index(i + 3)]) > 5:
                eim = 5
                olen = len(DR.ConvertReflact[attribute.index(i + 3)])
                while olen > 20:
                    olen /= 4
                    eim += 5
                EmbS = nn.Embedding(len(DR.ConvertReflact[attribute.index(i + 3)]), eim)
                if i in FR[2] and i not in dataFR['index']:
                    dataFR[str(i)] = EmbS.weight.detach().numpy()
                elif i not in FR[2] and i in dataFR['index']:
                    dataFR.pop(str(i))
    dataFR['name'] = FR[1]
    dataFR['index'] = FR[2]
    dataFR['state'] = [DR.State[i] for i in FR[2]]
    dataFR['result'] = FR[0]
    dataNameFR = '../Save/multiFea/preFRTest.mat'
    scio.savemat(dataNameFR, dataFR)
    # 读取编码
    dataFR = scio.loadmat(dataNameFR)
    return dataFR

def updateAuto(MR, method, dataFR, attribute, Buckets, Time, mcl = None, model = None, hisdata = 1):
    dt = -1
    feature = dataFR['index'][-1]
    adwin = KSWIN()#window_size=1000, stat_size=300
    # Train_X, Train_Y = P.changeLen(DR.Train.copy(), feature, task, 1)
    # 按月分割represents，画图分析
    # trainX = []
    # trainY = []
    # MonthTrace = {}
    # MonthCenter = {}
    # MonthTrace[str(DR.Train[0][-1][-4])+str(DR.Train[0][-1][-8])] = []
    # for trace in DR.Train:
    #     if str(trace[-1][-4]) + str(trace[-1][-8]) not in MonthTrace.keys():
    #         if method == 'NoFill':
    #             MAE, countg, represents = M.test(trainX, trainY, MR[-1], preType, dataFR, len(dataFR['index'][-1]))
    #             X = torch.zeros(len(represents), represents[0][0].shape[2])
    #             for j in range(len(represents)):
    #                 X[j, :] = represents[j][0].reshape(represents[0][0].shape[2])
    #             X[j, :] = represents[j][0].reshape(represents[0][0].shape[2])
    #         elif method == 'Transformer':
    #             MAE, countg, represents = M.testT(trainX, trainY, MR[-1], preType, dataFR)
    #             X = torch.zeros(len(represents), represents[0].shape[1])
    #             for j in range(len(represents)):
    #                 X[j, :] = represents[j].reshape(represents[0].shape[1])
    #             X[j, :] = represents[j].reshape(represents[0].shape[1])
    #         trainX = []
    #         trainY = []
    #         MonthTrace[str(trace[-1][-4]) + str(trace[-1][-8])] = []
    #         X = X.detach().numpy()
    #         center = [np.mean(X[:,i]) for i in range(X.shape[1])]
    #         MonthCenter[MonthKey] = center
    #     MonthKey = str(trace[-1][-4]) + str(trace[-1][-8])
    #     MonthTrace[str(trace[-1][-4]) + str(trace[-1][-8])].append(trace)
    #     traX = []
    #     traY = []
    #     for line in trace:
    #         temp = []
    #         for i in feature:
    #             temp.append(line[i])
    #         traX.append(temp)
    #     traY.append(line[task])
    #     trainX.append(torch.Tensor([traX]))
    #     trainY.append(torch.Tensor([traY]))
    # while 1:
    #     x = []
    #     y = []
    #     y2 = []
    #     first = []
    #     countH = 0
    #     for center in MonthCenter.keys():
    #         if first == []:
    #             first = center
    #             last = center
    #         x.append(center)
    #         a = np.linalg.norm(np.array(MonthCenter[first]) - np.array(MonthCenter[center]))
    #         a2 = np.linalg.norm(np.array(MonthCenter[last]) - np.array(MonthCenter[center]))
    #         y.append(a)
    #         y2.append(a2)
    #         if a > max(y2):
    #             countH += 1
    #         else:
    #             countH = 0
    #         last = center
    #     plt.plot(x, y, ls='-')
    #     plt.plot(x, y2, ls='-')
    #     plt.show()
    #     if countH > 11:
    #         MonthCenter.pop(first)
    #         MonthTrace.pop(first)
    #     else:
    #         break
    # ----------------------------------------------------------------------

    count = 0
    MAEsum = [0 for x in feature]
    BucketTest = [[] for x in Buckets]
    BucketAllTest = [[] for x in Buckets]
    BucketTrain = copy.deepcopy(DR.BucketTrain)
    MAEList = []
    MAEListI = []
    flag = 0
    timeS = time.time()
    for train in DR.Train:
        Train_X = []
        Train_Y = []
        for line in train:
            temp = []
            for j in feature:
                temp.append(line[j])
            Train_X.append(temp)
            Train_Y.append(line[task])
        Train_X = torch.Tensor([[Train_X]])
        Train_Y = torch.Tensor([[Train_Y]])
        if method == 'NoFill':
            MAE, countg, _ = M.test(Train_X, Train_Y, MR[-1], preType, dataFR, len(dataFR['index'][-1]))
        elif method == 'multiset':
            MAE, countg = multiset.test(DR.AllData, [train], MR[1], MR[-1], 'multiset', feature, isMul=1)
            MAE = [MAE.__float__()]
        elif method == 'Transformer':
            MAE, countg, _ = M.testT(Train_X, Train_Y, MR[-1], preType, dataFR)
        adwin.add_element(MAE[0])
    for test in DR.Test:
        # if str(test[-1][-4]) + str(test[-1][-8]) not in MonthTrace.keys():
        #     if method == 'NoFill':
        #         MAE, countg, represents = M.test(trainX, trainY, MR[-1], preType, dataFR, len(dataFR['index'][-1]))
        #         X = torch.zeros(len(represents), represents[0][0].shape[2])
        #         for j in range(len(represents)):
        #             X[j, :] = represents[j][0].reshape(represents[0][0].shape[2])
        #         X[j, :] = represents[j][0].reshape(represents[0][0].shape[2])
        #     elif method == 'Transformer':
        #         MAE, countg, represents = M.testT(trainX, trainY, MR[-1], preType, dataFR)
        #         X = torch.zeros(len(represents), represents[0].shape[1])
        #         for j in range(len(represents)):
        #             X[j, :] = represents[j].reshape(represents[0].shape[1])
        #         X[j, :] = represents[j].reshape(represents[0].shape[1])
        #     trainX = []
        #     trainY = []
        #     MonthTrace[str(test[-1][-4]) + str(test[-1][-8])] = []
        #     X = X.detach().numpy()
        #     MonthCenter[MonthKey] = [np.mean(X[:,i]) for i in range(X.shape[1])]
        # MonthKey = str(test[-1][-4]) + str(test[-1][-8])
        # MonthTrace[str(test[-1][-4]) + str(test[-1][-8])].append(test)
        # traX = []
        # traY = []
        # for line in test:
        #     temp = []
        #     for i in feature:
        #         temp.append(line[i])
        #     traX.append(temp)
        # traY.append(line[task])
        # trainX.append(torch.Tensor([traX]))
        # trainY.append(torch.Tensor([traY]))
        # --------------------------------------------------------------------------------------------------
        for i in range(len(Buckets)):
            if test[dt][-8] in Buckets[i][0] and test[dt][-7] in Buckets[i][1] and test[dt][-6] in Buckets[i][2]:
                if adwin.detected_change():# and len(BucketAllTest[i]) > 100
                    flag = 1
                    # print(len(BucketAllTest[i]))
                if flag == 1:
                    Test_X, Test_Y = P.changeLen(BucketAllTest[i], feature, task, 1)
                    if method == 'NoFill':
                        MAEO, _, represents = M.test(Test_X, Test_Y, MR[-1], preType, dataFR, len(feature))
                        MAE, countg, represents = M.test(Test_X, Test_Y, MR[i], preType, dataFR, len(feature))
                        count += countg
                        MAEsum = [MAEsum[j] + MAE[j] * countg for j in range(len(MAE))]
                    elif method == 'multiset':
                        if len(Buckets) == 1:
                            MAEO, _ = multiset.test(DR.AllData, BucketAllTest[i], MR[0], MR[-1], 'multiset', feature, isMul=1)
                            MAE, countg = multiset.test(DR.AllData, BucketAllTest[i], MR[1], MR[-1], 'multiset', feature, isMul=1)
                        else:
                            MAEO = 0
                            MAE, countg = multiset.testBucket(DR.AllData, BucketAllTest[i], MR[-2], MR[i], 'multiset', feature, isMul=1)
                        MAE = [MAE.__float__()]
                        MAEO = [MAEO.__float__()]
                        count += countg
                        MAEsum = [MAEsum[0] + MAE[0] * countg]
                    elif method == 'Transformer':
                        MAEO, _, _ = M.testT(Test_X, Test_Y, MR[-1], preType, dataFR)
                        MAE, countg, _ = M.testT(Test_X, Test_Y, MR[i], preType, dataFR)
                        count += countg
                        MAEsum = [MAEsum[i] + MAE[i] * countg for i in range(len(MAE))]
                    MAEListI.append(MAE)
                    MAEList.append(MAEO)
                    if hisdata == 1:
                        # 1.随机抽取已训练过的数据集
                        TrainNum = len(BucketTrain[i])
                        a = random.sample(range(0, TrainNum), math.ceil(0.2*TrainNum))
                        Tra = [BucketTrain[i][j] for j in a]
                        Tra.extend(BucketTest[i])
                    elif hisdata == 2:
                        # 2.所有数据
                        Tra = BucketTrain[i]
                        Tra.extend(BucketTest[i])
                    elif hisdata == 3:
                        # 3.不添加原始训练集
                        Tra = BucketTest[i]
                    # ------------------------------------------------------------------------------------
                    # while 1:
                    #     x = []
                    #     y = []
                    #     y2 = []
                    #     first = []
                    #     countH = 0
                    #     for center in MonthCenter.keys():
                    #         if first == []:
                    #             first = center
                    #             last = center
                    #         x.append(center)
                    #         a = np.linalg.norm(np.array(MonthCenter[first]) - np.array(MonthCenter[center]))
                    #         a2 = np.linalg.norm(np.array(MonthCenter[last]) - np.array(MonthCenter[center]))
                    #         y.append(a)
                    #         y2.append(a2)
                    #         if a > max(y2):
                    #             countH += 1
                    #         else:
                    #             countH = 0
                    #         last = center
                    #     plt.plot(x, y, ls='-')
                    #     plt.plot(x, y2, ls='-')
                    #     plt.show()
                    #     if countH > 11:
                    #         MonthCenter.pop(first)
                    #         MonthTrace.pop(first)
                    #     else:
                    #         break
                    # historyTrain = []
                    # for mt in MonthTrace:
                    #     historyTrain.extend(MonthTrace[mt])
                    # a = random.sample(range(0, len(historyTrain)), math.ceil(0.2 * len(historyTrain)))#len(BucketAllTest[i])
                    # Tra = [historyTrain[j] for j in a]  # 3.历史数据随机抽取
                    # Tra.extend(historyTrain)  # 2.全部数据
                    # -----------------------------------------------------------------------------------
                    # Tra.extend(BucketAllTest[i])  # 1.仅用新数据
                    # 重新特征选择-------------------------------------------------------
                    # ai, aiMAE = FS.AllFLightboost(DR.Train, DR.Train, DR.header, [attribute[j]-3 for j in range(len(attribute))], 2)
                    # aii = len(ai)-1
                    # for i in range(len(ai)-1):
                    #     if aiMAE[i] - aiMAE[i+1] > 0.1:
                    #         aii = i
                    # FR = [aiMAE[aii], [DR.header[i] for i in ai[0:aii]], ai[0:aii]]
                    # if FR[2] != dataFR['index']:#重构模型
                    #     code(FR, dataFR, attribute)
                    #     feature = dataFR['index'][-1]
                    #     epoch = 200
                    #     Train_batch = P.NoFill(historyTrain, feature, task, batchSize)
                    #     model = method
                    # else:
                    if method == 'NoFill':
                        epoch = 20
                    elif method == 'Transformer':
                        epoch = 20
                    model = MR[i]
                    Train_batch = P.NoFill(Tra, feature, task, batchSize)
                    # ------------------------------------------------------------------
                    if method == 'NoFill':
                        MRU = M.trian(Train_batch, Test_X, Test_Y, epoch, preType, -2, 32, 1, model, dataFR, isEarly=0)
                        MR[i] = MRU[0]
                    elif method == 'multiset':
                        if len(Buckets) == 1:
                            MRU = multiset.update(DR.AllData, Tra, MR[1], MR[-1], 'multiset', feature, isMul=1)  # 长度分桶
                            MR[-1] = MRU[2]
                            MR[1] = MRU[0]
                        else:
                            MRU = multiset.trainBucket(DR.AllData, Tra, BucketTest[i], method, feature, MR[-2], MR[-1], isMul=1)  # 周期分桶
                            MR[i] = MRU[0]
                    elif method == 'Transformer':
                        MRU = M.trianT(Train_batch, Test_X, Test_Y, epoch, preType, mcl, model, dataFR)
                        MR[i] = MRU[0]
                    BucketTrain[i].extend(BucketAllTest[i])
                    adwin.reset()
                    for train in BucketTrain[i]:#Tra:# z in range(len(BucketTrain[i])-10000, len(BucketTrain[i])):#
                        Train_X = []
                        Train_Y = []
                        for line in train:#BucketTrain[i][z]:#
                            temp = []
                            for j in feature:
                                temp.append(line[j])
                            Train_X.append(temp)
                            Train_Y.append(line[task])
                        Train_X = torch.Tensor([[Train_X]])
                        Train_Y = torch.Tensor([[Train_Y]])
                        if method == 'NoFill':
                            MAE, countg, _ = M.test(Train_X, Train_Y, MR[i], preType, dataFR, len(dataFR['index'][-1]))
                        elif method == 'multiset':
                            MAE, _ = multiset.test(DR.AllData, BucketTrain[i], MR[1], MR[-1], 'multiset', feature, isMul=1)
                            MAE = [MAE.__float__()]
                        elif method == 'Transformer':
                            MAE, countg, _ = M.testT(Train_X, Train_Y, MR[i], preType, dataFR)
                        adwin.add_element(MAE[0])
                    BucketTest[i] = []
                    BucketAllTest[i] = []
                    flag = 0
                Train_X = []
                Train_Y = []
                for line in test:
                    temp = []
                    for j in feature:
                        temp.append(line[j])
                    Train_X.append(temp)
                    Train_Y.append(line[task])
                Train_X = torch.Tensor([[Train_X]])
                Train_Y = torch.Tensor([[Train_Y]])
                if method == 'NoFill':
                    MAE, countg, represents = M.test(Train_X, Train_Y, MR[i], preType, dataFR, len(dataFR['index'][-1]))
                elif method == 'Transformer':
                    MAE, countg, represents = M.testT(Train_X, Train_Y, MR[i], preType, dataFR)
                adwin.add_element(MAE[0])
                BucketAllTest[i].append(test)
    timeE = time.time()
    print('更新集:', hisdata, '增量:MAE，MAE:', [MAEsum[j] / count for j in range(len(MAEsum))], timeE - timeS, 's')
    return MAEList, MAEListI

def updateNew1(MR, method, dataFR, attribute, Buckets, Time, datas = 0, model = None, mcl = None, hisdata = 1, DEL = 0):
    dt = -1  # -1
    count = 0
    feature = dataFR['index'][-1]
    # 按月分割represents，画图分析
    MonthTrace = {}
    MonthCenter = {}
    MonthTrace[str(DR.Train[0][dt][-4])+str(DR.Train[0][dt][-8])] = []
    for trace in DR.Train:
        if str(trace[dt][-4]) + str(trace[dt][-8]) not in MonthTrace.keys():
            trainX = []
            trainY = []
            for line in MonthTrace[MonthKey]:
                traX = []
                traY = []
                for line2 in line:
                    temp = []
                    for i in feature:
                        temp.append(line2[i])
                    traX.append(temp)
                traY.append(line2[task])
                trainX.append(torch.Tensor([traX]))
                trainY.append(torch.Tensor([traY]))
            if method == 'NoFill':
                MAE, countg, represents = M.test(trainX, trainY, MR[-1], preType, dataFR, len(dataFR['index'][-1]))
                X = torch.zeros(len(represents), represents[0][0].shape[2])
                for j in range(len(represents)):
                    X[j, :] = represents[j][0].reshape(represents[0][0].shape[2])
                X[j, :] = represents[j][0].reshape(represents[0][0].shape[2])
            elif method == 'Transformer':
                MAE, countg, represents = M.testT(trainX, trainY, MR[-1], preType, dataFR)
                X = torch.zeros(len(represents), represents[0].shape[1])
                for j in range(len(represents)):
                    X[j, :] = represents[j].reshape(represents[0].shape[1])
                X[j, :] = represents[j].reshape(represents[0].shape[1])
            # trainX = []
            # trainY = []
            MonthTrace[str(trace[dt][-4]) + str(trace[dt][-8])] = []
            X = X.detach().numpy()
            center = [np.mean(X[:,j]) for j in range(X.shape[1])]
            MonthCenter[MonthKey] = center
        MonthKey = str(trace[dt][-4]) + str(trace[dt][-8])
        MonthTrace[str(trace[dt][-4]) + str(trace[dt][-8])].append(trace)

        # traX = []
        # traY = []
        # for line in trace:
        #     temp = []
        #     for i in feature:
        #         temp.append(line[i])
        #     traX.append(temp)
        # traY.append(line[task])
        # trainX.append(torch.Tensor([traX]))
        # trainY.append(torch.Tensor([traY]))
    # 计算原始数据总质心
    LCenter = []
    countC = 0
    for j in MonthCenter.keys():
        if LCenter == []:
            for z in MonthCenter[j]:
                LCenter.append(z * len(MonthTrace[j]))
                countC += len(MonthTrace[j])
        else:
            for z in range(len(MonthCenter[j])):
                LCenter[z] = LCenter[z] + MonthCenter[j][z] * len(MonthTrace[j])
    for z in range(len(LCenter)):
        LCenter[z] = LCenter[z] / countC
    # 判断丢弃无用历史数据
    if DEL == 1:
        while 1:
            x = []
            y = []
            y2 = []
            first = []
            countH = 0
            for center in MonthCenter.keys():
                if first == []:
                    first = center
                    last = center
                x.append(center)
                a = np.linalg.norm(np.array(MonthCenter[first]) - np.array(MonthCenter[center]))
                a2 = np.linalg.norm(np.array(MonthCenter[last]) - np.array(MonthCenter[center]))
                y.append(a)
                y2.append(a2)
                if a > max(y2):
                    countH += 1
                else:
                    countH = 0
                last = center
            plt.plot(x, y, ls='-')
            plt.plot(x, y2, ls='-')
            plt.show()
            if countH > 11:
                MonthCenter.pop(first)
                MonthTrace.pop(first)
            else:
                break
    # ----------------------------------------------------------------------
    MAEsum = [0 for x in feature]
    BucketTest = [[] for x in Buckets]
    BucketTrain = copy.deepcopy(DR.BucketTrain)
    MAEList = []
    MAEListI = []
    flag = 0
    flagC = 0
    timeS = time.time()
    for test in DR.Test:
        if str(test[dt][-4]) + str(test[dt][-8]) not in MonthTrace.keys():
            trainX = []
            trainY = []
            for line in MonthTrace[MonthKey]:
                traX = []
                traY = []
                for line2 in line:
                    temp = []
                    for j in feature:
                        temp.append(line2[j])
                    traX.append(temp)
                traY.append(line2[task])
                trainX.append(torch.Tensor([traX]))
                trainY.append(torch.Tensor([traY]))
            if method == 'NoFill':
                MAE, countg, represents = M.test(trainX, trainY, MR[0], preType, dataFR, len(dataFR['index'][-1]))
                X = torch.zeros(len(represents), represents[0][0].shape[2])
                for j in range(len(represents)):
                    X[j, :] = represents[j][0].reshape(represents[0][0].shape[2])
                X[j, :] = represents[j][0].reshape(represents[0][0].shape[2])
            elif method == 'Transformer':
                MAE, countg, represents = M.testT(trainX, trainY, MR[0], preType, dataFR)
                X = torch.zeros(len(represents), represents[0].shape[1])
                for j in range(len(represents)):
                    X[j, :] = represents[j].reshape(represents[0].shape[1])
                X[j, :] = represents[j].reshape(represents[0].shape[1])
            MonthTrace[str(test[dt][-4]) + str(test[dt][-8])] = []
            X = X.detach().numpy()
            MonthCenter[MonthKey] = [np.mean(X[:,j]) for j in range(X.shape[1])]
            # 判断丢弃无用历史数据
            if DEL == 1:
                while 1:
                    x = []
                    y = []
                    y2 = []
                    first = []
                    countH = 0
                    for center in MonthCenter.keys():
                        if first == []:
                            first = center
                            last = center
                        x.append(center)
                        a = np.linalg.norm(np.array(MonthCenter[first]) - np.array(MonthCenter[center]))
                        a2 = np.linalg.norm(np.array(MonthCenter[last]) - np.array(MonthCenter[center]))
                        y.append(a)
                        y2.append(a2)
                        if a > max(y2):
                            countH += 1
                        else:
                            countH = 0
                        last = center
                    if countH > 11:
                        plt.plot(x, y, ls='-')
                        plt.plot(x, y2, ls='-')
                        plt.show()
                        MonthCenter.pop(first)
                        MonthTrace.pop(first)
                    else:
                        break
            # 计算新数据总质心
            CCenter = []
            countC = 0
            for j in MonthCenter.keys():
                if CCenter == []:
                    for z in MonthCenter[j]:
                        CCenter.append(z * len(MonthTrace[j]))
                        countC += len(MonthTrace[j])
                else:
                    for z in range(len(MonthCenter[j])):
                        CCenter[z] = CCenter[z] + MonthCenter[j][z] * len(MonthTrace[j])
            for z in range(len(CCenter)):
                CCenter[z] = CCenter[z] / countC
            # 判断是否进行特征重选
            a = np.linalg.norm(np.array(LCenter) - np.array(CCenter))
            print('质心距离:',a)
            if a > 2:
                flagC = 1
        MonthKey = str(test[dt][-4]) + str(test[dt][-8])
        MonthTrace[str(test[dt][-4]) + str(test[dt][-8])].append(test)
        # --------------------------------------------------------------------------------------------------
        for i in range(len(Buckets)):
            if test[dt][-8] in Buckets[i][0] and test[dt][-7] in Buckets[i][1] and test[dt][-6] in Buckets[i][2]:
                if isinstance(Time, int) and len(BucketTest[i]) == Time:
                    flag = 1
                elif Time == 'month' and BucketTest[i] != [] and test[dt][-8] != BucketTest[i][-1][dt][-8]:
                    flag = 1
                elif Time == 'compre' and len(BucketTest[i]) >= datas and test[dt][-8] != BucketTest[i][-1][dt][-8]:
                    flag = 1
                if flag == 1:
                    # print(len(BucketTest[i]))
                    Test_X, Test_Y = P.changeLen(BucketTest[i], feature, task, 1)
                    if method == 'NoFill':
                        MAEO, _, _ = M.test(Test_X, Test_Y, MR[-1], preType, dataFR, len(feature))
                        MAE, countg, _ = M.test(Test_X, Test_Y, MR[i], preType, dataFR, len(feature))
                        count += countg
                        MAEsum = [MAEsum[j] + MAE[j] * countg for j in range(len(MAE))]
                    elif method == 'multiset':
                        if len(Buckets)==1:
                            MAEO, _ = multiset.test(DR.AllData, BucketTest[i], MR[3], MR[0], 'multiset', feature, isMul=1)
                            MAE, countg = multiset.test(DR.AllData, BucketTest[i], MR[2], MR[0], 'multiset', feature, isMul=1)
                        else:
                            MAEO = 0
                            MAE, countg = multiset.testBucket(DR.AllData, BucketTest[i], MR[-2], MR[i], 'multiset', feature, isMul=1)
                        count += countg
                        MAEsum = [MAEsum[0] + MAE.__float__() * countg]
                    elif method == 'Transformer':
                        MAE, countg, _ = M.testT(Test_X, Test_Y, MR[i], preType, dataFR)
                        count += countg
                        MAEsum = [MAEsum[i] + MAE[i] * countg for i in range(len(MAE))]
                    MAEListI.append(MAE)
                    BucketTrain[i] = []
                    for mt in MonthTrace:
                        BucketTrain[i].extend(MonthTrace[mt])
                    if hisdata == 1:
                        # 1.随机抽取已训练过的数据集
                        TrainNum = len(BucketTrain[i])
                        a = random.sample(range(0, TrainNum), math.ceil(0.2 * TrainNum))
                        Tra = [BucketTrain[i][j] for j in a]
                        Tra.extend(BucketTest[i])
                    elif hisdata == 2:
                        # 2.所有数据
                        Tra = BucketTrain[i]
                        # Tra.extend(BucketTest[i])
                    elif hisdata == 3:
                        # 3.不添加原始训练集
                        Tra = BucketTest[i]
                    # 特征重选
                    if flagC == 1:
                        flagC = 0
                        TrainF, ValF = train_test_split(BucketTrain[i], test_size=0.2, random_state=20)#
                        FR = FS.LightGBM(TrainF, ValF, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
                        # 模型重构
                        if len(FR[2]) != len(dataFR['index'][0]) or (FR[2] != dataFR['index']).all():
                            print('重构+1')
                            dataFR = code(FR, dataFR, attribute)
                            feature = dataFR['index'][-1]
                            epoch = 100 #200
                            Train_batch = P.NoFill(Tra, feature, task, batchSize)#BucketTrain[i]
                            Test_X, Test_Y = P.changeLen(BucketTest[i], feature, task, 1)
                            if method == 'Transformer':
                                model = 'tran'
                            else:
                                model = method
                        else:
                            model = MR[i]
                            Train_batch = P.NoFill(Tra, feature, task, batchSize)
                            epoch = 20
                    else:
                        model = MR[i]
                        Train_batch = P.NoFill(Tra, feature, task, batchSize)
                        epoch = 20
                    if method == 'NoFill':
                        MRU = M.trian(Train_batch, Test_X, Test_Y, epoch, preType, -2, 32, 1, model, dataFR, isEarly=0)
                        MR[i] = MRU[0]
                    elif method == 'multiset':
                        if len(Buckets) == 1:
                            MRU = multiset.update(DR.AllData, Tra, MR[2], MR[0], 'multiset', feature, isMul=1) # 长度分桶
                            MR[0] = MRU[2]
                            MR[2] = MRU[0]
                        else:
                            MRU = multiset.trainBucket(DR.AllData, Tra, BucketTest[i], method, feature, MR[-2], MR[-1], isMul=1)# 周期分桶
                            MR[i] = MRU[0]
                    elif method == 'Transformer':
                        MRU = M.trianT(Train_batch, Test_X, Test_Y, epoch, preType, mcl, model, dataFR)
                        MR[i] = MRU[0]
                    MonthTrace = {}
                    MonthCenter = {}
                    MonthTrace[str(BucketTrain[i][0][dt][-4]) + str(BucketTrain[i][0][dt][-8])] = []
                    for trace in BucketTrain[i]:
                        if str(trace[dt][-4]) + str(trace[dt][-8]) not in MonthTrace.keys():
                            trainX = []
                            trainY = []
                            for line in MonthTrace[MonthKey]:
                                traX = []
                                traY = []
                                for line2 in line:
                                    temp = []
                                    for j in feature:
                                        temp.append(line2[j])
                                    traX.append(temp)
                                traY.append(line2[task])
                                trainX.append(torch.Tensor([traX]))
                                trainY.append(torch.Tensor([traY]))
                            if method == 'NoFill':
                                MAE, countg, represents = M.test(trainX, trainY, MR[i], preType, dataFR, len(dataFR['index'][-1]))
                                X = torch.zeros(len(represents), represents[0][0].shape[2])
                                for j in range(len(represents)):
                                    X[j, :] = represents[j][0].reshape(represents[0][0].shape[2])
                                X[j, :] = represents[j][0].reshape(represents[0][0].shape[2])
                            elif method == 'Transformer':
                                MAE, countg, represents = M.testT(trainX, trainY, MR[i], preType, dataFR)
                                X = torch.zeros(len(represents), represents[0].shape[1])
                                for j in range(len(represents)):
                                    X[j, :] = represents[j].reshape(represents[0].shape[1])
                                X[j, :] = represents[j].reshape(represents[0].shape[1])
                            MonthTrace[str(trace[dt][-4]) + str(trace[dt][-8])] = []
                            X = X.detach().numpy()
                            center = [np.mean(X[:, j]) for j in range(X.shape[1])]
                            MonthCenter[MonthKey] = center
                        MonthKey = str(trace[dt][-4]) + str(trace[dt][-8])
                        MonthTrace[str(trace[dt][-4]) + str(trace[dt][-8])].append(trace)
                    # 计算原始数据总质心
                    if model == 'tran' or model == method:
                        LCenter = []
                        countC = 0
                        for j in MonthCenter.keys():
                            if LCenter == []:
                                for z in MonthCenter[j]:
                                    LCenter.append(z * len(MonthTrace[j]))
                                    countC += len(MonthTrace[j])
                            else:
                                for z in range(len(MonthCenter[j])):
                                    LCenter[z] = LCenter[z] + MonthCenter[j][z] * len(MonthTrace[j])
                        for z in range(len(LCenter)):
                            LCenter[z] = LCenter[z] / countC
                    # BucketTrain[i].extend(BucketTest[i])
                    BucketTest[i] = []
                    flag = 0
                BucketTest[i].append(test)
    timeE = time.time()
    print('更新集:', hisdata, '增量:', Time, "MAE:", [MAEsum[j]/count for j in range(len(MAEsum))], timeE - timeS, 's')
    del MR
    return MAEList, MAEListI

def updateNew2(MR, method, dataFR, attribute, Buckets, Time, datas = 0, model = None, mcl = None, hisdata = 1, DEL = 0):
    dt = -1  # -1
    count = 0
    feature = dataFR['index'][-1]
    MAEsum = [0 for x in feature]
    BucketTest = [[] for x in Buckets]
    BucketTrain = copy.deepcopy(DR.BucketTrain)
    MAEListI = []
    flag = 0
    flagC = 0
    timeS = time.time()
    for test in DR.Test:
        for i in range(len(Buckets)):
            if test[dt][-8] in Buckets[i][0] and test[dt][-7] in Buckets[i][1] and test[dt][-6] in Buckets[i][2]:
                if isinstance(Time, int) and len(BucketTest[i]) == Time:
                    flag = 1
                elif Time == 'month' and BucketTest[i] != [] and test[dt][-8] != BucketTest[i][-1][dt][-8]:
                    flag = 1
                elif Time == 'compre' and len(BucketTest[i]) >= datas and test[dt][-8] != BucketTest[i][-1][dt][-8]:
                    flag = 1
                if flag == 1:
                    # print(len(BucketTest[i]))
                    Test_X, Test_Y = P.changeLen(BucketTest[i], feature, task, 1)
                    if method == 'NoFill':
                        MAEO, _, _ = M.test(Test_X, Test_Y, MR[-1], preType, dataFR, len(feature))
                        MAE, countg, _ = M.test(Test_X, Test_Y, MR[i], preType, dataFR, len(feature))
                        count += countg
                        MAEsum = [MAEsum[j] + MAE[j] * countg for j in range(len(MAE))]
                    elif method == 'multiset':
                        if len(Buckets)==1:
                            MAEO, _ = multiset.test(DR.AllData, BucketTest[i], MR[3], MR[0], 'multiset', feature, isMul=1)
                            MAE, countg = multiset.test(DR.AllData, BucketTest[i], MR[2], MR[0], 'multiset', feature, isMul=1)
                        else:
                            MAEO = 0
                            MAE, countg = multiset.testBucket(DR.AllData, BucketTest[i], MR[-2], MR[i], 'multiset', feature, isMul=1)
                        count += countg
                        MAEsum = [MAEsum[0] + MAE.__float__() * countg]
                    elif method == 'Transformer':
                        MAE, countg, _ = M.testT(Test_X, Test_Y, MR[i], preType, dataFR)
                        count += countg
                        MAEsum = [MAEsum[i] + MAE[i] * countg for i in range(len(MAE))]
                    MAEListI.append(MAE)
                    if hisdata == 1:
                        # 1.随机抽取已训练过的数据集
                        TrainNum = len(BucketTrain[i])
                        a = random.sample(range(0, TrainNum), math.ceil(0.2 * TrainNum))
                        Tra = [BucketTrain[i][j] for j in a]
                        Tra.extend(BucketTest[i])
                    elif hisdata == 2:
                        # 2.所有数据
                        Tra = BucketTrain[i]
                        Tra.extend(BucketTest[i])
                    elif hisdata == 3:
                        # 3.不添加原始训练集
                        Tra = BucketTest[i]
                    # 特征重选
                    BucketTrain[i].extend(BucketTest[i])
                    # TrainF, ValF = train_test_split(BucketTrain[i], test_size=0.2, random_state=20)#
                    FR = FS.LightGBM(BucketTrain[i], BucketTrain[i], DR.header, [attribute[i] - 3 for i in range(len(attribute))])
                    # 模型重构
                    flagF = 0
                    for f in range(int(len(FR[2])/2)):
                        if FR[2][f] not in dataFR['index'][0]:
                            flagF = 1
                            break
                    if flagF == 1:
                        print('重构+1')
                        dataFR = code(FR, dataFR, attribute)
                        feature = dataFR['index'][-1]
                        epoch = 200 #100
                        Train_batch = P.NoFill(BucketTrain[i], feature, task, batchSize)#BucketTrain[i]
                        Test_X, Test_Y = P.changeLen(BucketTest[i], feature, task, 1)
                        if method == 'Transformer':
                            model = 'tran'
                        else:
                            model = method
                    else:
                        model = MR[i]
                        Train_batch = P.NoFill(Tra, feature, task, batchSize)
                        epoch = 20
                    if method == 'NoFill':
                        MRU = M.trian(Train_batch, Test_X, Test_Y, epoch, preType, -2, 32, 1, model, dataFR, isEarly=0)
                        MR[i] = MRU[0]
                    elif method == 'multiset':
                        if len(Buckets) == 1:
                            MRU = multiset.update(DR.AllData, Tra, MR[2], MR[0], 'multiset', feature, isMul=1) # 长度分桶
                            MR[0] = MRU[2]
                            MR[2] = MRU[0]
                        else:
                            MRU = multiset.trainBucket(DR.AllData, Tra, BucketTest[i], method, feature, MR[-2], MR[-1], isMul=1)# 周期分桶
                            MR[i] = MRU[0]
                    elif method == 'Transformer':
                        MRU = M.trianT(Train_batch, Test_X, Test_Y, epoch, preType, mcl, model, dataFR)
                        MR[i] = MRU[0]
                    BucketTest[i] = []
                    flag = 0
                BucketTest[i].append(test)
    timeE = time.time()
    print('更新集:', hisdata, '增量:', Time, "MAE:", [MAEsum[j]/count for j in range(len(MAEsum))], timeE - timeS, 's')
    del MR
    return MAEListI

def updateNew(MR, method, dataFR, Buckets, Time, datas = 0, model = None, mcl = None, hisdata = 1):
    dt = -1  # -1
    count = 0
    feature = dataFR['index'][-1]
    MAEsum = [0 for x in feature]
    BucketTest = [[] for x in Buckets]
    BucketTrain = copy.deepcopy(DR.BucketTrain)
    MAEList = []
    MAEListI = []
    flag = 0
    timeS = time.time()
    for test in DR.Test:
        for i in range(len(Buckets)):
            if test[dt][-8] in Buckets[i][0] and test[dt][-7] in Buckets[i][1] and test[dt][-6] in Buckets[i][2]:
                if isinstance(Time, int) and len(BucketTest[i]) == Time:
                    flag = 1
                elif Time == 'month' and BucketTest[i] != [] and test[dt][-8] != BucketTest[i][-1][dt][-8]:
                    flag = 1
                elif Time == 'compre' and len(BucketTest[i]) >= datas and test[dt][-8] != BucketTest[i][-1][dt][-8]:
                    flag = 1
                if flag == 1:
                    # print(len(BucketTest[i]))
                    Test_X, Test_Y = P.changeLen(BucketTest[i], feature, task, 1)
                    if method == 'NoFill':
                        MAEO, _, _ = M.test(Test_X, Test_Y, MR[-1], preType, dataFR, len(feature))
                        MAE, countg, _ = M.test(Test_X, Test_Y, MR[i], preType, dataFR, len(feature))
                        count += countg
                        MAEsum = [MAEsum[j] + MAE[j] * countg for j in range(len(MAE))]
                    elif method == 'multiset':
                        if len(Buckets)==1:
                            MAEO, _ = multiset.test(DR.AllData, BucketTest[i], MR[0], MR[-1], 'multiset', feature, isMul=1)
                            MAE, countg = multiset.test(DR.AllData, BucketTest[i], MR[1], MR[-1], 'multiset', feature, isMul=1)
                        else:
                            MAEO = 0
                            MAE, countg = multiset.testBucket(DR.AllData, BucketTest[i], MR[-2], MR[i], 'multiset', feature, isMul=1)
                        MAE = [MAE.__float__()]
                        MAEO = [MAEO.__float__()]
                        count += countg
                        MAEsum = [MAEsum[0] + MAE[0] * countg]
                    elif method == 'Transformer':
                        MAEO, _, _ = M.testT(Test_X, Test_Y, MR[-1], preType, dataFR)
                        MAE, countg, _ = M.testT(Test_X, Test_Y, MR[i], preType, dataFR)
                        count += countg
                        MAEsum = [MAEsum[i] + MAE[i] * countg for i in range(len(MAE))]
                    MAEListI.append(MAE)
                    MAEList.append(MAEO)
                    if hisdata == 1:
                        # 1.随机抽取已训练过的数据集
                        TrainNum = len(BucketTrain[i])
                        a = random.sample(range(0, TrainNum), math.ceil(0.2*TrainNum))
                        Tra = [BucketTrain[i][j] for j in a]
                        Tra.extend(BucketTest[i])
                    elif hisdata == 2:
                        # 2.所有数据
                        Tra = BucketTrain[i]
                        Tra.extend(BucketTest[i])
                    elif hisdata == 3:
                        # 3.不添加原始训练集
                        Tra = BucketTest[i]
                    Train_batch = P.NoFill(Tra, feature, task, batchSize)
                    if method == 'NoFill':
                        MRU = M.trian(Train_batch, Test_X, Test_Y, 20, preType, -2, 32, 1, MR[i], dataFR, isEarly=0)
                        MR[i] = MRU[0]
                    elif method == 'multiset':
                        if len(Buckets) == 1:
                            MRU = multiset.update(DR.AllData, Tra, MR[1], MR[-1], 'multiset', feature, isMul=1) # 长度分桶
                            MR[-1] = MRU[2]
                            MR[1] = MRU[0]
                        else:
                            MRU = multiset.trainBucket(DR.AllData, Tra, BucketTest[i], method, feature, MR[-2], MR[-1], isMul=1)# 周期分桶
                            MR[i] = MRU[0]
                    elif method == 'Transformer':
                        MRU = M.trianT(Train_batch, Test_X, Test_Y, 20, preType, mcl, MR[i], dataFR)
                        MR[i] = MRU[0]
                    BucketTrain[i].extend(BucketTest[i])
                    BucketTest[i] = []
                    flag = 0
                BucketTest[i].append(test)
    timeE = time.time()
    print('更新集:', hisdata, '增量:', Time, "MAE:", [MAEsum[j]/count for j in range(len(MAEsum))], timeE - timeS, 's')
    del MR
    return MAEList, MAEListI

if __name__ == "__main__":
    import numpy as np

    # data_stream = np.random.randint(2, size=2000)
    # for i in range(999, 1500):
    #     data_stream[i] = 0
    num = 30000
    data_stream = np.random.randint(2, size=30000)
    for i in range(9999, 20000):
        data_stream[i] = np.random.randint(0, high=3)
    for i in range(19999, 30000):
        data_stream[i] = np.random.randint(4, high=8)

    # print('PageHinkley')
    # from skmultiflow.drift_detection import PageHinkley
    # ph = PageHinkley()
    # for i in range(num):
    #     ph.add_element(data_stream[i])
    #     if ph.detected_change():
    #         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    #
    # print('DDM')
    # from skmultiflow.drift_detection import DDM
    # ddm = DDM()
    # for i in range(num):
    #     ddm.add_element(data_stream[i])
    #     if ddm.detected_change():
    #         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    #
    # print('EDDM')
    # from skmultiflow.drift_detection.eddm import EDDM
    # eddm = EDDM()
    # for i in range(num):
    #     eddm.add_element(data_stream[i])
    #     if eddm.detected_change():
    #         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    #
    # print('HDDM_A')
    # from skmultiflow.drift_detection.hddm_a import HDDM_A
    # hddm_a = HDDM_A()
    # for i in range(num):
    #     hddm_a.add_element(data_stream[i])
    #     if hddm_a.detected_change():
    #         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    #
    # print('HDDM_W')
    # from skmultiflow.drift_detection.hddm_w import HDDM_W
    # hddm_w = HDDM_W()
    # for i in range(num):
    #     hddm_w.add_element(data_stream[i])
    #     if hddm_w.detected_change():
    #         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    #
    # print('ADWIN')
    # from skmultiflow.drift_detection.adwin import ADWIN
    # adwin = ADWIN()
    # for i in range(num):
    #     adwin.add_element(data_stream[i])
    #     if adwin.detected_change():
    #         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))

    print('KSWIN')
    from skmultiflow.drift_detection import KSWIN
    kswin = KSWIN(alpha=0.001,window_size=1000,stat_size=300)
    for i in range(num):
        kswin.add_element(data_stream[i])
        if kswin.detected_change():
            print("\rIteration {}".format(i))
            kswin.reset()

    print('KSWIN')
    kswin = KSWIN(window_size=1000, stat_size=300)
    for i in range(num):
        kswin.add_element(data_stream[i])
        if kswin.detected_change():
            print("\rIteration {}".format(i))

    print('KSWIN')
    kswin1 = KSWIN()  #
    for i in range(num):
        kswin1.add_element(data_stream[i])
        if kswin1.detected_change():
            print("\rIteration {}".format(i))