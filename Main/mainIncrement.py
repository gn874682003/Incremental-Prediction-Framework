import copy
import math
import random
import time
import warnings
warnings.filterwarnings('ignore')
from BPP_Frame.Log.DataRecord import DataRecord as DR
import BPP_Frame.Log.LogConvert as LC
import BPP_Frame.Log.LogAnalysis as LA
import BPP_Frame.Log.DivideData as DD
import BPP_Frame.Log.Prefix as P
import BPP_Frame.Feature.FeatureSel as FS
import BPP_Frame.Method.My.Model as M
import BPP_Frame.Code.word2vec as w2v
import BPP_Frame.Method.AETS.multiset as multiset
import torch
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def LogDividid(method, split, time = None):
    DR.Convert, DR.header, DR.ConvertReflact, maxA, maxR = LC.LogC(eventlog, attribute)  # 需转换属性值的下标
    # 特征类别编号 0 活动；1 分类静态特征；2 分类动态特征；3 数值静态特征；4 数值动态特征
    DR.State = []
    DR.State.append(0)
    for i in range(1, len(DR.Convert[0]) - 12):
        if i + 3 in attribute:
            DR.State.append(1)
        else:
            DR.State.append(3)
    for i in range(6):
        DR.State.append(4)
    # 数据集划分
    DR.Train, DR.Test, DR.AllData = DD.DiviData(DR.Convert, DR.State)
    # 轨迹按时间排序，以给定比例和周期划分数据集
    DR.Train, DR.Test, DR.Tests = DD.DiviDataByTime(DR.AllData, split, time)  # week day

def code(FR):
    # 活动编码 训练
    DR.Train_XA, DR.Train_YA = P.cutPrefixBy(DR.AllData, [0],label=-3,batchSize=20,LEN=3)#[FR[2][0]]
    EmbA, ACCE = w2v.word2vec(DR.Train_XA, DR.Train_YA, DR.ConvertReflact)
    # dataFE = {'0':EmbA.detach().numpy(),'name':FE[1],'index':FE[2],'state':[DR.State[i] for i in FE[2]],'result':FE[0],'ACCE':ACCE}
    # dataFD = {'0':EmbA.detach().numpy(),'name':FD[1],'index':FD[2],'state':[DR.State[i] for i in FD[2]],'result':FD[0]}
    dataFR = {'0':EmbA.detach().numpy(),'name':FR[1],'index':FR[2],'state':[DR.State[i] for i in FR[2]],'result':FR[0]}

    # 其他分类特征编码 随机初始化Embding
    for i in range(1,len(DR.Train[0][0])-3):
        if i+3 in attribute:
            # if len(DR.ConvertReflact[attribute.index(i+3)])>5:
            eim = 4#5
            olen = len(DR.ConvertReflact[attribute.index(i+3)])
            while olen > 16:#20
                olen /= 4
                eim += 4#5
            EmbS = nn.Embedding(len(DR.ConvertReflact[attribute.index(i+3)]), eim)
            X_tsne = TSNE(n_components=2, learning_rate=0.1).fit_transform(EmbS.weight.detach().numpy())
            for j in range(len(X_tsne)):
                plt.scatter(X_tsne[j, 0], X_tsne[j, 1])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(DR.header[i])
            plt.show()
            if i in dataFR['index']:#FR[2]
                dataFR[str(i)] = EmbS.weight.detach().numpy()
            # if i in dataFE['index']:
            #     dataFE[str(i)] = EmbS.weight.detach().numpy()
            # if i in dataFD['index']:
            #     dataFD[str(i)] = EmbS.weight.detach().numpy()
    return dataFR#, dataFE,dataFD

def train(method, feature, Buckets, model = None, mcl = None):
    timeS = time.time()
    if method == 'diffW':
        DR.Train_X, DR.Train_Y = P.diffWindow(DR.Train, feature, task, batchSize, LEN)
        DR.Test_X, DR.Test_Y = P.changeLen(DR.Test.copy(), feature, task, 1)
        MR = M.trianD(DR.Train_X, DR.Train_Y, DR.Test_X, DR.Test_Y, epoch, preType, len(feature), 32, 1, 'inn', dataFR)
    elif method == 'NoFill':
        DR.Train_batch = P.NoFill(DR.Train, feature, task, batchSize)
        DR.Test_X, DR.Test_Y = P.changeLen(DR.Test.copy(), feature, task, 1)
        MR = M.trian(DR.Train_batch, DR.Test_X, DR.Test_Y, epoch, preType, len(feature), 32, 1, model, dataFR)
    elif method == 'multiset':
        MR = multiset.train(DR.AllData, DR.Train, DR.Test, method, feature)
    elif method == 'Transformer':
        DR.Train_batch = P.NoFill(DR.Train, feature, task, batchSize)
        DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, feature, task, 1)
        dim = 36  # dataFR['0'].shape[1]
        # MR = M.trianPT1(DR.Train_batch, DR.Test_X, DR.Test_Y, 100, preType, len(feature), mcl, 'tran', dataFR)
        MR = M.trianPT(DR.Train_batch, DR.Test_X, DR.Test_Y, 100, preType, dim, mcl, 'tran', dataFR)
        # MR = M.trianT(DR.Train_batch, DR.Test_X, DR.Test_Y, epoch, preType, dim, 8*dim, 6, dim, 'tran', dataFR)#int(dim/5)
    timeE = time.time()
    print('不分桶、不增量——MAE：', MR[1], timeE - timeS, 's')
    if method == 'multiset':# or method == 'Transformer':
        DR.BucketTrain = [DR.Train]
        DR.BucketTest = [DR.Test]
        return MR
    timeS = time.time()
    sumMAE = [0 for x in feature]
    count = 0
    if len(Buckets) > 1:
        BucketTrain = [[] for x in Buckets]
        BucketTest = [[] for x in Buckets]
        BucketModel = []
        for tra in DR.Train:
            for i in range(len(Buckets)):
                if tra[0][-8] in Buckets[i][0] and tra[0][-7] in Buckets[i][1] and tra[0][-6] in Buckets[i][2]:
                    BucketTrain[i].append(tra)
        for tes in DR.Test:
            for i in range(len(Buckets)):
                if tes[0][-8] in Buckets[i][0] and tes[0][-7] in Buckets[i][1] and tes[0][-6] in Buckets[i][2]:
                    BucketTest[i].append(tes)
        for tra, tes in zip(BucketTrain, BucketTest):
            if tra != [] and tes != []:
                if method == 'diffW':
                    Train_X, Train_Y = P.diffWindow(tra, feature, task, batchSize, LEN)
                    Test_X, Test_Y = P.changeLen(tes, feature, task, 1)
                    if model == 'inn':
                        MR = M.trianD(Train_X, Train_Y, Test_X, Test_Y, epoch, preType, len(feature), 32, 1, MR[0], dataFR)
                    else:
                        MR = M.trianD(Train_X, Train_Y, Test_X, Test_Y, epoch, preType, 1, 32, 1, MR[0], dataFR)
                elif method == 'NoFill':
                    Train_batch = P.NoFill(tra, feature, task, batchSize)
                    Test_X, Test_Y = P.changeLen(tes, feature, task, 1)
                    if model == 'inn':
                        MRB = M.trian(Train_batch, Test_X, Test_Y, epoch, preType, len(feature), 32, 1, MR[0], dataFR)
                    else:
                        MRB = M.trian(Train_batch, Test_X, Test_Y, 50, preType, 1, 32, 1, MR[0], dataFR)
                elif method == 'multiset':
                    MR = multiset.train(DR.AllData, tra, tes, 'multiset', MR[0])
                elif method == 'Transformer':
                    Train_batch = P.NoFill(tra, feature, task, batchSize)
                    Test_X, Test_Y = P.changeLen(tes, feature, task, 1)
                    dim = 36 # 20dataFR['0'].shape[1]
                    MR = M.trianPT(Train_batch, Test_X, Test_Y, 100, preType, dim, mcl, 'tran', dataFR)
                count += len(tes)
                sumMAE = [sumMAE[x] + MRB[1][x] * len(tes) for x in range(len(MRB[1]))]
            elif tes != []:
                if method == 'diffW' or method == 'NoFill':
                    Test_X, Test_Y = P.changeLen(tes, feature, task, 1)
                    if model == 'inn':
                        MAE = M.test(Test_X, Test_Y, MR[0], preType, dataFR, 1, len(feature))
                    else:
                        MAE = M.test(Test_X, Test_Y, MR[0], preType, dataFR, 1, 1)
                elif method == 'multiset':
                    MAE, countg = multiset.test(DR.AllData, tes, MR[0], MR[2], 'multiset')
                elif method == 'Transformer':
                    Test_X, Test_Y = P.changeLen(tes, feature, task, 1)
                    MAE, _ = M.testPT(Test_X, Test_Y, MR[0], preType, dataFR, 1)
                count += len(tes)
                sumMAE = [sumMAE[x] + MAE[x] * len(tes) for x in range(len(MAE))]
            BucketModel.append(MRB[0])
        DR.BucketTrain = BucketTrain
        DR.BucketTest = BucketTest
        timeE = time.time()
        print('分桶、不增量——MAE：', [sumMAE[x]/count for x in range(len(sumMAE))], timeE - timeS, 's')
        BucketModel.extend([MR[0]])
        return BucketModel
    else:
        DR.BucketTrain = [DR.Train]
        DR.BucketTest = [DR.Test]
        return [MR[0]]

def update(MR, method, feature):
    count = 0
    MAEsum = [0 for i in feature]
    for test in DR.Tests:
        if len(DR.Tests[test]) == 1:
            continue
        if method == 'diffW' or method == 'NoFill':
            DR.Test_X, DR.Test_Y = P.changeLen(DR.Tests[test], feature, task, 1)
            MAE = M.test(DR.Test_X, DR.Test_Y, MR[0], preType, dataFR, 1, len(feature))
            # print(test, "预测MAE:", MAE[0])
            count += len(DR.Test_X)
            MAEsum = [MAEsum[i] + MAE[0][i] * len(DR.Test_X) for i in range(len(MAE[0]))]
        elif method == 'multiset':
            MAE, countg = multiset.test(DR.AllData, DR.Tests[test], MR[0], MR[2], 'multiset')
            # print(test, "预测MAE:", MAE)
            count += countg
            MAEsum = [MAEsum[i] + MAE * countg for i in range(len(MAEsum))]
        # 修改训练集,在已训练过的数据集中随机抽取
        TrainNum = len(DR.Train)
        a = random.sample(range(0, TrainNum), math.ceil(0.2*TrainNum))
        Tra = [DR.Train[i] for i in a]
        Tra.extend(DR.Tests[test])
        if method == 'diffW':
            DR.Train_X, DR.Train_Y = P.diffWindow(Tra, feature, task, batchSize, LEN)
            MR = M.trianD(DR.Train_X, DR.Train_Y, DR.Test_X, DR.Test_Y, 50, preType, len(feature), 32, 1, MR[0], dataFR)
        elif method == 'NoFill':
            DR.Train_batch = P.NoFill(Tra, feature, task, batchSize)
            MR = M.trian(DR.Train_batch, DR.Test_X, DR.Test_Y, 50, preType, len(feature), 32, 1, MR[0], dataFR)
        elif method == 'multiset':# 根据轨迹长度训练不同的模型
            MR = multiset.update(DR.AllData, Tra, MR[0], MR[2], 'multiset')

        DR.Train.extend(DR.Tests[test])
    print("MAE:", [MAEsum[i]/count for i in range(len(MAEsum))])

def updateNew(MR, method, feature, Buckets, Time, datas = 0, model = None, mcl = None):
    dt = -1 # 0
    count = 0
    MAEsum = [0 for x in feature]
    BucketTest = [[] for x in Buckets]
    BucketTrain = copy.deepcopy(DR.BucketTrain)
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
                    if method == 'diffW' or method == 'NoFill':
                        Test_X, Test_Y = P.changeLen(BucketTest[i], feature, task, 1)
                        if model == 'inn':
                            MAE = M.test(Test_X, Test_Y, MR[i], preType, dataFR, 1, len(feature))
                        else:
                            MAE = M.test(Test_X, Test_Y, MR[i], preType, dataFR, 1, 1)
                        count += len(Test_X)
                        MAEsum = [MAEsum[j] + MAE[j] * len(Test_X) for j in range(len(MAE))]
                    elif method == 'multiset':
                        MAE, countg = multiset.test(DR.AllData, BucketTest[i], MR[0], MR[2], 'multiset')
                        count += countg
                        MAEsum = [MAEsum[0] + MAE.__float__() * countg]
                    elif method == 'Transformer':
                        Test_X, Test_Y = P.changeLen(BucketTest[i], feature, task, 1)
                        MAE, _ = M.testPT(Test_X, Test_Y, MR[0], preType, dataFR, 1)
                        count += len(Test_X)
                        MAEsum = [MAEsum[i] + MAE[i] * len(Test_X) for i in range(len(MAE))]
                    # 修改训练集,在已训练过的数据集中随机抽取
                    TrainNum = len(BucketTrain[i])
                    a = random.sample(range(0, TrainNum), math.ceil(0.2*TrainNum))
                    Tra = [BucketTrain[i][j] for j in a]
                    Tra.extend(BucketTest[i])
                    # 不添加原始训练集
                    # Tra = BucketTest[i]
                    if method == 'diffW':
                        Train_X, Train_Y = P.diffWindow(Tra, feature, task, batchSize, LEN)
                        if model == 'rnn':
                            MRU = M.trianD(Train_X, Train_Y, Test_X, Test_Y, 20, preType, len(feature), 32, 1, MR[i], dataFR)
                        else:
                            MRU = M.trianD(Train_X, Train_Y, Test_X, Test_Y, 20, preType, 1, 32, 1, MR[i], dataFR)
                        MR[i] = MRU[0]
                    elif method == 'NoFill':
                        Train_batch = P.NoFill(Tra, feature, task, batchSize)
                        if model == 'rnn':
                            MRU = M.trian(Train_batch, Test_X, Test_Y, 50, preType, len(feature), 32, 1, MR[i], dataFR)
                        else:
                            MRU = M.trian(Train_batch, Test_X, Test_Y, 50, preType, 1, 32, 1, MR[i], dataFR)
                        MR[i] = MRU[0]
                    elif method == 'multiset':# 根据轨迹长度训练不同的模型,需修改
                        MR = multiset.update(DR.AllData, Tra, MR[0], MR[2], 'multiset')
                    elif method == 'Transformer':
                        Train_batch = P.NoFill(Tra, feature, task, batchSize)
                        MR = M.trianPT(Train_batch, DR.Test_X, DR.Test_Y, 20, preType, 36, mcl, MR[0], dataFR)
                    BucketTrain[i].extend(BucketTest[i])
                    BucketTest[i] = []
                    flag = 0
                BucketTest[i].append(test)
    timeE = time.time()
    print('分桶:', len(Buckets), '增量:', Time, "MAE:", [MAEsum[j]/count for j in range(len(MAEsum))], timeE - timeS, 's')

def verify(dataNameFR):
    # 验证
    dataFR = scio.loadmat(dataNameFR)
    # DR.Test_X, DR.Test_Y = P.changeLen(DR.Test, dataFR['index'][0], -1, 1)
    DR.Test_X, DR.Test_Y = P.diffWindow(DR.Test, dataFR['index'][0], -1, 1, 10)
    inn = M.INN(len(dataFR['index'][0]), 32, 1, dataFR, preType)
    print(sum(param.numel() for param in inn.parameters()))
    inn.load_state_dict(torch.load('model/MR_'+eventlog+'S.pkl'))
    MAE = M.TestMetricD(DR.Test_X, DR.Test_Y, inn, preType, dataFR, 1, len(dataFR['index'][0]))
    print(MAE)
    M.viewResultD(DR.Test_X, DR.Test_Y, inn, preType, dataFR, DR.ConvertReflact, attribute)

if __name__ == "__main__":
    # hd 3,4,5,6,7,8,9,10,11,12,13,14
    # BPIC2012 3,5
    # BPIC2015 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
    # hospital 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
    # Production_Data 3,4,5,6,7,8,9,10,11,12
    # RRT 3,5,6,7,9,10,11,12,13
    # AllData [3,4,5,6,8] NewScrapLogO NewScrapLog
    # AF,BF,CF [3,4,5]
    # eventlog= 'BPIC2012'
    # attribute = [3,5]
    # 属性转换
    EL = ['hd','BPIC2015_1','BPIC2015_2','BPIC2015_3','BPIC2015_4','BPIC2015_5','RRT','hospital','NewScrapLog']#
    #'CF','Production_Data','BPIC2012','actCase',[3,4,5],[3,4,5,6,7,8,9,10,11,12],[3,5],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
    Att = [[3,4,5,6,7,8,9,10,11,12,13,14],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],#
           [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
           [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
           [3,5,6,7,9,10,11,12,13],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],[3,4,5,6,7]]#
    for eventlog,attribute in zip(EL, Att):
        print(eventlog)
        # 1.日志划分为初始训练集和增量训练集，测试集按时间或数据量划分(迭代更新时实现，模拟现实)
        LogDividid('', 0.5, 'year') #划分方法 分割比例 month week day

        # 计算日志通用分析指标
        max_case_length = LA.GeneralIndicator(DR, DR.AllData)
        # for td in DR.Tests:
        #     LA.GeneralIndicator(DR, DR.Tests[td])

        # 以年为单位了解分布趋势
        # LA.PeriodicAnalysis(DR.AllData, 20, 'Year')
        # # 2.依据月、日、周的顺序计算分桶
        Buckets = []
        NoBuckets = [[[1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],[0,1,2,3,4,5,6]]]
        # BucketM = LA.PeriodicMerge(DR.Train, 20, 'Month')
        # BucketD = LA.PeriodicMerge(DR.Train, 20, 'Day')
        # BucketW = LA.PeriodicMerge(DR.Train, 20, 'Week')
        # BucketM = LA.PeriodicAnalysis(DR.Train, 20, 'Month')
        # BucketD = LA.PeriodicAnalysis(DR.Train, 20, 'Day')
        # BucketW = LA.PeriodicAnalysis(DR.Train, 20, 'Week')
        # for line in BucketM:
        #     temp = [line]
        #     for line2 in BucketD:
        #         temp.append(line2)
        #         for line3 in BucketW:
        #             temp.append(line3)
        #             Buckets.append(temp.copy())
        #             temp.pop(-1)
        #         temp.pop(-1)
        # BucketH = LA.PeriodicAnalysis(DR.AllData, 20, 'Hour') # 若执行时间小于1天，可尝试

        # 3.特征选取
        # 下一事件
        # ai, aiMAE = FS.AllFLightboost(DR.Train, DR.Train, DR.header, [attribute[i] - 3 for i in range(len(attribute))], 0)
        # # 计算FR
        # aii = len(ai) - 1
        # for i in range(len(ai) - 1):
        #     if aiMAE[i] - aiMAE[i + 1] < 0.01:
        #         aii = i
        # FE = [aiMAE[aii], [DR.header[i] for i in ai[0:aii]], ai[0:aii]]
        # # 下一事件时间
        # ai, aiMAE = FS.AllFLightboost(DR.Train, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))], 1)
        # # 计算FR
        # aii = len(ai) - 1
        # for i in range(len(ai) - 1):
        #     if aiMAE[i] - aiMAE[i + 1] < 0.01:
        #         aii = i
        # FD = [aiMAE[aii], [DR.header[i] for i in ai[0:aii]], ai[0:aii]]
        # 剩余时间
        # ai, aiMAE = FS.AllFLightboost(DR.Train, DR.Test, DR.header, [attribute[i]-3 for i in range(len(attribute))], 2)
        # # 计算FR
        # aii = len(ai)-1
        # for i in range(len(ai)-1):
        #     if aiMAE[i] - aiMAE[i+1] > 0.1:
        #         aii = i
        # FR = [aiMAE[aii], [DR.header[i] for i in ai[0:aii]], ai[0:aii]]
        # FR = [0, [DR.header[0]], [0]]
        FR = FS.LightGBM(DR.Train, DR.Test, DR.Test, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
        # 训练保存编码
        dataNameFR = '../Save/MultiTask/R_' + eventlog + 'N.mat'
        # dataNameFE = '../Save/MultiTask/E_' + eventlog + '.mat'
        # dataNameFD = '../Save/MultiTask/D_' + eventlog + '.mat'
        # dataFR = code(FR)#, dataFE, dataFD
        # scio.savemat(dataNameFR, dataFR)
        # scio.savemat(dataNameFE, dataFE)
        # scio.savemat(dataNameFD, dataFD)
        # 读取编码
        dataFR = scio.loadmat(dataNameFR)
        # dataFE = scio.loadmat(dataNameFE)
        # dataFD = scio.loadmat(dataNameFD)

        # 训练模型超参数
        preType = 2 #预测类型 1分类 2回归
        task = -1 #预测任务 -1剩余时间 -2下一事件时间 -3下一事件
        epoch = 200
        batchSize = 100
        LEN = 10
        method = ['NoFill']#, 'diffW', 'Transformer','Transformer''multiset',
        for m in method:
            print(m)
            if m == 'Transformer':
                MR = train('Transformer', dataFR['index'][-1], NoBuckets, mcl=max_case_length)
                # updateNew(MR, 'Transformer', [0, -9, -10], Buckets, 'month', mcl=max_case_length)
                # updateNew(MR, 'Transformer', [0, -9, -10], Buckets, 100, mcl=max_case_length)
                # updateNew(MR, 'Transformer', [0, -9, -10], Buckets, 'compre', 100, mcl=max_case_length)
                updateNew(copy.deepcopy(MR), 'Transformer', dataFR['index'][-1], NoBuckets, 'month', mcl=max_case_length)
                updateNew(copy.deepcopy(MR), 'Transformer', dataFR['index'][-1], NoBuckets, 100, mcl=max_case_length)
                updateNew(copy.deepcopy(MR), 'Transformer', dataFR['index'][-1], NoBuckets, 'compre', 100, mcl=max_case_length)
            elif m == 'multiset':
                MR = train('multiset', dataFR['index'][-1], [])
                updateNew([copy.deepcopy(MR[0]),MR[1],copy.deepcopy(MR[2])], 'multiset', dataFR['index'][-1], NoBuckets, 'month')
                updateNew([copy.deepcopy(MR[0]),MR[1],copy.deepcopy(MR[2])], 'multiset', dataFR['index'][-1], NoBuckets, 100)
                updateNew([copy.deepcopy(MR[0]),MR[1],copy.deepcopy(MR[2])], 'multiset', dataFR['index'][-1], NoBuckets, 'compre', 100)
            elif m == 'NoFill':
                for model in ['DNN2']:#, 'DNN1', 'DNN3'
                    print(model)
                    # 4.训练模型（不分层多特征输入）不增量 ①不分桶False ②分桶True（迁移）
                    MR = train('NoFill', dataFR['index'][-1], Buckets, model) # 方法名 特征列表dataFR['index']
                    # 5.更新模型 ①按周期增（month）②按数据量增（datas=100）③周期+数据量（compre 数据量够按周期增）
                    # 增量，不分桶
                    # updateNew([copy.deepcopy(MR[-1])], 'NoFill', dataFR['index'][-1], NoBuckets, 'month', model=model)
                    updateNew([copy.deepcopy(MR[-1])], 'NoFill', dataFR['index'][-1], NoBuckets, 100, model=model)
                    updateNew([copy.deepcopy(MR[-1])], 'NoFill', dataFR['index'][-1], NoBuckets, 'compre', 100, model=model)
                    # 增量，分桶
                    # updateNew(MR[0:-1].copy(), 'NoFill', dataFR['index'][-1], Buckets, 'month', model=model)
                    # updateNew(MR[0:-1].copy(), 'NoFill', dataFR['index'][-1], Buckets, 100, model=model)
                    # updateNew(MR[0:-1].copy(), 'NoFill', dataFR['index'][-1], Buckets, 'compre', 100, model=model)
