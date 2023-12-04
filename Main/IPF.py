import math
import random
import time
import warnings
warnings.filterwarnings('ignore')
import BPP_Frame.Log.DataRecord as DR
import BPP_Frame.Log.LogConvert as LC
import BPP_Frame.Log.LogAnalysis as LA
import BPP_Frame.Log.DivideData as DD
import BPP_Frame.Log.Prefix as P
import BPP_Frame.Feature.FeatureSel as FS
import BPP_Frame.Method.My.Model as M0
import BPP_Frame.Method.My.multiModel as M
import BPP_Frame.Code.word2vec as w2v
import BPP_Frame.Method.AETS.multiset as multiset
import BPP_Frame.Main.updata as updata
import torch
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import copy
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

def LogDividid(method, split, time = None):
    DR.Convert, DR.header, DR.ConvertReflact, maxA, maxR = LC.LogC(eventlog, attribute)  # 需转换属性值的下标
    # 特征类别编号
    # 0 活动；1 分类静态特征；2 分类动态特征；3 数值静态特征；4 数值动态特征
    DR.State = []
    DR.State.append(0)
    for i in range(4, len(DR.Convert[0]) - 9):
        if i in attribute:
            DR.State.append(1)
        else:
            DR.State.append(3)
    for i in range(6):
        DR.State.append(3)
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
    DR.Train_batch = P.NoFill(DR.Train.copy(), feature, task, batchSize)
    DR.Test_X, DR.Test_Y = P.changeLen(DR.Test.copy(), feature, task, 1)
    if method == 'NoFill':
        MR = M.trian(DR.Train_batch, DR.Test_X, DR.Test_Y, epoch, preType, -2, 32, 1, model, dataFR, isEarly=0)
    elif method == 'multiset':
        MR = multiset.train(DR.AllData, DR.Train, DR.Test, method, feature, isMul=1)
    elif method == 'Transformer':
        MR = M.trianT(DR.Train_batch, DR.Test_X, DR.Test_Y, epoch, preType, mcl, 'tran', dataFR)
    timeE = time.time()
    print('不分桶、不增量——MAE：', MR[1], timeE - timeS, 's')

    timeS = time.time()
    sumMAE = 0
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
                Train_batch = P.NoFill(tra, feature, task, batchSize)
                Test_X, Test_Y = P.changeLen(tes, feature, task, 1)
                if method == 'NoFill':
                    MRB = M.trian(Train_batch, Test_X, Test_Y, epoch, preType, -2, 32, 1, copy.deepcopy(MR[0]), dataFR, isEarly=0)
                    count += MRB[2]
                    sumMAE += (MRB[1][0] * MRB[2])
                elif method == 'multiset':
                    MRB = multiset.trainBucket(DR.AllData, tra, tes, method, feature, MR[2], MR[3], isMul=1)
                    count += MRB[1][1]
                    sumMAE += (MRB[1][1] * MRB[1][0])
                elif method == 'Transformer':
                    MRB = M.trianT(Train_batch, Test_X, Test_Y, epoch, preType, mcl, copy.deepcopy(MR[0]), dataFR)
                    count += MRB[2]
                    sumMAE += (MRB[1][0] * MRB[2])
                BucketModel.append(MRB[0])
            elif tes != []:
                Test_X, Test_Y = P.changeLen(tes, feature, task, 1)
                if method == 'NoFill':
                    MAE, countg = M.test(Test_X, Test_Y, MR[0], preType, dataFR, len(feature))
                    count += countg
                    sumMAE +=  MAE * countg
                elif method == 'multiset':
                    MAE, countg = multiset.testBucket(DR.AllData, tes, MR[2], MR[3], 'multiset', feature, isMul=1)
                    count += countg
                    sumMAE += countg * MAE
                elif method == 'Transformer':
                    MAE, countg = M.testT(Test_X, Test_Y, MR[0], preType, dataFR)
                    count += countg
                    sumMAE += MAE * countg
                BucketModel.append(MR[0])
        DR.BucketTrain = BucketTrain
        DR.BucketTest = BucketTest
        timeE = time.time()
        print('分桶、不增量——MAE：', sumMAE/count, timeE - timeS, 's')
        if method == 'multiset':
            BucketModel.extend([MR[2], MR[3], MR[0]])  # encoder, total, dict
        else:
            BucketModel.extend([MR[0]])
        return BucketModel
    else:
        DR.BucketTrain = [DR.Train]
        if method == 'multiset':
            return MR
        DR.BucketTest = [DR.Test]
        return [MR[0]]

if __name__ == "__main__":
    # 属性转换
    EL = ['hd','BPIC2015_1', 'BPIC2015_2', 'BPIC2015_3', 'BPIC2015_4', 'BPIC2015_5', 'sepsis']# ,'hospital','RRT'
    Att = [[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],#
           [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
           [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
           [3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]

    for eventlog,attribute in zip(EL, Att):
        if eventlog=='RRT' or eventlog=='hospital':
            updataNum = 1000
        else:
            updataNum = 100
        if eventlog=='hd':
            hisdata = 1
        else:
            hisdata = 2
        print(eventlog)
        dataPlot = {}
        # 1.日志划分为初始训练集和增量训练集，测试集按时间或数据量划分(迭代更新时实现，模拟现实)
        LogDividid('', 0.5, 'year') #划分方法 分割比例 month week day
        # 计算日志通用分析指标
        max_case_length = LA.GeneralIndicator(DR, DR.AllData)
        # for td in DR.Tests:
        #     LA.GeneralIndicator(DR, DR.Tests[td])

        # 以年为单位了解分布趋势
        # LA.PeriodicAnalysis(DR.AllData, 20, 'Year')

        Buckets = []
        NoBuckets = [[[1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],[0,1,2,3,4,5,6]]]

        Train, Val = train_test_split(DR.Train, test_size=0.2, random_state=20)
        FR = FS.LightGBM(Train, Val, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
        state = [j for i, j in zip(DR.State, range(len(DR.State))) if i == 2 or i == 4]
        PR = FS.PrefixLightGBM(Train, Val, DR.header, state, [attribute[i] - 3 for i in range(len(attribute))], FR)
        # 活动编码 训练
        Train_XA, Train_YA = P.cutPrefixBy(DR.Train, [0], label=-3, batchSize=20, LEN=3)  # [FR[2][0]]
        EmbA, ACCE = w2v.word2vec(Train_XA, Train_YA, DR.ConvertReflact)
        dataFR = {'0': EmbA.detach().numpy(), 'name': FR[1], 'index': FR[2], 'state': [DR.State[i] for i in FR[2]], 'result': FR[0], 'prefix': PR}
        # 其他分类特征编码 随机初始化Embding
        for i in range(1, len(DR.Train[0][0]) - 3):
            if i + 3 in attribute:
                if len(DR.ConvertReflact[attribute.index(i + 3)]) > 5:
                    eim = 5
                    olen = len(DR.ConvertReflact[attribute.index(i + 3)])
                    while olen > 20:
                        olen /= 4
                        eim += 5
                    EmbS = nn.Embedding(len(DR.ConvertReflact[attribute.index(i + 3)]), eim)
                    if i in dataFR['index']:
                        dataFR[str(i)] = EmbS.weight.detach().numpy()
        # 训练保存编码
        dataNameFR = '../Save/multiFea/preFR' + eventlog + 'Inc.mat'
        dataFR = code()
        scio.savemat(dataNameFR, dataFR)
        # 读取编码
        dataFR = scio.loadmat(dataNameFR)

        # variant, activites = LA.LogRecord(DR)
        variant = None
        activites = None
        # 训练模型超参数
        preType = 2 #预测类型 1分类 2回归
        task = -1 #预测任务 -1剩余时间 -2下一事件时间 -3下一事件
        epoch = 200 # 10 #
        batchSize = 100
        LEN = 10
        method = ['NoFill', 'multiset', 'Transformer']
        for m in method:
            print(m)
            if m == 'Transformer':#MLTM, MLTMI MLT100, MLT100I MLTC, MLTCI
                MR = train('Transformer', dataFR['index'][-1], Buckets, mcl=max_case_length)
                # Step 1:
                # dataPlot['Transmonth'], dataPlot['TransmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR, NoBuckets, 'month', mcl=max_case_length,hisdata=1)
                # dataPlot['Transmonth'], dataPlot['TransmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR, NoBuckets, 'month', mcl=max_case_length,hisdata=2)
                # dataPlot['Transmonth'], dataPlot['TransmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR, NoBuckets, 'month', mcl=max_case_length,hisdata=3)
                # Step 2:
                # dataPlot['Transres'], dataPlot['TransresI'] = updata.updateAuto0(None,[copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], m, dataFR, attribute, NoBuckets, updataNum, max_case_length,hisdata=hisdata)
                # dataPlot['Transmonth'], dataPlot['TransmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR, NoBuckets, 'month', mcl=max_case_length,hisdata=hisdata)
                # dataPlot['Trans100'], dataPlot['Trans100I'] = updata.updateNew([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'Transformer', dataFR, NoBuckets, updataNum, mcl=max_case_length, hisdata=hisdata)
                # Step 3:
                if eventlog == 'hd':
                    dataPlot['Trans100'], dataPlot['Trans100I'] = updata.updateNew1([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR.copy(), attribute, NoBuckets, updataNum, mcl=max_case_length,hisdata=hisdata,DEL=0)
                    dataPlot['Trans100'], dataPlot['Trans100I'] = updata.updateNew1([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR.copy(), attribute, NoBuckets, updataNum, mcl=max_case_length,hisdata=hisdata,DEL=1)
                elif eventlog == 'sepsis':
                    dataPlot['Transcom'], dataPlot['TranscomI'] = updata.updateNew1([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR.copy(), attribute, NoBuckets, 'compre', updataNum, mcl=max_case_length,hisdata=hisdata,DEL=0)
                    dataPlot['Transcom'], dataPlot['TranscomI'] = updata.updateNew1([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR.copy(), attribute, NoBuckets, 'compre', updataNum, mcl=max_case_length,hisdata=hisdata,DEL=1)
                else:
                    dataPlot['TransmonthRT'] = updata.updateNew2([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'Transformer', dataFR.copy(), attribute, NoBuckets, 'month', mcl=max_case_length, hisdata=hisdata,DEL=0)
                    dataPlot['Transmonth'], dataPlot['TransmonthI'] = updata.updateNew1([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'Transformer', dataFR.copy(), attribute, NoBuckets, 'month', mcl=max_case_length, hisdata=hisdata,DEL=1)
            elif m == 'multiset':
                MR = train('multiset', dataFR['index'][-1], Buckets)
                dataPlot['Autores'], dataPlot['AutoresI'] = updata.updateAuto0(MR[-1], [copy.deepcopy(MR[0]), copy.deepcopy(MR[0]), copy.deepcopy(MR[2])], m, dataFR, attribute, NoBuckets, 100, hisdata=hisdata)
                dataPlot['Automonth'], dataPlot['AutomonthI'] = updata.updateNew([copy.deepcopy(MR[0]), copy.deepcopy(MR[0]), copy.deepcopy(MR[2])], m, dataFR, NoBuckets, 'month', hisdata=hisdata)
                dataPlot['Auto100'], dataPlot['Auto100I'] = updata.updateNew([copy.deepcopy(MR[0]), copy.deepcopy(MR[0]), copy.deepcopy(MR[2])], m, dataFR, NoBuckets, updataNum, hisdata=hisdata)
            elif m == 'NoFill':
                model = 'rnn'
                # 4.训练模型（不分层多特征输入）不增量 ①不分桶False ②分桶True（迁移）
                MR = train('NoFill', dataFR['index'][-1], Buckets, model) # 方法名 特征列表dataFR['index']
                # 5.更新模型 ①按周期增（month）②按数据量增（datas=100）③周期+数据量（compre 数据量够按周期增）
                # Step 1:
                # dataPlot['LSTMmonth'], dataPlot['LSTMmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'NoFill', dataFR, NoBuckets, 'month', model=model, hisdata=1)
                # dataPlot['LSTMmonth'], dataPlot['LSTMmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'NoFill', dataFR, NoBuckets, 'month', model=model, hisdata=2)
                # dataPlot['LSTMmonth'], dataPlot['LSTMmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'NoFill', dataFR, NoBuckets, 'month', model=model, hisdata=3)
                # Step 2:
                dataPlot['LSTMres'], dataPlot['LSTMresI'] = updata.updateAuto0(None, [copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], m, dataFR, attribute, NoBuckets, 100,hisdata=hisdata)
                dataPlot['LSTMmonth'], dataPlot['LSTMmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'NoFill', dataFR, NoBuckets, 'month', model=model,hisdata=hisdata)
                dataPlot['LSTM100'], dataPlot['LSTM100I'] = updata.updateNew( [copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'NoFill', dataFR, NoBuckets, updataNum, model=model,hisdata=hisdata)

        dataName = '../Save/increPlot/' + eventlog + 'Step3N.mat'#TL Step3 Auto
        scio.savemat(dataName, dataPlot)