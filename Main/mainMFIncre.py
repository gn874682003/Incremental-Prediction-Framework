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

def plotIncre(LSTM,LSTMI,Trans,TransI,AutoE,AutoEI):
    # names = ['5', '10', '15', '20', '25']
    # x = range(len(names))
    # y = [0.855, 0.84, 0.835, 0.815, 0.81]
    # y1 = [0.86, 0.85, 0.853, 0.849, 0.83]
    # plt.plot(x, y, 'bo-', label='LSTM')
    # plt.plot(x, y1, 'bo-.', label='LSTMIncre')
    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # plt.plot(range(len(LSTM)), LSTM, marker='o', mec='r', mfc='w', label='LSTM')
    # plt.plot(range(len(Trans)), Trans, marker='*', ms=10, label='Transformer')
    # pl.xlim(-1, 11) # 限定横轴的范围
    # pl.ylim(-1, 110) # 限定纵轴的范围
    if LSTM != None:
        plt.plot(range(len(LSTM)), LSTM, 'r-', label='LSTM')
    if LSTMI != None:
        plt.plot(range(len(LSTMI)), LSTMI, 'r-.', label='LSTMIncre')
    if Trans != None:
        plt.plot(range(len(Trans)), Trans, 'b-', label='Transformer')
    if TransI != None:
        plt.plot(range(len(TransI)), TransI, 'b-.', label='TransformerIncre')
    plt.legend()  # 让图例生效
    # plt.xticks(x, names, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("Updata Times")  # X轴标签
    plt.ylabel("MAE")  # Y轴标签
    plt.title('Helpdesk')  # 标题

    plt.show()

if __name__ == "__main__":
    # plotIncre(None, None, None, None, None, None)
    # 属性转换
    EL = ['hd','BPIC2015_1', 'BPIC2015_2', 'BPIC2015_3', 'BPIC2015_4', 'BPIC2015_5', 'sepsis']# ,'hospital','RRT'
    Att = [[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],#
           [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
           [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
           [3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]

           # [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [3, 5, 6, 7, 9, 10, 11, 12, 13]]
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

        # Train, Val = train_test_split(DR.Train, test_size=0.2, random_state=20)
        # FR = FS.LightGBM(Train, Val, DR.header, [attribute[i] - 3 for i in range(len(attribute))])
        # state = [j for i, j in zip(DR.State, range(len(DR.State))) if i == 2 or i == 4]
        # PR = FS.PrefixLightGBM(Train, Val, DR.header, state, [attribute[i] - 3 for i in range(len(attribute))], FR)
        # # 活动编码 训练
        # Train_XA, Train_YA = P.cutPrefixBy(DR.Train, [0], label=-3, batchSize=20, LEN=3)  # [FR[2][0]]
        # EmbA, ACCE = w2v.word2vec(Train_XA, Train_YA, DR.ConvertReflact)
        # dataFR = {'0': EmbA.detach().numpy(), 'name': FR[1], 'index': FR[2], 'state': [DR.State[i] for i in FR[2]], 'result': FR[0], 'prefix': PR}
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
        #             if i in dataFR['index']:
        #                 dataFR[str(i)] = EmbS.weight.detach().numpy()
        # 训练保存编码
        dataNameFR = '../Save/multiFea/preFR' + eventlog + 'Inc.mat'
        # dataNameFE = '../Save/MultiTask/E_' + eventlog + '.mat'
        # dataNameFD = '../Save/MultiTask/D_' + eventlog + '.mat'
        # dataFR = code()#, dataFE, dataFD
        # scio.savemat(dataNameFR, dataFR)
        # scio.savemat(dataNameFE, dataFE)
        # scio.savemat(dataNameFD, dataFD)
        # 读取编码
        dataFR = scio.loadmat(dataNameFR)
        # dataFE = scio.loadmat(dataNameFE)
        # dataFD = scio.loadmat(dataNameFD)
        # variant, activites = LA.LogRecord(DR)
        variant = None
        activites = None
        # 训练模型超参数
        preType = 2 #预测类型 1分类 2回归
        task = -1 #预测任务 -1剩余时间 -2下一事件时间 -3下一事件
        epoch = 200 # 10 #
        batchSize = 100
        LEN = 10
        method = ['Transformer']# 'diffW','NoFill', , 'multiset'
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
                # dataPlot['TransMAE'], dataPlot['TransMAEI'] = updata.updateAuto([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], m, dataFR, attribute, NoBuckets, updataNum,  max_case_length,hisdata=hisdata)
                # dataPlot['Transmonth'], dataPlot['TransmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR, NoBuckets, 'month', mcl=max_case_length,hisdata=hisdata)
                # dataPlot['Trans100'], dataPlot['Trans100I'] = updata.updateNew([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'Transformer', dataFR, NoBuckets, updataNum, mcl=max_case_length, hisdata=hisdata)
                # dataPlot['Transcom'], dataPlot['TranscomI'] = updata.updateNew([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'Transformer', dataFR, NoBuckets, 'compre', updataNum, mcl=max_case_length, hisdata=hisdata)
                # Step 3:
                # if eventlog == 'hd':
                #     dataPlot['Trans100'], dataPlot['Trans100I'] = updata.updateNew1([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR.copy(), attribute, NoBuckets, updataNum, mcl=max_case_length,hisdata=hisdata,DEL=0)
                #     dataPlot['Trans100'], dataPlot['Trans100I'] = updata.updateNew1([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR.copy(), attribute, NoBuckets, updataNum, mcl=max_case_length,hisdata=hisdata,DEL=1)
                # elif eventlog == 'sepsis':
                #     dataPlot['Transcom'], dataPlot['TranscomI'] = updata.updateNew1([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR.copy(), attribute, NoBuckets, 'compre', updataNum, mcl=max_case_length,hisdata=hisdata,DEL=0)
                #     dataPlot['Transcom'], dataPlot['TranscomI'] = updata.updateNew1([copy.deepcopy(MR[-1]),copy.deepcopy(MR[-1])], 'Transformer', dataFR.copy(), attribute, NoBuckets, 'compre', updataNum, mcl=max_case_length,hisdata=hisdata,DEL=1)
                # else:
                dataPlot['TransmonthRT'] = updata.updateNew2([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'Transformer', dataFR.copy(), attribute, NoBuckets, 'month', mcl=max_case_length, hisdata=hisdata,DEL=0)
                # dataPlot['Transmonth'], dataPlot['TransmonthI'] = updata.updateNew1([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'Transformer', dataFR.copy(), attribute, NoBuckets, 'month', mcl=max_case_length, hisdata=hisdata,DEL=1)
            elif m == 'multiset':
                MR = train('multiset', dataFR['index'][-1], Buckets)
                # multiset.represent(DR.AllData, DR.Test, 'multiset', dataFR['index'][-1], copy.deepcopy(MR[2]), MR[-1])
                dataPlot['Autores'], dataPlot['AutoresI'] = updata.updateAuto0(MR[-1], [copy.deepcopy(MR[0]), copy.deepcopy(MR[0]), copy.deepcopy(MR[2])], m, dataFR, attribute, NoBuckets, 100, hisdata=hisdata)
                # dataPlot['AutoMAE'], dataPlot['AutoMAEI'] = updata.updateAuto([copy.deepcopy(MR[0]), copy.deepcopy(MR[0]), copy.deepcopy(MR[2])], m, dataFR, attribute, NoBuckets, 100, hisdata=hisdata)
                # dataPlot['Automonth'], dataPlot['AutomonthI'] = updata.updateNew([copy.deepcopy(MR[0]), copy.deepcopy(MR[0]), copy.deepcopy(MR[2])], m, dataFR, NoBuckets, 'month', hisdata=hisdata)
                # dataPlot['Auto100'], dataPlot['Auto100I'] = updata.updateNew([copy.deepcopy(MR[0]), copy.deepcopy(MR[0]), copy.deepcopy(MR[2])], m, dataFR, NoBuckets, updataNum, hisdata=hisdata)
                # dataPlot['Autocom'], dataPlot['AutocomI'] = updata.updateNew([copy.deepcopy(MR[0]), copy.deepcopy(MR[0]), copy.deepcopy(MR[2])], m, dataFR, NoBuckets, 'compre', updataNum, hisdata=hisdata)
                # dataPlot['Automonth'], dataPlot['AutomonthI'] = updateNew([copy.deepcopy(MR[-3]),copy.deepcopy(MR[-2]), copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'multiset', dataFR['index'][-1], NoBuckets, 'month')
                # dataPlot['Auto100'], dataPlot['Auto100I'] = updateNew([copy.deepcopy(MR[-3]),copy.deepcopy(MR[-2]), copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'multiset', dataFR['index'][-1], NoBuckets, updataNum)
                # dataPlot['Autocom'], dataPlot['AutocomI'] = updateNew([copy.deepcopy(MR[-3]),copy.deepcopy(MR[-2]), copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'multiset', dataFR['index'][-1], NoBuckets, 'compre', updataNum)
                # if len(Buckets) > 1:
                #     updateNew(copy.deepcopy(MR[0:-1]), 'multiset', dataFR['index'][-1], Buckets, 'month')
                #     updateNew(copy.deepcopy(MR[0:-1]), 'multiset', dataFR['index'][-1], Buckets, updataNum)
                #     updateNew(copy.deepcopy(MR[0:-1]), 'multiset', dataFR['index'][-1], Buckets, 'compre', updataNum)
            elif m == 'NoFill':
                model = 'rnn'
                # 4.训练模型（不分层多特征输入）不增量 ①不分桶False ②分桶True（迁移）
                MR = train('NoFill', dataFR['index'][-1], Buckets, model) # 方法名 特征列表dataFR['index']
                # 5.更新模型 ①按周期增（month）②按数据量增（datas=100）③周期+数据量（compre 数据量够按周期增）
                # 增量，不分桶 MLRM, MLRMI MLR100, MLR100I MLRC, MLRCI
                # Step 1:
                # dataPlot['LSTMmonth'], dataPlot['LSTMmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'NoFill', dataFR, NoBuckets, 'month', model=model, hisdata=1)
                # dataPlot['LSTMmonth'], dataPlot['LSTMmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'NoFill', dataFR, NoBuckets, 'month', model=model, hisdata=2)
                # dataPlot['LSTMmonth'], dataPlot['LSTMmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'NoFill', dataFR, NoBuckets, 'month', model=model, hisdata=3)
                # Step 2:
                dataPlot['LSTMres'], dataPlot['LSTMresI'] = updata.updateAuto0(None, [copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], m, dataFR, attribute, NoBuckets, 100,hisdata=hisdata)
                # dataPlot['LSTMMAE'], dataPlot['LSTMMAEI'] = updata.updateAuto([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], m, dataFR, attribute, NoBuckets, 100,hisdata=hisdata)
                # dataPlot['LSTMmonth'], dataPlot['LSTMmonthI'] = updata.updateNew([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'NoFill', dataFR, NoBuckets, 'month', model=model,hisdata=hisdata)
                # dataPlot['LSTM100'], dataPlot['LSTM100I'] = updata.updateNew( [copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'NoFill', dataFR, NoBuckets, updataNum, model=model,hisdata=hisdata)
                # dataPlot['LSTMcom'], dataPlot['LSTMcomI'] = updata.updateNew([copy.deepcopy(MR[-1]), copy.deepcopy(MR[-1])], 'NoFill', dataFR, NoBuckets, 'compre', updataNum, model,hisdata=hisdata)
                # 增量，分桶
                # if len(Buckets) > 1:
                #     updateNew(copy.deepcopy(MR), 'NoFill', dataFR['index'][-1], Buckets, 'month', model=model)
                #     updateNew(copy.deepcopy(MR), 'NoFill', dataFR['index'][-1], Buckets, updataNum, model=model)
                #     updateNew(copy.deepcopy(MR), 'NoFill', dataFR['index'][-1], Buckets, 'compre', updataNum, model=model)
        # plotIncre(MLRM, MLRMI, MLTM, MLTMI, None, None)
        # plotIncre([MLR100, MLR100I, MLT100, MLT100I])
        # plotIncre([MLRC, MLRCI, MLTC, MLTCI])
        dataName = '../Save/increPlot/' + eventlog + 'Step3N.mat'#TL Step3 Auto
        scio.savemat(dataName, dataPlot)