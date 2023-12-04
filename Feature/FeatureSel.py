import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from xgboost import plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import sklearn.feature_selection as feaSel
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
from minepy import MINE
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import lightgbm as lgb
# import shap
import time
import Frame.tree as tp

# 返回模型在Train训练后，Test的测试结果
def TestK(Train, Test, header, catId, aii):
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    global ai, ak
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]
    modelR = lgb.LGBMRegressor()
    modelR.fit(X_train[:, aii], y_train[:, 2], feature_name=[str(i) for i in aii],
               categorical_feature=[str(i) for i in aii if i in catId])
    y_pred = modelR.predict(X_test[:, aii])
    MAE = mean_absolute_error(y_test[:, 2], y_pred)
    FR = [MAE, [header[i] for i in aii], aii]
    print(FR)
    return FR

def TestLGBM(Train, Test, dataFE, dataFD, dataFR, PF=5):
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            list_to_float1.append(line1)
    for line in Test:
        for line1 in line:
            list_to_float2.append(line1)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    ai = dataFE['index'][-1].tolist()
    # print([str(i) for i in ai])
    # print([str(ai[i]) for i in range(len(ai)) if dataFE['state'][-1][i] < 3])
    modelE = lgb.LGBMClassifier(learning_rate=0.01, max_depth=5, num_leaves=150, n_estimators=100)
    modelE.fit(dataTra[:, ai], dataTra[:, -3], feature_name=[str(i) for i in ai],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if dataFE['state'][-1][i] < 3])
    y_pred = modelE.predict(dataTes[:, ai])
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(dataTes[:, -3], predictions)
    print('下一事件：', accuracy)
    # 持续时间
    ai = dataFD['index'][-1].tolist()
    modelD = lgb.LGBMRegressor()
    modelD.fit(dataTra[:, ai], dataTra[:, -2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if dataFD['state'][-1][i] < 3])
    y_pre = modelD.predict(dataTes[:, ai])
    MAE = mean_absolute_error(dataTes[:, -2], y_pre)
    print('持续时间：', MAE)
    # 剩余时间
    ai = dataFR['index'][-1].tolist()
    modelR = lgb.LGBMRegressor()
    modelR.fit(dataTra[:, ai], dataTra[:, -1], feature_name=[str(i) for i in ai],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if dataFR['state'][-1][i] < 3])
    y_pre = modelR.predict(dataTes[:, ai])
    MAE = mean_absolute_error(dataTes[:, -1], y_pre)
    print('剩余时间：', MAE)

    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        lineP = []
        for line1 in line:
            lineP.append(line1)
            if len(lineP) < PF:
                continue
            if len(lineP) > PF:
                lineP.pop(0)
            IP = []
            for lp in lineP:
                IP.extend(lp[0:-3])
            IP.extend(lp[-3:-1])
            IP.append(lp[-1])
            list_to_float1.append(IP)
    for line in Test:
        lineP = []
        for line1 in line:
            lineP.append(line1)
            if len(lineP) < PF:
                continue
            if len(lineP) > PF:
                lineP.pop(0)
            IP = []
            for lp in lineP:
                IP.extend(lp[0:-3])
            IP.extend(lp[-3:-1])
            IP.append(lp[-1])
            list_to_float2.append(IP)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    ai = [i + (len(Train[0][0]) - 3) * j for j in range(PF) for i in dataFE['prefix'][-1]]
    ai.extend([i + (len(Train[0][0]) - 3) * (PF-1) for i in dataFE['index'][-1] if i not in dataFE['prefix'][-1]])
    cf = [np.where(dataFE['index'][-1] == i % (len(Train[0][0]) - 3))[0][0] for i in ai]
    modelE = lgb.LGBMClassifier(learning_rate=0.01, max_depth=5, num_leaves=150, n_estimators=100)
    modelE.fit(dataTra[:, ai], dataTra[:, -3], feature_name=[str(i) for i in ai],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if cf[i] < 3])
    y_pred = modelE.predict(dataTes[:, ai])
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(dataTes[:, -3], predictions)
    print('下一事件：', accuracy)
    # 持续时间
    ai = [i + (len(Train[0][0]) - 3) * j for j in range(PF) for i in dataFD['prefix'][-1]]
    ai.extend([i + (len(Train[0][0]) - 3) * (PF - 1) for i in dataFD['index'][-1] if i not in dataFD['prefix'][-1]])
    cf = [np.where(dataFD['index'][-1] == i % (len(Train[0][0]) - 3))[0][0] for i in ai]
    modelD = lgb.LGBMRegressor()
    modelD.fit(dataTra[:, ai], dataTra[:, -2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if cf[i] < 3])
    y_pre = modelD.predict(dataTes[:, ai])
    MAE = mean_absolute_error(dataTes[:, -2], y_pre)
    print('持续时间：', MAE)
    # 剩余时间
    ai = [i + (len(Train[0][0]) - 3) * j for j in range(PF) for i in dataFR['prefix'][-1]]
    ai.extend([i + (len(Train[0][0]) - 3) * (PF - 1) for i in dataFR['index'][-1] if i not in dataFR['prefix'][-1]])
    cf = [np.where(dataFR['index'][-1] == i % (len(Train[0][0]) - 3))[0][0] for i in ai]
    modelR = lgb.LGBMRegressor()
    modelR.fit(dataTra[:, ai], dataTra[:, -1], feature_name=[str(i) for i in ai],
               categorical_feature=[str(ai[i]) for i in range(len(ai)) if cf[i] < 3])
    y_pre = modelR.predict(dataTes[:, ai])
    MAE = mean_absolute_error(dataTes[:, -1], y_pre)
    print('剩余时间：', MAE)

def plotFeature(X,name):
    aki = sorted(range(len(X)), key=lambda k: X[k])
    X.sort()
    name = [name[aki[i]] for i in range(len(aki))]
    # 图像绘制
    fig, ax = plt.subplots()
    b = ax.barh(range(len(name)), X, color='k')
    # 添加数据标签
    for rect in b:
        w = rect.get_width()
        ax.text(w, rect.get_y() + rect.get_height() / 2, '%f' % w, ha='left', va='center')
    # 设置Y轴刻度线标签
    ax.set_yticks(range(len(name)))
    ax.set_yticklabels(name)
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xlabel('importance value')
    plt.ylabel('attribute')
    plt.show()

#前缀选取
def PrefixLightGBM(Train, Val, Val1, header, state, catId, FR, PF=5):#FE, FD,
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        lineP = []
        for line1 in line:
            lineP.append(line1)
            if len(lineP) < PF:
                continue
            if len(lineP) > PF:
                lineP.pop(0)
            IP = []
            for lp in lineP:
                IP.extend(lp[0:-3])
            IP.extend(lp[-3:-1])
            IP.append(lp[-1])
            list_to_float1.append(IP)
    for line in Val:
        lineP = []
        for line1 in line:
            lineP.append(line1)
            if len(lineP) < PF:
                continue
            if len(lineP) > PF:
                lineP.pop(0)
            IP = []
            for lp in lineP:
                IP.extend(lp[0:-3])
            IP.extend(lp[-3:-1])
            IP.append(lp[-1])
            list_to_float2.append(IP)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    # 下一事件
    # ak = [header[i]+str(j) for j in range(PF) for i in FE[2]]
    # ai = [i+(len(header)-3)*j for j in range(PF) for i in FE[2]]
    # X_train = dataTra[:, ai]
    y_train = dataTra[:, (len(header)-3)*PF:(len(header)-3)*PF + 3]
    # X_test = dataTes[:, ai]
    y_test = dataTes[:, (len(header)-3)*PF:(len(header)-3)*PF + 3]

    list_to_float2 = []
    for line in Val1:
        lineP = []
        for line1 in line:
            lineP.append(line1)
            if len(lineP) < PF:
                continue
            if len(lineP) > PF:
                lineP.pop(0)
            IP = []
            for lp in lineP:
                IP.extend(lp[0:-3])
            IP.extend(lp[-3:-1])
            IP.append(lp[-1])
            list_to_float2.append(IP)
    dataTes1 = np.array(list_to_float2)
    y_test1 = dataTes1[:, (len(header) - 3) * PF:(len(header) - 3) * PF + 3]
    # 调参
    # lg = lgb.LGBMClassifier(silent=False)
    # param_dist = {"max_depth": [5, 7], "learning_rate": [0.01], "num_leaves": [150, 200, 250], "n_estimators": [100]}
    # grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv=3, scoring="roc_auc", verbose=5)
    # grid_search.fit(X_train, y_train[:, 0])
    # gb = grid_search.best_estimator_
    # y_pre = grid_search.predict(X_test)
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)

    # modelE = xgb.XGBClassifier()
    # modelE = lgb.LGBMClassifier(learning_rate=0.01,max_depth=5,num_leaves=150,n_estimators=100)
    # sk = []
    # si = []
    # for i in FE[2][1:]:  #
    #     if i not in state:
    #         sk.extend([header[i] + str(j) for j in range(PF - 1)])
    #         si.extend([i + (len(header) - 3) * j for j in range(PF - 1)])
    # for j, k in zip(si, sk):
    #     ak.remove(k)
    #     ai.remove(j)
    # modelE.fit(dataTra[:, ai], y_train[:, 0], feature_name=[str(i) for i in ai],
    #            categorical_feature=[str(i) for i in ai if i%(len(header)-3) in catId])
    # y_pre = modelE.predict(dataTes[:, ai])
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)
    # print(accuracy)
    # # X = modelE.feature_importances_
    # # plotFeature(X, ak)
    # # 迭代策略
    # lkk = []
    # lii = []
    # for i in FE[2]:#[1:]
    #     aii = ai.copy()
    #     akk = ak.copy()
    #     lk = [header[i]+str(j) for j in range(PF-1)]
    #     li = [i+(len(header)-3)*j for j in range(PF-1)]
    #     if i in state:
    #         for j,k in zip(li, lk):
    #             akk.remove(k)
    #             aii.remove(j)
    #     else:
    #         continue
    #     modelE.fit(dataTra[:, aii], y_train[:, 0], feature_name=[str(i) for i in aii],
    #            categorical_feature=[str(i) for i in aii if i%(len(header)-3) in catId])
    #     y_pre = modelE.predict(dataTes[:, aii])
    #     predictions = [round(value) for value in y_pre]
    #     accuracyi = accuracy_score(y_test[:, 0], predictions)
    #     print(i, ':', accuracyi)
    #     if accuracyi > accuracy:
    #         lkk.extend(lk)
    #         lii.extend(li)
    # for j, k in zip(lii, lkk):
    #     ak.remove(k)
    #     ai.remove(j)
    # modelE.fit(dataTra[:, ai], y_train[:, 0], feature_name=[str(i) for i in ai],
    #            categorical_feature=[str(i) for i in ai if i%(len(header)-3) in catId])
    # y_pre = modelE.predict(dataTes[:, ai])
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)
    # print(accuracy, ak)
    # # X = modelE.feature_importances_
    # # plotFeature(X, ak)
    # PE = [i for i in ai if i < len(header)-3]
    # # 持续时间
    # ak = [header[i] + str(j) for j in range(PF) for i in FD[2]]
    # ai = [i + (len(header) - 3) * j for j in range(PF) for i in FD[2]]
    # X_train = dataTra[:, ai]
    # # modelD = xgb.XGBRegressor()
    # modelD = lgb.LGBMRegressor(max_depth=5)
    # sk = []
    # si = []
    # for i in FD[2][1:]:  #
    #     if i not in state:
    #         sk.extend([header[i] + str(j) for j in range(PF - 1)])
    #         si.extend([i + (len(header) - 3) * j for j in range(PF - 1)])
    # for j, k in zip(si, sk):
    #     ak.remove(k)
    #     ai.remove(j)
    # modelD.fit(dataTra[:, ai], y_train[:, 1], feature_name=[str(i) for i in ai],
    #            categorical_feature=[str(i) for i in ai if i%(len(header)-3) in catId])
    # y_pre = modelD.predict(dataTes[:, ai])
    # MAE = mean_absolute_error(y_test[:, 1], y_pre)
    # print(MAE)
    # # X2 = modelD.feature_importances_
    # # plotFeature(X2, ak)
    # # 迭代策略
    # lkk = []
    # lii = []
    # for i in FD[2]:#[1:]
    #     aii = ai.copy()
    #     akk = ak.copy()
    #     lk = [header[i] + str(j) for j in range(PF - 1)]
    #     li = [i + (len(header) - 3) * j for j in range(PF - 1)]
    #     if i in state:
    #         for j, k in zip(li, lk):
    #             akk.remove(k)
    #             aii.remove(j)
    #     else:
    #         continue
    #     modelD.fit(dataTra[:, aii], y_train[:, 1], feature_name=[str(i) for i in aii],
    #            categorical_feature=[str(i) for i in aii if i%(len(header)-3) in catId])
    #     y_pre = modelD.predict(dataTes[:, aii])
    #     MAEi = mean_absolute_error(y_test[:, 1], y_pre)
    #     print(i, ':', MAEi)
    #     if MAEi < MAE:
    #         lkk.extend(lk)
    #         lii.extend(li)
    # for j, k in zip(lii, lkk):
    #     ak.remove(k)
    #     ai.remove(j)
    # modelD.fit(dataTra[:, ai], y_train[:, 1], feature_name=[str(i) for i in ai],
    #            categorical_feature=[str(i) for i in ai if i%(len(header)-3) in catId])
    # y_pre = modelD.predict(dataTes[:, ai])
    # MAE = mean_absolute_error(y_test[:, 1], y_pre)
    # print(MAE,ak)
    # # X = modelD.feature_importances_
    # # plotFeature(X, ak)
    # PD = [i for i in ai if i < len(header) - 3]
    # 剩余时间
    ak = [header[i] + str(j) for j in range(PF) for i in FR[2]]
    ai = [i + (len(header) - 3) * j for j in range(PF) for i in FR[2]]
    X_train = dataTra[:, ai]
    # modelR = xgb.XGBRegressor()
    modelR = lgb.LGBMRegressor(max_depth=5)
    sk = []
    si = []
    for i in FR[2][1:]:  #
        if i not in state:
            sk.extend([header[i] + str(j) for j in range(PF - 1)])
            si.extend([i + (len(header) - 3) * j for j in range(PF - 1)])
    for j, k in zip(si, sk):
        ak.remove(k)
        ai.remove(j)
    modelR.fit(dataTra[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i%(len(header)-3) in catId])
    y_pre = modelR.predict(dataTes1[:, ai])
    MAE = mean_absolute_error(y_test1[:, 2], y_pre)
    print(MAE)
    X3 = modelR.feature_importances_
    plotFeature(X3, ak)
    # 迭代策略
    lkk = []
    lii = []
    for i in FR[2]:#[1:]
        aii = ai.copy()
        akk = ak.copy()
        lk = [header[i] + str(j) for j in range(PF - 1)]
        li = [i + (len(header) - 3) * j for j in range(PF - 1)]
        if i in state:
            for j, k in zip(li, lk):
                akk.remove(k)
                aii.remove(j)
        else:
            continue
        modelR.fit(dataTra[:, aii], y_train[:, 2], feature_name=[str(i) for i in aii],
               categorical_feature=[str(i) for i in aii if i%(len(header)-3) in catId])
        y_pre = modelR.predict(dataTes[:, aii])
        MAEi = mean_absolute_error(y_test[:, 2], y_pre)
        print(i, ':', MAEi)
        if MAEi < MAE:
            lkk.extend(lk)
            lii.extend(li)
    for j, k in zip(lii, lkk):
        ak.remove(k)
        ai.remove(j)
    modelR.fit(dataTra[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i%(len(header)-3) in catId])
    y_pre = modelR.predict(dataTes[:, ai])
    MAE = mean_absolute_error(y_test[:, 2], y_pre)
    print(MAE, ak)
    # X = modelR.feature_importances_
    # plotFeature(X, ak)
    PR = [i for i in ai if i < len(header) - 3]

    modelR.fit(dataTra[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i % (len(header) - 3) in catId])
    y_pre = modelR.predict(dataTes1[:, ai])
    MAE = mean_absolute_error(y_test1[:, 2], y_pre)
    print("NewMAE:",MAE)
    return PR#,PE, PD,

def LightGBMNew(Train, Val, TrainAll, Test, header, catId):
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    global ai, ak
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    list_to_float = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float.append(each_line)
    dataTra = np.array(list_to_float)
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    list_to_float = []
    for line in Val:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float.append(each_line)
    dataTra = np.array(list_to_float)
    X_val = dataTra[:, ai]
    y_val = dataTra[:, attribNum:attribNum + 3]
    list_to_float = []
    for line in TrainAll:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float.append(each_line)
    dataTra = np.array(list_to_float)
    X_trainAll = dataTra[:, ai]
    y_trainAll = dataTra[:, attribNum:attribNum + 3]
    list_to_float = []
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float.append(each_line)
    dataTra = np.array(list_to_float)
    X_test = dataTra[:, ai]
    y_test = dataTra[:, attribNum:attribNum + 3]

    modelR = lgb.LGBMRegressor()  # max_depth=5
    # modelR.fit(X_trainAll[:, 0:1], y_trainAll[:, 2], feature_name='0', categorical_feature='0')
    # y_pre = modelR.predict(X_test[:, 0:1])
    # MAE = mean_absolute_error(y_test[:, 2], y_pre)
    # print('NR:Activity', MAE)
    # modelR.fit(X_trainAll[:, ai], y_trainAll[:, 2], feature_name=[str(i) for i in ai],
    #            categorical_feature=[str(i) for i in ai if i in catId])
    # y_pre = modelR.predict(X_test[:, ai])
    # MAE = mean_absolute_error(y_test[:, 2], y_pre)
    # print('NR:All', MAE)

    # 后向删除消极特征
    timeS = time.time()
    ai = [i for i in hd.values()]
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = 30
    temp3 = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        # 训练模型，计算准确率
        modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
                   categorical_feature=[str(i) for i in ai if i in catId])
        y_pred = modelR.predict(X_val[:, ai])
        MAE = mean_absolute_error(y_val[:, 2], y_pred)
        # 判断准确率是否下降，若下降则更改优先级
        if temp3 != []:
            d_value[ti] = MAE - temp3[-1][0]
            if MAE > temp3[-1][0]:  # + 0.005
                temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
                           categorical_feature=[str(i) for i in ai if i in catId])
                y_pred = modelR.predict(X_val[:, ai])
                MAE = mean_absolute_error(y_val[:, 2], y_pred)
            else:
                priority.pop(ti)
                # d_value.pop(ti)
        # 删除优先级最小的属性中，重要性值最低的属性
        fi = max(modelR.feature_importances_)
        mfi = 0
        for i, j in zip(ai, range(len(ai))):
            if priority[i] == min(priority.values()):
                if fi >= modelR.feature_importances_[j]:
                    fi = modelR.feature_importances_[j]
                    mfi = j
        temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ai[mfi]])
        if min(priority.values()) > minPriority:
            if fn == len(ai):
                break
            else:
                fn = len(ai)
            minPriority = min(priority.values())
        if len(ai) == 1:
            break
        ti = ai[mfi]
        ai.remove(ai[mfi])
    timeE = time.time()
    print('NR:Step1', temp3[-1], ',特征选取时间：', timeE - timeS, len(ai))
    # 重要性值画图
    d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    X = [d_value[i][1] for i in range(len(d_value))]
    aki = [d_value[i][0] for i in range(len(d_value))]
    plotFeature(np.array(X), [ak[i] for i in aki])
    modelR.fit(X_trainAll[:, ai], y_trainAll[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
    y_pred = modelR.predict(X_test[:, ai])
    MAE = mean_absolute_error(y_test[:, 2], y_pred)
    print("New1_MAE:", MAE)
    timeS = time.time()
    ai.sort()
    ai, aiMAE = showLocalTree(Train, Val, header, ai, catId, 2)
    # 自动选取
    aii = aiMAE.index(min(aiMAE))
    for i in range(aii):
        if aiMAE[i] - aiMAE[aii] < 0.2:  # .2 BPIC2012 .5
            aii = i
            break
    FR = [aiMAE[aii-1], [header[i] for i in ai[0:aii]], ai[0:aii]]
    timeE = time.time()
    print('NR:Step2', FR, ',特征选取时间：', timeE - timeS, len(ai))
    ai = FR[2]
    modelR.fit(X_trainAll[:, ai], y_trainAll[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
    y_pred = modelR.predict(X_test[:, ai])
    MAE = mean_absolute_error(y_test[:, 2], y_pred)
    print("New2_MAE:", MAE)
    return FR  # FE, FD,

#前向后向特征选择策略
def LightGBM(Train, Test, Test1, header, catId):
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    global ai, ak
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]#-1
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]

    list_to_float2 = []
    for line in Test1:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTes = np.array(list_to_float2)
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_test1 = dataTes[:, ai]
    y_test1 = dataTes[:, attribNum:attribNum + 3]
    modelR = lgb.LGBMRegressor()  # max_depth=5
    modelR.fit(X_train[:, 0:1], y_train[:, 2], feature_name='0', categorical_feature='0')
    y_pre = modelR.predict(X_test1[:, 0:1])
    MAE = mean_absolute_error(y_test1[:, 2], y_pre)
    print('NR:Activity', MAE)
    modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
    y_pre = modelR.predict(X_test1[:, ai])
    MAE = mean_absolute_error(y_test1[:, 2], y_pre)
    print('NR:All', MAE)

    # 后向删除消极特征
    timeS = time.time()
    ai = [i for i in hd.values()]
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = 30
    temp3 = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        # 训练模型，计算准确率
        modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
        y_pred = modelR.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 2], y_pred)
        # 判断准确率是否下降，若下降则更改优先级
        if temp3 != []:
            d_value[ti] = MAE - temp3[-1][0]
            if MAE > temp3[-1][0]:  # + 0.005
                temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
                y_pred = modelR.predict(X_test[:, ai])
                MAE = mean_absolute_error(y_test[:, 2], y_pred)
            else:
                priority.pop(ti)
                # d_value.pop(ti)
        # 删除优先级最小的属性中，重要性值最低的属性
        fi = max(modelR.feature_importances_)
        mfi = 0
        for i, j in zip(ai, range(len(ai))):
            if priority[i] == min(priority.values()):
                if fi >= modelR.feature_importances_[j]:
                    fi = modelR.feature_importances_[j]
                    mfi = j
        temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ai[mfi]])
        if min(priority.values()) > minPriority:
            if fn == len(ai):
                break
            else:
                fn = len(ai)
            minPriority = min(priority.values())
        if len(ai) == 1:
            break
        ti = ai[mfi]
        ai.remove(ai[mfi])
    timeE = time.time()
    print('NR:Step1', temp3[-1],',特征选取时间：', timeE - timeS, len(ai))
    # 重要性值画图
    # d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # X = [d_value[i][1] for i in range(len(d_value))]
    # aki = [d_value[i][0] for i in range(len(d_value))]
    # plotFeature(np.array(X), [ak[i] for i in aki])
    modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
    y_pred = modelR.predict(X_test1[:, ai])
    MAE = mean_absolute_error(y_test1[:, 2], y_pred)
    print("New1_MAE:", MAE)
    timeS = time.time()
    ai.sort()
    ai, aiMAE = showLocalTree(Train, Test, header, ai, catId, 2)
    # 自动选取
    aii = aiMAE.index(min(aiMAE))
    for i in range(aii):
        if aiMAE[i] - aiMAE[aii] < 0.2:  # .1
            aii = i
            break
    FR = [aiMAE[aii], [header[i] for i in ai[0:aii+1]], ai[0:aii+1]]
    timeE = time.time()
    print('NR:Step2', FR,',特征选取时间：', timeE - timeS, len(ai))
    ai = FR[2]
    modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(i) for i in ai],
               categorical_feature=[str(i) for i in ai if i in catId])
    y_pred = modelR.predict(X_test1[:, ai])
    MAE = mean_absolute_error(y_test1[:, 2], y_pred)
    print("New2_MAE:",MAE)
    return FR #FE, FD,

def FSFilter(Train, Test, header):
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x,line1))#lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x,line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]

    # 卡方检验，检验自变量对因变量的相关性，选取前k项特征，分类问题
    # X1 = feaSel.chi2(X_train, y_train[:, 0])#.SelectKBest(chi2, k=2).fit_transform(X_train, y_train[:, 0])
    # plotFeature(X1[0], ak)

    # pearson相关系数，衡量变量之间的线性相关性，相关系数和p-value，回归问题
    X2 = []
    for i in range(len(X_train[0])):
        X2.append(pearsonr(X_train[:, i], y_train[:, 2])[1])
    plotFeature(X2, ak)

    # MIC互信息和最大信息系数，评价自变量对因变量的相关性，当零假设不成立时，MIC的统计会受到影响
    X3 = []
    m = MINE()
    for i in range(len(X_train[0])):
        m.compute_score(X_train[:, i], X_train[:, 0])
        X3.append(m.mic())
    plotFeature(X3, ak)
    X3 = []
    m = MINE()
    for i in range(len(X_train[0])):
        m.compute_score(X_train[:, i], y_train[:, 0])
        X3.append(m.mic())
    plotFeature(X3, ak)
    X3 = []
    m = MINE()
    for i in range(len(X_train[0])):
        m.compute_score(X_train[:, i], y_train[:, 1])
        X3.append(m.mic())
    plotFeature(X3, ak)
    X3 = []
    m = MINE()
    for i in range(len(X_train[0])):
        m.compute_score(X_train[:, i], y_train[:, 2])
        X3.append(m.mic())
    plotFeature(X3, ak)
    print()

#两阶段选取
def FinalFLightboost(Train, Test, header):
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    # ai = [0]
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]

    # 调参
    # lg = lgb.LGBMClassifier(silent=False)
    # param_dist = {"max_depth": [5, 7], "learning_rate": [0.01], "num_leaves": [150, 200, 250], "n_estimators": [100]}
    # grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv=3, scoring="roc_auc", verbose=5)
    # grid_search.fit(X_train, y_train[:, 0])
    # gb = grid_search.best_estimator_
    # y_pre = grid_search.predict(X_test)
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)

    # modelE = xgb.XGBClassifier()
    modelE = lgb.LGBMClassifier(learning_rate=0.01,max_depth=5,num_leaves=150,n_estimators=100)
    modelE.fit(X_train[:,[0]], y_train[:, 0])
    y_pre = modelE.predict(X_test[:,[0]])
    predictions = [round(value) for value in y_pre]
    accuracy = accuracy_score(y_test[:, 0], predictions)
    print(accuracy)
    # X = modelE.feature_importances_
    # plotFeature(X, ak)

    # modelD = xgb.XGBRegressor()
    modelD = lgb.LGBMRegressor()
    modelD.fit(X_train[:,[0,7,5,4]], y_train[:, 1])
    y_pre = modelD.predict(X_test[:,[0,7,5,4]])
    MAE = mean_absolute_error(y_test[:, 1], y_pre)
    print(MAE)
    # X2 = modelD.feature_importances_
    # plotFeature(X2, ak)

    # modelR = xgb.XGBRegressor()
    modelR = lgb.LGBMRegressor()
    modelR.fit(X_train[:,[0,6,4]], y_train[:, 2])
    y_pre = modelR.predict(X_test[:,[0,6,4]])
    MAE = mean_absolute_error(y_test[:, 2], y_pre)
    print(MAE)
    # X3 = modelR.feature_importances_
    # plotFeature(X3, ak)

    # 下一事件特征选取
    # timeS = time.time()
    # priority = {ai[i]: 0 for i in range(len(ai))}
    # d_value = {ai[i]: 0 for i in range(1, len(ai))}
    # priority[0] = 5
    # temp = []
    # ti = []
    # minPriority = 0
    # fn = len(ai)
    # while 1:
    #     #训练模型，计算准确率
    #     modelE.fit(X_train[:, ai], y_train[:, 0])
    #     y_pred = modelE.predict(X_test[:, ai])
    #     predictions = [round(value) for value in y_pred]
    #     accuracy = accuracy_score(y_test[:, 0], predictions)
    #     # 判断准确率是否下降，若下降则更改优先级
    #     if temp != []:
    #         d_value[ti] = temp[-1][0] - accuracy
    #         if accuracy < temp[-1][0]:# - 0.001
    #             temp.append([accuracy, [ak[i] for i in ai], ai.copy(), ti])
    #             priority[ti] += 1
    #             if len(ai) == 1:
    #                 break
    #             ai.append(ti)
    #             modelE.fit(X_train[:, ai], y_train[:, 0])
    #             y_pred = modelE.predict(X_test[:, ai])
    #             predictions = [round(value) for value in y_pred]
    #             accuracy = accuracy_score(y_test[:, 0], predictions)
    #         else:
    #             priority.pop(ti)
    #             d_value.pop(ti)
    #     #删除优先级最小的属性中，重要性值最低的属性
    #     fi = max(modelE.feature_importances_)
    #     mfi = 0
    #     for i, j in zip(ai,range(len(ai))):
    #         if priority[i] == min(priority.values()):
    #             if fi >= modelE.feature_importances_[j]:
    #                 fi = modelE.feature_importances_[j]
    #                 mfi = j
    #     temp.append([accuracy, [ak[i] for i in ai], ai.copy(), ai[mfi]])
    #     if min(priority.values()) > minPriority:
    #         if fn == len(ai):
    #             break
    #         else:
    #             fn = len(ai)
    #         minPriority = min(priority.values())
    #     if len(ai) == 1:
    #         break
    #     ti = ai[mfi]
    #     ai.remove(ai[mfi])
    #
    # d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # # 重要性值画图
    # X = [d_value[i][1] for i in range(len(d_value))]
    # aki = [d_value[i][0] for i in range(len(d_value))]
    # plotFeature(np.array(X), [ak[i] for i in aki])
    # ai = []
    # ai.append(0)
    # tempFE = []
    # iFE = 0
    # for i in range(1, len(temp[-1][2])+1):  # min(len(temp3[-1][2]), 6)):#
    #
    #     modelE.fit(X_train[:, ai], y_train[:, 0])
    #     y_pred = modelE.predict(X_test[:, ai])
    #     predictions = [round(value) for value in y_pred]
    #     accuracy = accuracy_score(y_test[:, 0], predictions)
    #     FE = [accuracy, [ak[j] for j in ai], ai.copy()]
    #     tempFE.append(FE)
    #     if i != 1:
    #         if accuracy > tempFE[iFE][0]:
    #             iFE = i - 1
    #         else:
    #             ai.pop(-1)
    #     if i == len(temp[-1][2]):
    #         break
    #     else:
    #         ai.append(d_value[i - 1][0])
    # timeE = time.time()
    # print('特征选取时间：', timeE - timeS)
    # # print('下一事件时间：', tempFE[iFE])
    #
    # # 下一事件时间特征选取
    # timeS = time.time()
    # ai = [i for i in hd.values()]
    # priority = {ai[i]: 0 for i in range(len(ai))}
    # d_value = {ai[i]: 0 for i in range(1, len(ai))}
    # priority[0] = 5
    # temp2 = []
    # ti = []
    # minPriority = 0
    # fn = len(ai)
    # while 1:
    #     # 训练模型，计算准确率
    #     modelD.fit(X_train[:, ai], y_train[:, 1])
    #     y_pred = modelD.predict(X_test[:, ai])
    #     MAE = mean_absolute_error(y_test[:, 1], y_pred)
    #     # 判断准确率是否下降，若下降则更改优先级
    #     if temp2 != []:
    #         d_value[ti] = MAE - temp2[-1][0]
    #         if MAE > temp2[-1][0]:# + 0.005
    #             temp2.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
    #             priority[ti] += 1
    #             ai.append(ti)
    #             modelD.fit(X_train[:, ai], y_train[:, 1])
    #             y_pred = modelD.predict(X_test[:, ai])
    #             MAE = mean_absolute_error(y_test[:, 1], y_pred)
    #         else:
    #             priority.pop(ti)
    #             d_value.pop(ti)
    #     # 删除优先级最小的属性中，重要性值最低的属性
    #     fi = max(modelD.feature_importances_)
    #     mfi = 0
    #     for i, j in zip(ai, range(len(ai))):
    #         if priority[i] == min(priority.values()):
    #             if fi >= modelD.feature_importances_[j]:
    #                 fi = modelD.feature_importances_[j]
    #                 mfi = j
    #     temp2.append([MAE, [ak[i] for i in ai], ai.copy(), ai[mfi]])
    #     if min(priority.values()) > minPriority:
    #         if fn == len(ai):
    #             break
    #         else:
    #             fn = len(ai)
    #         minPriority = min(priority.values())
    #     if len(ai) == 1:
    #         break
    #     ti = ai[mfi]
    #     ai.remove(ai[mfi])
    #
    # d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # # 重要性值画图
    # X = [d_value[i][1] for i in range(len(d_value))]
    # aki = [d_value[i][0] for i in range(len(d_value))]
    # plotFeature(np.array(X), [ak[i] for i in aki])
    # ai = []
    # ai.append(0)
    # tempFD = []
    # iFD = 0
    # for i in range(1, len(temp2[-1][2])+1):  # min(len(temp3[-1][2]), 6)):#
    #
    #     modelD.fit(X_train[:, ai], y_train[:, 1])
    #     y_pred = modelD.predict(X_test[:, ai])
    #     MAE = mean_absolute_error(y_test[:, 1], y_pred)
    #     FD = [MAE, [ak[j] for j in ai], ai.copy()]
    #     tempFD.append(FD)
    #     if i != 1:
    #         if MAE < tempFD[iFD][0]:
    #             iFD = i - 1
    #         else:
    #             ai.pop(-1)
    #     if i == len(temp2[-1][2]):
    #         break
    #     else:
    #         ai.append(d_value[i - 1][0])
    # timeE = time.time()
    # print('特征选取时间：', timeE - timeS)
    # print('下一事件时间：', tempFD[iFD])

    # 剩余时间特征选取
    timeS = time.time()
    ai = [i for i in hd.values()]
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = 5
    temp3 = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        # 训练模型，计算准确率
        modelR.fit(X_train[:, ai], y_train[:, 2])
        y_pred = modelR.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 2], y_pred)
        # 判断准确率是否下降，若下降则更改优先级
        if temp3 != []:
            d_value[ti] = MAE - temp3[-1][0]
            if MAE > temp3[-1][0]:# + 0.005
                temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                modelR.fit(X_train[:, ai], y_train[:, 2])
                y_pred = modelR.predict(X_test[:, ai])
                MAE = mean_absolute_error(y_test[:, 2], y_pred)
            else:
                priority.pop(ti)
                d_value.pop(ti)
        # 删除优先级最小的属性中，重要性值最低的属性
        fi = max(modelR.feature_importances_)
        mfi = 0
        for i, j in zip(ai, range(len(ai))):
            if priority[i] == min(priority.values()):
                if fi >= modelR.feature_importances_[j]:
                    fi = modelR.feature_importances_[j]
                    mfi = j
        temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ai[mfi]])
        if min(priority.values()) > minPriority:
            if fn == len(ai):
                break
            else:
                fn = len(ai)
            minPriority = min(priority.values())
        if len(ai) == 1:
            break
        ti = ai[mfi]
        ai.remove(ai[mfi])
    timeE = time.time()
    print('特征选取时间：', timeE - timeS)
    d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    #重要性值画图
    X = [d_value[i][1] for i in range(len(d_value))]
    aki = [d_value[i][0] for i in range(len(d_value))]
    plotFeature(np.array(X), [ak[i] for i in aki])
    ai = []
    ai.append(0)
    tempFR = []
    iFR = 0
    for i in range(1, len(temp3[-1][2])+1):#min(len(temp3[-1][2]), 6)):#
        modelR.fit(X_train[:, ai], y_train[:, 2])
        y_pred = modelR.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 2], y_pred)
        FR = [MAE, [ak[j] for j in ai], ai.copy()]
        tempFR.append(FR)
        if i != 1:
            if MAE < tempFR[iFR][0]:
                iFR = i-1
            else:
                ai.pop(-1)
        if i == len(temp3[-1][2]):
            break
        else:
            ai.append(d_value[i - 1][0])
    timeE = time.time()
    print('特征选取时间：',timeE-timeS)
    print('剩余时间：', tempFR[iFR])
    return tempFR[iFR]#temp[-1],temp2[-1],temp3[-1]#tempFE[iFE],tempFD[iFD],

#top6选取
def NewFLightboost(Train, Test, header):
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]

    # 调参
    # lg = lgb.LGBMClassifier(silent=False)
    # param_dist = {"max_depth": [5, 7], "learning_rate": [0.01], "num_leaves": [150, 200, 250], "n_estimators": [100]}
    # grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv=3, scoring="roc_auc", verbose=5)
    # grid_search.fit(X_train, y_train[:, 0])
    # gb = grid_search.best_estimator_
    # y_pre = grid_search.predict(X_test)
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)


    modelE = lgb.LGBMClassifier(learning_rate=0.01,max_depth=5,num_leaves=150,n_estimators=100)
    # modelE.fit(X_train, y_train[:, 0])
    # y_pre = modelE.predict(X_test)
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)
    # print(accuracy)
    # X = modelE.feature_importances_
    # plotFeature(X, ak)

    modelD = lgb.LGBMRegressor()
    # modelD.fit(X_train, y_train[:, 1])
    # y_pre = modelD.predict(X_test)
    # MAE = mean_absolute_error(y_test[:, 1], y_pre)
    # print(MAE)
    # X2 = modelD.feature_importances_
    # plotFeature(X2, ak)

    modelR = lgb.LGBMRegressor()
    # modelR.fit(X_train, y_train[:, 2])
    # y_pre = modelR.predict(X_test)
    # MAE = mean_absolute_error(y_test[:, 2], y_pre)
    # print(MAE)
    # X3 = modelR.feature_importances_
    # plotFeature(X3, ak)

    # 下一事件特征选取
    timeS = time.time()
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = 5
    temp = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        #训练模型，计算准确率
        modelE.fit(X_train[:, ai], y_train[:, 0])
        y_pred = modelE.predict(X_test[:, ai])
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test[:, 0], predictions)
        # 判断准确率是否下降，若下降则更改优先级
        if temp != []:
            d_value[ti] = temp[-1][0] - accuracy
            if accuracy < temp[-1][0] - 0.001:
                temp.append([accuracy, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                modelE.fit(X_train[:, ai], y_train[:, 0])
                y_pred = modelE.predict(X_test[:, ai])
                predictions = [round(value) for value in y_pred]
                accuracy = accuracy_score(y_test[:, 0], predictions)
            else:
                priority.pop(ti)
                d_value.pop(ti)
        #删除优先级最小的属性中，重要性值最低的属性
        fi = max(modelE.feature_importances_)
        mfi = 0
        for i, j in zip(ai,range(len(ai))):
            if priority[i] == min(priority.values()):
                if fi >= modelE.feature_importances_[j]:
                    fi = modelE.feature_importances_[j]
                    mfi = j
        temp.append([accuracy, [ak[i] for i in ai], ai.copy(), ai[mfi]])
        if min(priority.values()) > minPriority:
            if fn == len(ai):
                break
            else:
                fn = len(ai)
            minPriority = min(priority.values())
        ti = ai[mfi]
        ai.remove(ai[mfi])
    d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)
    ai = []
    ai.append(0)
    for i in range(1, min(len(temp[-1][2]), 6)):#len(temp[-1][2])):#
        ai.append(d_value[i - 1][0])
    modelE.fit(X_train[:, ai], y_train[:, 0])
    y_pred = modelE.predict(X_test[:, ai])
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test[:, 0], predictions)
    FE = [accuracy, [ak[i] for i in ai], ai.copy()]
    timeE = time.time()
    print('特征选取时间：', timeE - timeS)
    print('下一事件：', FE)

    # 下一事件时间特征选取
    timeS = time.time()
    ai = [i for i in hd.values()]
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = 5
    temp2 = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        # 训练模型，计算准确率
        modelD.fit(X_train[:, ai], y_train[:, 1])
        y_pred = modelD.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 1], y_pred)
        # 判断准确率是否下降，若下降则更改优先级
        if temp2 != []:
            d_value[ti] = MAE - temp2[-1][0]
            if MAE > temp2[-1][0] + 0.005:
                temp2.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                modelD.fit(X_train[:, ai], y_train[:, 1])
                y_pred = modelD.predict(X_test[:, ai])
                MAE = mean_absolute_error(y_test[:, 1], y_pred)
            else:
                priority.pop(ti)
                d_value.pop(ti)
        # 删除优先级最小的属性中，重要性值最低的属性
        fi = max(modelD.feature_importances_)
        mfi = 0
        for i, j in zip(ai, range(len(ai))):
            if priority[i] == min(priority.values()):
                if fi >= modelD.feature_importances_[j]:
                    fi = modelD.feature_importances_[j]
                    mfi = j
        temp2.append([MAE, [ak[i] for i in ai], ai.copy(), ai[mfi]])
        if min(priority.values()) > minPriority:
            if fn == len(ai):
                break
            else:
                fn = len(ai)
            minPriority = min(priority.values())
        ti = ai[mfi]
        ai.remove(ai[mfi])
    d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    ai = []
    ai.append(0)
    for i in range(1, min(len(temp2[-1][2]), 6)):#len(temp2[-1][2])):#
        ai.append(d_value[i - 1][0])
    modelD.fit(X_train[:, ai], y_train[:, 1])
    y_pred = modelD.predict(X_test[:, ai])
    MAE = mean_absolute_error(y_test[:, 1], y_pred)
    FD = [MAE, [ak[i] for i in ai], ai.copy()]
    timeE = time.time()
    print('特征选取时间：', timeE - timeS)
    print('下一事件时间：', FD)

    # 剩余时间特征选取
    timeS = time.time()
    ai = [i for i in hd.values()]
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = 5
    temp3 = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        # 训练模型，计算准确率
        modelR.fit(X_train[:, ai], y_train[:, 2])
        y_pred = modelR.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 2], y_pred)
        # 判断准确率是否下降，若下降则更改优先级
        if temp3 != []:
            d_value[ti] = MAE - temp3[-1][0]
            if MAE > temp3[-1][0]:# + 0.005
                temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                modelR.fit(X_train[:, ai], y_train[:, 2])
                y_pred = modelR.predict(X_test[:, ai])
                MAE = mean_absolute_error(y_test[:, 2], y_pred)
            else:
                priority.pop(ti)
                d_value.pop(ti)
        # 删除优先级最小的属性中，重要性值最低的属性
        fi = max(modelR.feature_importances_)
        mfi = 0
        for i, j in zip(ai, range(len(ai))):
            if priority[i] == min(priority.values()):
                if fi >= modelR.feature_importances_[j]:
                    fi = modelR.feature_importances_[j]
                    mfi = j
        temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ai[mfi]])
        if min(priority.values()) > minPriority:
            if fn == len(ai):
                break
            else:
                fn = len(ai)
            minPriority = min(priority.values())
        ti = ai[mfi]
        ai.remove(ai[mfi])

    d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    #重要性值画图
    X = [d_value[i][1] for i in range(len(d_value))]
    aki = [d_value[i][0] for i in range(len(d_value))]
    plotFeature(np.array(X), [ak[i] for i in aki])
    ai = []
    ai.append(0)
    tempFR = []
    iFR = 0
    for i in range(1, len(temp3[-1][2])):#min(len(temp3[-1][2]), 6)):#
        ai.append(d_value[i - 1][0])
        modelR.fit(X_train[:, ai], y_train[:, 2])
        y_pred = modelR.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 2], y_pred)
        FR = [MAE, [ak[j] for j in ai], ai.copy()]
        tempFR.append(FR)
        if i != 1:
            if MAE < tempFR[iFR][0]:
                iFR = i-1
            else:
                ai.pop(-1)
    timeE = time.time()
    print('特征选取时间：',timeE-timeS)
    print('剩余时间：', tempFR[iFR])
    return FE,FD,FR

#递减选取
def FLightboost(Train, Test, header):
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]

    # 调参
    # lg = lgb.LGBMClassifier(silent=False)
    # param_dist = {"max_depth": [5, 7], "learning_rate": [0.01], "num_leaves": [150, 200, 250], "n_estimators": [100]}
    # grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv=3, scoring="roc_auc", verbose=5)
    # grid_search.fit(X_train, y_train[:, 0])
    # gb = grid_search.best_estimator_
    # y_pre = grid_search.predict(X_test)
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)

    timeS = time.time()
    modelE = lgb.LGBMClassifier(learning_rate=0.01,max_depth=5,num_leaves=150,n_estimators=100)
    modelE.fit(X_train, y_train[:, 0])
    y_pre = modelE.predict(X_test)
    predictions = [round(value) for value in y_pre]
    accuracy = accuracy_score(y_test[:, 0], predictions)
    print(accuracy)
    X = modelE.feature_importances_
    plotFeature(X, ak)

    modelD = lgb.LGBMRegressor()
    modelD.fit(X_train, y_train[:, 1])
    y_pre = modelD.predict(X_test)
    MAE = mean_absolute_error(y_test[:, 1], y_pre)
    print(MAE)
    X2 = modelD.feature_importances_
    plotFeature(X2, ak)

    modelR = lgb.LGBMRegressor()
    modelR.fit(X_train, y_train[:, 2])
    y_pre = modelR.predict(X_test)
    MAE = mean_absolute_error(y_test[:, 2], y_pre)
    print(MAE)
    X3 = modelR.feature_importances_
    plotFeature(X3, ak)

    # 下一事件特征选取
    temp = []
    for i in range(attribNum):
        modelE.fit(X_train[:, ai], y_train[:, 0])
        y_pred = modelE.predict(X_test[:, ai])
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test[:, 0], predictions)
        temp.append([accuracy, ak.copy(), ai.copy()])
        if len(ak) == 1:
            break
        fi = modelE.feature_importances_[1:]
        mfi = np.argmin(fi) + 1
        ak.remove(ak[mfi])
        ai.remove(ai[mfi])
    maxi = np.argmax(np.array(temp)[:, 0])
    # for i in range(maxi+1, attribNum - 1):
    #     if temp[i][0] - temp[maxi][0] < 0.0005:
    #         maxi = i

    # 下一事件时间特征选取
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    temp2 = []
    for i in range(attribNum):
        modelD.fit(X_train[:, ai], y_train[:, 1])
        y_pred = modelD.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 1], y_pred)
        temp2.append([MAE, ak.copy(), ai.copy()])
        if len(ak) == 1:
            break
        fi = modelD.feature_importances_[1:]
        mfi = np.argmin(fi) + 1
        ak.remove(ak[mfi])
        ai.remove(ai[mfi])
    mini = np.argmin(np.array(temp2)[:, 0])
    # for i in range(mini+1, attribNum - 1):
    #     if temp2[mini][0] - temp2[i][0] < 0.0005:
    #         mini = i

    # 剩余时间特征选取
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    temp3 = []
    for i in range(attribNum):
        modelR.fit(X_train[:, ai], y_train[:, 2])
        y_pred = modelR.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 2], y_pred)
        temp3.append([MAE, ak.copy(), ai.copy()])
        if len(ak) == 1:
            break
        fi = modelR.feature_importances_[1:]
        mfi = np.argmin(fi) + 1
        ak.remove(ak[mfi])
        ai.remove(ai[mfi])
    mini2 = np.argmin(np.array(temp3)[:, 0])
    # for i in range(mini2+1, attribNum - 1):
    #     if temp3[mini2][0] - temp3[i][0] < 0.0005:
    #         mini2 = i
    timeE = time.time()
    print(timeE-timeS)
    return temp[maxi], temp2[mini], temp3[mini2]

def FCatboost(Train, Test, header):
    # attribNum = len(header) - 3
    # hd = {header[i]: i for i in range(attribNum)}
    # ai = [i for i in hd.values()]
    # ak = [i for i in hd]
    # X_train = []
    # Y_trainE = []
    # Y_trainD = []
    # Y_trainR = []
    # X_test = []
    # Y_testE = []
    # Y_testD = []
    # Y_testR = []
    # for line in Train:
    #     for line1 in line:
    #         X_train.append(line1[0:attribNum])
    #         Y_trainE.append(line1[attribNum])
    #         Y_trainD.append(line1[attribNum+1])
    #         Y_trainR.append(line1[attribNum+2])
    # for line in Test:
    #     for line1 in line:
    #         X_test.append(line1[0:attribNum])
    #         Y_testE.append(line1[attribNum])
    #         Y_testD.append(line1[attribNum+1])
    #         Y_testR.append(line1[attribNum+2])

    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]
    # params = {'depth': [4, 7, 10],
    #           'learning_rate': [0.03, 0.1, 0.15],
    #           'l2_leaf_reg': [1, 4, 9],
    #           'iterations': [300]}
    # cb = CatBoostClassifier()
    # cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv=3)
    # cb_model.fit(X_train, Y_trainE)
    # pre_y = cb_model.predict(X_test)
    # accuracy = accuracy_score(Y_testE, pre_y)
    # print(accuracy)
    cat_features_index = [0]
    timeS = time.time()
    modelE = CatBoostClassifier()#iterations=500, one_hot_max_size=30
    modelE.fit(X_train[:,[0]], y_train[:, 0], verbose=False)#, cat_features=cat_features_index
    pre_y = modelE.predict(X_test[:,[0]])
    accuracy = accuracy_score(y_test[:, 0], pre_y)
    print(accuracy)
    X4 = modelE.get_feature_importance()
    plotFeature(X4, ak)

    modelD = CatBoostRegressor()
    modelD.fit(X_train[:,[0,7,5,4]], y_train[:, 1], verbose=False)
    pre_y = modelD.predict(X_test[:,[0,7,5,4]])
    MAE = mean_absolute_error(y_test[:, 1], pre_y)
    print(MAE)
    X4 = modelD.get_feature_importance()
    plotFeature(X4, ak)

    modelR = CatBoostRegressor()
    modelR.fit(X_train[:,[0,6,4]], y_train[:, 2], verbose=False)
    pre_y = modelR.predict(X_test[:,[0,6,4]])
    MAE = mean_absolute_error(y_test[:, 2], pre_y)
    print(MAE)
    X4 = modelR.get_feature_importance()
    plotFeature(X4, ak)

    # 下一事件特征选取
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = 5
    temp = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        # 训练模型，计算准确率
        modelE.fit(X_train[:, ai], y_train[:, 0])
        y_pred = modelE.predict(X_test[:, ai])
        accuracy = accuracy_score(y_test[:, 0], y_pred)
        # 判断准确率是否下降，若下降则更改优先级
        if temp != []:
            d_value[ti] = temp[-1][0] - accuracy
            if accuracy < temp[-1][0]:  # - 0.001
                temp.append([accuracy, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                modelE.fit(X_train[:, ai], y_train[:, 0])
                y_pred = modelE.predict(X_test[:, ai])
                accuracy = accuracy_score(y_test[:, 0], y_pred)
            else:
                priority.pop(ti)
                d_value.pop(ti)
        # 删除优先级最小的属性中，重要性值最低的属性
        fi = max(modelE.feature_importances_)
        mfi = 0
        for i, j in zip(ai, range(len(ai))):
            if priority[i] == min(priority.values()):
                if fi >= modelE.feature_importances_[j]:
                    fi = modelE.feature_importances_[j]
                    mfi = j
        temp.append([accuracy, [ak[i] for i in ai], ai.copy(), ai[mfi]])
        if min(priority.values()) > minPriority:
            if fn == len(ai):
                break
            else:
                fn = len(ai)
            minPriority = min(priority.values())
        if len(ai) == 1:
            break
        ti = ai[mfi]
        ai.remove(ai[mfi])

    d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # 重要性值画图
    X = [d_value[i][1] for i in range(len(d_value))]
    aki = [d_value[i][0] for i in range(len(d_value))]
    plotFeature(np.array(X), [ak[i] for i in aki])
    # ai = []
    # ai.append(0)
    # tempFE = []
    # iFE = 0
    # for i in range(1, len(temp[-1][2]) + 1):  # min(len(temp3[-1][2]), 6)):#
    #
    #     modelE.fit(X_train[:, ai], y_train[:, 0])
    #     y_pred = modelE.predict(X_test[:, ai])
    #     accuracy = accuracy_score(y_test[:, 0], y_pred)
    #     FE = [accuracy, [ak[j] for j in ai], ai.copy()]
    #     tempFE.append(FE)
    #     if i != 1:
    #         if accuracy > tempFE[iFE][0]:
    #             iFE = i - 1
    #         else:
    #             ai.pop(-1)
    #     if i == len(temp[-1][2]):
    #         break
    #     else:
    #         ai.append(d_value[i - 1][0])
    # print('下一事件时间：', tempFE[iFE])

    # 下一事件时间特征选取
    ai = [i for i in hd.values()]
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = 5
    temp2 = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        # 训练模型，计算准确率
        modelD.fit(X_train[:, ai], y_train[:, 1])
        y_pred = modelD.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 1], y_pred)
        # 判断准确率是否下降，若下降则更改优先级
        if temp2 != []:
            d_value[ti] = MAE - temp2[-1][0]
            if MAE > temp2[-1][0]:  # + 0.005
                temp2.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                modelD.fit(X_train[:, ai], y_train[:, 1])
                y_pred = modelD.predict(X_test[:, ai])
                MAE = mean_absolute_error(y_test[:, 1], y_pred)
            else:
                priority.pop(ti)
                d_value.pop(ti)
        # 删除优先级最小的属性中，重要性值最低的属性
        fi = max(modelD.feature_importances_)
        mfi = 0
        for i, j in zip(ai, range(len(ai))):
            if priority[i] == min(priority.values()):
                if fi >= modelD.feature_importances_[j]:
                    fi = modelD.feature_importances_[j]
                    mfi = j
        temp2.append([MAE, [ak[i] for i in ai], ai.copy(), ai[mfi]])
        if min(priority.values()) > minPriority:
            if fn == len(ai):
                break
            else:
                fn = len(ai)
            minPriority = min(priority.values())
        if len(ai) == 1:
            break
        ti = ai[mfi]
        ai.remove(ai[mfi])

    d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # 重要性值画图
    X = [d_value[i][1] for i in range(len(d_value))]
    aki = [d_value[i][0] for i in range(len(d_value))]
    plotFeature(np.array(X), [ak[i] for i in aki])
    # ai = []
    # ai.append(0)
    # tempFD = []
    # iFD = 0
    # for i in range(1, len(temp2[-1][2]) + 1):  # min(len(temp3[-1][2]), 6)):#
    #
    #     modelD.fit(X_train[:, ai], y_train[:, 1])
    #     y_pred = modelD.predict(X_test[:, ai])
    #     MAE = mean_absolute_error(y_test[:, 1], y_pred)
    #     FD = [MAE, [ak[j] for j in ai], ai.copy()]
    #     tempFD.append(FD)
    #     if i != 1:
    #         if MAE < tempFD[iFD][0]:
    #             iFD = i - 1
    #         else:
    #             ai.pop(-1)
    #     if i == len(temp2[-1][2]):
    #         break
    #     else:
    #         ai.append(d_value[i - 1][0])
    # print('下一事件时间：', tempFD[iFD])

    # 剩余时间特征选取
    ai = [i for i in hd.values()]
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = 5
    temp3 = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        # 训练模型，计算准确率
        modelR.fit(X_train[:, ai], y_train[:, 2])
        y_pred = modelR.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 2], y_pred)
        # 判断准确率是否下降，若下降则更改优先级
        if temp3 != []:
            d_value[ti] = MAE - temp3[-1][0]
            if MAE > temp3[-1][0]:  # + 0.005
                temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                modelR.fit(X_train[:, ai], y_train[:, 2])
                y_pred = modelR.predict(X_test[:, ai])
                MAE = mean_absolute_error(y_test[:, 2], y_pred)
            else:
                priority.pop(ti)
                d_value.pop(ti)
        # 删除优先级最小的属性中，重要性值最低的属性
        fi = max(modelR.feature_importances_)
        mfi = 0
        for i, j in zip(ai, range(len(ai))):
            if priority[i] == min(priority.values()):
                if fi >= modelR.feature_importances_[j]:
                    fi = modelR.feature_importances_[j]
                    mfi = j
        temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ai[mfi]])
        if min(priority.values()) > minPriority:
            if fn == len(ai):
                break
            else:
                fn = len(ai)
            minPriority = min(priority.values())
        if len(ai) == 1:
            break
        ti = ai[mfi]
        ai.remove(ai[mfi])

    d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # 重要性值画图
    X = [d_value[i][1] for i in range(len(d_value))]
    aki = [d_value[i][0] for i in range(len(d_value))]
    plotFeature(np.array(X), [ak[i] for i in aki])
    # ai = []
    # ai.append(0)
    # tempFR = []
    # iFR = 0
    # for i in range(1, len(temp3[-1][2]) + 1):  # min(len(temp3[-1][2]), 6)):#
    #     modelR.fit(X_train[:, ai], y_train[:, 2])
    #     y_pred = modelR.predict(X_test[:, ai])
    #     MAE = mean_absolute_error(y_test[:, 2], y_pred)
    #     FR = [MAE, [ak[j] for j in ai], ai.copy()]
    #     tempFR.append(FR)
    #     if i != 1:
    #         if MAE < tempFR[iFR][0]:
    #             iFR = i - 1
    #         else:
    #             ai.pop(-1)
    #     if i == len(temp3[-1][2]):
    #         break
    #     else:
    #         ai.append(d_value[i - 1][0])
    # print('剩余时间：', tempFR[iFR])
    timeE = time.time()
    print('特征选取时间：',timeE-timeS)
    return temp[-1],temp2[-1],temp3[-1]#tempFE[iFE], tempFD[iFD], tempFR[iFR]
    # # 下一事件特征选取
    # temp = []
    # for i in range(attribNum):
    #     modelE.fit(X_train[:, ai], y_train[:, 0], verbose=False)
    #     y_pred = modelE.predict(X_test[:, ai])
    #     accuracy = accuracy_score(y_test[:, 0], y_pred)
    #     temp.append([accuracy, ak.copy(), ai.copy()])
    #     if len(ak) == 1:
    #         break
    #     fi = modelE.feature_importances_[1:]
    #     mfi = np.argmin(fi) + 1
    #     ak.remove(ak[mfi])
    #     ai.remove(ai[mfi])
    # maxi = np.argmax(np.array(temp)[:, 0])
    # for i in range(maxi + 1, attribNum - 1):
    #     if temp[maxi][0] - temp[i][0] < 0.0005:
    #         maxi = i
    #
    # # 下一事件时间特征选取
    # ai = [i for i in hd.values()]
    # ak = [i for i in hd]
    # temp2 = []
    # for i in range(attribNum):
    #     modelD.fit(X_train[:, ai], y_train[:, 1], verbose=False)
    #     y_pred = modelD.predict(X_test[:, ai])
    #     MAE = mean_absolute_error(y_test[:, 1], y_pred)
    #     temp2.append([MAE, ak.copy(), ai.copy()])
    #     if len(ak) == 1:
    #         break
    #     fi = modelD.feature_importances_[1:]
    #     mfi = np.argmin(fi) + 1
    #     ak.remove(ak[mfi])
    #     ai.remove(ai[mfi])
    # mini = np.argmin(np.array(temp2)[:, 0])
    # for i in range(mini + 1, attribNum - 1):
    #     if temp2[i][0] - temp2[mini][0] < 0.0005:
    #         mini = i
    #
    # # 剩余时间特征选取
    # ai = [i for i in hd.values()]
    # ak = [i for i in hd]
    # temp3 = []
    # for i in range(attribNum):
    #     modelR.fit(X_train[:, ai], y_train[:, 2], verbose=False)
    #     y_pred = modelR.predict(X_test[:, ai])
    #     MAE = mean_absolute_error(y_test[:, 2], y_pred)
    #     temp3.append([MAE, ak.copy(), ai.copy()])
    #     if len(ak) == 1:
    #         break
    #     fi = modelR.feature_importances_[1:]
    #     mfi = np.argmin(fi) + 1
    #     ak.remove(ak[mfi])
    #     ai.remove(ai[mfi])
    # mini2 = np.argmin(np.array(temp3)[:, 0])
    # for i in range(mini2 + 1, attribNum - 1):
    #     if temp3[i][0] - temp3[mini2][0] < 0.0005:
    #         mini2 = i
    # timeE = time.time()
    # print(timeE - timeS)
    # return temp[maxi], temp2[mini], temp3[mini2]

def FSXGboost(Train, Test, header):
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: float(x), line1))
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: float(x), line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum+3]
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum+3]
    #划分数据集
    # seed = 7
    # test_size = 0.33
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    #画重要值属性图
    # ak = ['活动','状态','资源','执行时间','总执行时间','月份','日期','星期','时间点']
    # model = xgb.XGBClassifier()
    # param_dist = {"max_depth": [5, 7, 10], "min_child_weight": [1, 3, 6], "n_estimators": [50,100,150],  "learning_rate": [0.01, 0.05, 0.1], }
    # grid_search = GridSearchCV(model, param_grid=param_dist, cv=3, verbose=10, n_jobs=-1)
    # grid_search.fit(X_train, y_train[:, 0])
    # gb = grid_search.best_estimator_
    # y_pre = grid_search.predict(X_test)
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)
    #0.9833 5 1 150 0.05
    timeS = time.time()
    modelE = xgb.XGBClassifier()
    modelE.fit(X_train[:, ai], y_train[:, 0])
    X = modelE.feature_importances_
    plotFeature(X, ak)
    # explainer = shap.TreeExplainer(modelE)
    # shap_values = explainer.shap_values(X_train[:, ai])
    # shap.plots.bar(shap_values, max_display=10)
    # shap.initjs()
    # shap.plots.force(explainer.expected_value[0], shap_values[0], X_train[:, ai])
    # shap.summary_plot(shap_values, X_train[:, ai])
    # shap.summary_plot(shap_values, X_train[:, ai], plot_type="bar")
    # shap.dependence_plot('Feature 0', shap_values[0], X_train[:, ai], interaction_index=None, show=False)
    # shap_interaction_values = shap.TreeExplainer(modelE).shap_interaction_values(X_train[:, ai])
    # shap.summary_plot(shap_interaction_values, X_train[:, ai], max_display=4)
    # shap.dependence_plot('potential', shap_values, X_train[:, ai], interaction_index='international_reputation', show=False)
    # print()
    modelE.get_booster().feature_names = ak
    plot_importance(modelE, importance_type='gain',xlim=tuple([0,80]),title='特征重要性',xlabel='F值', ylabel='特征名')
    modelD = xgb.XGBRegressor()
    modelD.fit(X_train[:, ai], y_train[:, 1])
    X = modelD.feature_importances_
    plotFeature(X, ak)
    modelD.get_booster().feature_names = ak
    plot_importance(modelD, importance_type='gain',xlim=tuple([0, 50]),title='特征重要性',xlabel='F值', ylabel='特征名')
    modelR = xgb.XGBRegressor()
    modelR.fit(X_train[:, ai], y_train[:, 2])
    X = modelR.feature_importances_
    plotFeature(X, ak)
    modelR.get_booster().feature_names = ak
    plot_importance(modelR, importance_type='gain',xlim=tuple([0, 360]),title='特征重要性',xlabel='F值', ylabel='特征名')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

    #下一事件特征选取
    temp = []
    for i in range(attribNum):
        modelE.fit(X_train[:,ai], y_train[:,0])
        modelE.get_booster().feature_names = ak
        y_pred = modelE.predict(X_test[:,ai])
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test[:,0], predictions)
        temp.append([accuracy, ak.copy(), ai.copy()])
        if len(ak)==1:
            break
        fi = modelE.feature_importances_[1:]
        mfi = np.argmin(fi)+1
        ak.remove(ak[mfi])
        ai.remove(ai[mfi])
    maxi = np.argmax(np.array(temp)[:,0])
    # for i in range(maxi,attribNum-1):
    #     if temp[i][0] - temp[i+1][0] < 0.0005:
    #         maxi = i+1
    for i in range(maxi+1, attribNum - 1):
        if temp[maxi][0] - temp[i][0] < 0.0005:
            maxi = i

    # 下一事件时间特征选取
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    temp2 = []
    for i in range(attribNum):
        modelD.fit(X_train[:, ai], y_train[:, 1])
        modelD.get_booster().feature_names = ak
        y_pred = modelD.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 1], y_pred)
        temp2.append([MAE, ak.copy(), ai.copy()])
        if len(ak)==1:
            break
        fi = modelD.feature_importances_[1:]
        mfi = np.argmin(fi)+1
        ak.remove(ak[mfi])
        ai.remove(ai[mfi])
    mini = np.argmin(np.array(temp2)[:, 0])
    for i in range(mini+1, attribNum - 1):
        if temp2[i][0] - temp2[mini][0] < 0.0005:
            mini = i

    # 剩余时间特征选取
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    temp3 = []
    for i in range(attribNum):
        modelR.fit(X_train[:, ai], y_train[:, 2])
        modelR.get_booster().feature_names = ak
        y_pred = modelR.predict(X_test[:, ai])
        MAE = mean_absolute_error(y_test[:, 2], y_pred)
        temp3.append([MAE, ak.copy(), ai.copy()])
        if len(ak)==1:
            break
        fi = modelR.feature_importances_[1:]
        mfi = np.argmin(fi)+1

        ak.remove(ak[mfi])
        ai.remove(ai[mfi])
    mini2 = np.argmin(np.array(temp3)[:, 0])
    for i in range(mini2+1, attribNum - 1):
        if temp3[i][0] - temp3[mini2][0] < 0.0005:
            mini2 = i
    timeE = time.time()
    print(timeE - timeS)
    return temp[maxi],temp2[mini],temp3[mini2]

#特征遍历树
def AllFLightboost(Train, Test, header, catId, task):
    cid = catId.copy()
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    # ai = [0]
    global ai,ak
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]

    # 调参
    # lg = lgb.LGBMClassifier(silent=False)
    # param_dist = {"max_depth": [5, 7], "learning_rate": [0.01], "num_leaves": [150, 200, 250], "n_estimators": [100]}
    # grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv=3, scoring="roc_auc", verbose=5)
    # grid_search.fit(X_train, y_train[:, 0])
    # gb = grid_search.best_estimator_
    # y_pre = grid_search.predict(X_test)
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)

    # modelR = xgb.XGBRegressor()
    if task == 0:
        modelR = lgb.LGBMClassifier(learning_rate=0.01, max_depth=5, num_leaves=150, n_estimators=100)
    else:
        modelR = lgb.LGBMRegressor()
    modelR.fit(X_train[:, 0:1], y_train[:, task], feature_name='0', categorical_feature='0')
    y_pre = modelR.predict(X_test[:, 0:1])
    if task == 0:
        predictions = [round(value) for value in y_pre]
        MAE = accuracy_score(y_test[:, task], predictions)
    else:
        MAE = mean_absolute_error(y_test[:, task], y_pre)
    print('Activity', MAE)

    modelR.fit(X_train[:,ai], y_train[:, task], feature_name=[str(ai[i]) for i in range(len(ai))],
               categorical_feature=[str(cid[i]) for i in range(len(cid))])
    # ax = lgb.plot_split_value_histogram(modelR, feature='Column_1', bins='auto')
    # plt.show()#画直方图
    y_pre = modelR.predict(X_test[:, ai])
    if task == 0:
        predictions = [round(value) for value in y_pre]
        MAE = accuracy_score(y_test[:, task], predictions)
    else:
        MAE = mean_absolute_error(y_test[:, task], y_pre)
    print('All', MAE)
    # X3 = modelR.feature_importances_
    # plotFeature(X3, ak)

    # 后向删除消极特征
    timeS = time.time()
    ai = [i for i in hd.values()]
    priority = {ai[i]: 0 for i in range(len(ai))}
    d_value = {ai[i]: 0 for i in range(1, len(ai))}
    priority[0] = float('inf')
    temp3 = []
    ti = []
    minPriority = 0
    fn = len(ai)
    while 1:
        # 训练模型，计算准确率
        modelR.fit(X_train[:, ai], y_train[:, task], feature_name=[str(ai[i]) for i in range(len(ai))],
               categorical_feature=[str(cid[i]) for i in range(len(cid))])
        y_pre = modelR.predict(X_test[:, ai])
        if task == 0:
            predictions = [round(value) for value in y_pre]
            MAE = accuracy_score(y_test[:, task], predictions)
        else:
            MAE = mean_absolute_error(y_test[:, task], y_pre)
        # 判断准确率是否下降，若下降则更改优先级
        if temp3 != []:
            if task == 0:
                d_value[ti] = temp3[-1][0] - MAE
            else:
                d_value[ti] = MAE - temp3[-1][0]
            if (MAE > temp3[-1][0] and task != 0) or (MAE < temp3[-1][0] and task == 0):  # + 0.005
                temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ti])
                priority[ti] += 1
                ai.append(ti)
                if ti in catId:
                    cid.append(ti)
                modelR.fit(X_train[:, ai], y_train[:, task], feature_name=[str(ai[i]) for i in range(len(ai))],
                    categorical_feature=[str(cid[i]) for i in range(len(cid))])
                y_pre = modelR.predict(X_test[:, ai])
                if task == 0:
                    predictions = [round(value) for value in y_pre]
                    MAE = accuracy_score(y_test[:, task], predictions)
                else:
                    MAE = mean_absolute_error(y_test[:, task], y_pre)
            else:
                priority.pop(ti)
                # d_value.pop(ti)
        # 删除优先级最小的属性中，重要性值最低的属性
        fi = max(modelR.feature_importances_)
        mfi = 0
        for i, j in zip(ai, range(len(ai))):
            if priority[i] == min(priority.values()):
                if fi >= modelR.feature_importances_[j]:
                    fi = modelR.feature_importances_[j]
                    mfi = j
        temp3.append([MAE, [ak[i] for i in ai], ai.copy(), ai[mfi]])
        if min(priority.values()) > minPriority:
            if fn == len(ai):
                break
            else:
                fn = len(ai)
            minPriority = min(priority.values())
        if len(ai) == 1:
            break
        ti = ai[mfi]
        ai.remove(ai[mfi])
        if ti in catId:
            cid.remove(ti)
    timeE = time.time()
    print('Step1特征选取时间：', timeE - timeS, len(ai))
    print('MAE：', temp3[-1])
    d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # 重要性值画图
    X = [d_value[i][1] for i in range(len(d_value))]
    aki = [d_value[i][0] for i in range(len(d_value))]
    plotFeature(np.array(X), [ak[i] for i in aki])
    ai.sort()
    print(ai)
    FR = showLocalTree(Train, Test, header, ai, cid, task)
    return FR
    # timeS = time.time()
    # d_value = sorted(d_value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # # 重要性值画图
    # X = [d_value[i][1] for i in range(len(d_value))]
    # aki = [d_value[i][0] for i in range(len(d_value))]
    # plotFeature(np.array(X), [ak[i] for i in aki])
    # ai = []
    # cid = []
    # ai.append(0)
    # cid.append(0)
    # tempFR = []
    # iFR = 0
    # for i in range(1, len(temp3[-1][2]) + 1):  # min(len(temp3[-1][2]), 6)):#
    #     modelR.fit(X_train[:, ai], y_train[:, 2], feature_name=[str(ai[i]) for i in range(len(ai))],
    #                 categorical_feature=[str(cid[i]) for i in range(len(cid))])
    #     y_pred = modelR.predict(X_test[:, ai])
    #     MAE = mean_absolute_error(y_test[:, 2], y_pred)
    #     FR = [MAE, [ak[j] for j in ai], ai.copy()]
    #     tempFR.append(FR)
    #     if i != 1:
    #         if MAE < tempFR[iFR][0]:
    #             iFR = i - 1
    #         else:
    #             if ai[-1] in cid:
    #                 cid.pop(-1)
    #             ai.pop(-1)
    #
    #     if i == len(temp3[-1][2]):
    #         break
    #     else:
    #         if d_value[i - 1][0] == 0:
    #             continue
    #         ai.append(d_value[i - 1][0])
    #         if d_value[i - 1][0] in catId:
    #             cid.append(d_value[i - 1][0])
    # timeE = time.time()
    # print('Step2特征选取时间：', timeE - timeS, len(ai))
    # print('MAE：', tempFR[iFR])

    # 剩余时间特征选取全遍历
    # aai = []
    # aai.append(ai[0])
    # cid = []
    # cid.append(0)
    # global TR,minIn
    # TR = []
    # minIn = []
    # global tree
    # tree = tp.Tree('0')
    # timeS = time.time()
    # fnTree(modelR,aai,1,X_train,y_train,X_test,y_test,catId,cid)#加入特征类别
    # # fnTree2(modelR, aai, 1, X_train, y_train, X_test, y_test)
    # minVI = np.argmin(np.array(TR)[:, 0])
    # timeE = time.time()
    # print('特征选取时间：',timeE-timeS)
    # print('剩余时间：', TR[minVI])
    # # 画局部树图
    # minV = min(np.array(TR)[:, 0])
    # maxV = max(np.array(TR)[:, 0])
    # tree.show(20,minV,(maxV-minV)/19)
    # return TR[minVI]

def fnTree(modelR,aai,n,X_train,y_train,X_test,y_test,catId,cid):
    modelR.fit(X_train[:, aai], y_train[:, 2], feature_name=[str(aai[i]) for i in range(len(aai))],
               categorical_feature=[str(cid[i]) for i in range(len(cid))])
    y_pred = modelR.predict(X_test[:, aai])
    MAE = mean_absolute_error(y_test[:, 2], y_pred)
    # print(MAE, aai)
    TR.append([MAE, [ak[i] for i in aai], aai.copy()])
    if n == 1:# or MAE < minIn[-1][1]:
        minIn.append([len(TR) - 1, MAE, [ak[i] for i in aai], aai.copy()])
        tree.root.data = '0'
        tree.root.tag = '0'
        tree.root.value = '0 : '+str(round(MAE,3))
    elif MAE <= minIn[-1][1]:
        minIn.append([len(TR) - 1, MAE, [ak[i] for i in aai], aai.copy()])
        p = tree.root
        line = []
        line.append(aai[0])
        for i in aai[1:]:
            line.append(i)
            q = tree.searchOne(p, str(i))
            if q is None:
                MAE = TR[list(map(lambda x:x[2] ,TR)).index(line)][0]
                q = tp.Node(data=str(i),tag=str(line),value=str(i)+' : '+str('%.3f'%MAE))#round(MAE,3)
                tree.insert(p, q)
            p = q
    if aai[-1] == ai[-1]:
        return tree
    else:
        for i in range(n, len(ai)):
            if aai[-1] >= ai[i]:
                continue
            if ai[i] in aai:
                break
            aai.append(ai[i])
            if ai[i] in catId:
                cid.append(ai[i])
            fnTree(modelR, aai, n+1, X_train, y_train, X_test, y_test,catId,cid)
            if aai[-1] in catId:
                cid.pop(-1)
            aai.pop(-1)

def fnTree2(modelR,aai,n,X_train,y_train,X_test,y_test):
    modelR.fit(X_train[:, aai], y_train[:, 2])
    y_pred = modelR.predict(X_test[:, aai])
    MAE = mean_absolute_error(y_test[:, 2], y_pred)
    # print(MAE, aai)
    TR.append([MAE, [ak[i] for i in aai], aai.copy()])
    if n == 1:# or MAE < minIn[-1][1]:
        minIn.append([len(TR) - 1, MAE, [ak[i] for i in aai], aai.copy()])
        tree.root.data = '0'
        tree.root.tag = '0'
        tree.root.value = '0 : '+str(round(MAE,3))
    elif MAE <= minIn[-1][1]:
        minIn.append([len(TR) - 1, MAE, [ak[i] for i in aai], aai.copy()])
        p = tree.root
        line = []
        line.append(aai[0])
        for i in aai[1:]:
            line.append(i)
            q = tree.searchOne(p, str(i))
            if q is None:
                MAE = TR[list(map(lambda x:x[2] ,TR)).index(line)][0]
                q = tp.Node(data=str(i),tag=str(line),value=str(i)+' : '+str('%.3f'%MAE))#round(MAE,3)
                tree.insert(p, q)
            p = q
    if aai[-1] == ai[-1]:
        return tree
    else:
        for i in range(n, len(ai)):
            if aai[-1] >= ai[i]:
                continue
            if ai[i] in aai:
                break
            aai.append(ai[i])
            fnTree2(modelR, aai, n+1, X_train, y_train, X_test, y_test)
            aai.pop(-1)

#特征选取参考树
def showLocalTree(Train, Test, header, ai, cid, task):
    attribNum = len(header) - 3
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    X_train = dataTra[:, 0:attribNum]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = dataTes[:, 0:attribNum]
    y_test = dataTes[:, attribNum:attribNum + 3]

    # 调参
    # lg = lgb.LGBMClassifier(silent=False)
    # param_dist = {"max_depth": [5, 7], "learning_rate": [0.01], "num_leaves": [150, 200, 250], "n_estimators": [100]}
    # grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv=3, scoring="roc_auc", verbose=5)
    # grid_search.fit(X_train, y_train[:, 0])
    # gb = grid_search.best_estimator_
    # y_pre = grid_search.predict(X_test)
    # predictions = [round(value) for value in y_pre]
    # accuracy = accuracy_score(y_test[:, 0], predictions)

    # modelR = xgb.XGBRegressor()
    if task == 0:
        modelR = lgb.LGBMClassifier(learning_rate=0.01, max_depth=5, num_leaves=150, n_estimators=100)
    else:
        modelR = lgb.LGBMRegressor()
    aai = []
    aaiMAE = []
    cci = []
    aai.append(ai[0])
    cci.append(ai[0])
    ai.remove(ai[0])
    tree = tp.Tree('0')
    modelR.fit(X_train[:, aai], y_train[:, task], feature_name=[str(i) for i in aai],
               categorical_feature=[str(i) for i in aai if i in cid])
    y_pre = modelR.predict(X_test[:, aai])
    if task == 0:
        predictions = [round(value) for value in y_pre]
        MAE = accuracy_score(y_test[:, task], predictions)
    else:
        MAE = mean_absolute_error(y_test[:, task], y_pre)
    aaiMAE.append(MAE)
    tree.root.data = '0'
    tree.root.tag = '0'
    tree.root.value = '0 : ' + str(round(MAE, 3))
    p = tree.root
    minMAE = MAE
    maxMAE = MAE
    while len(ai) != 0:
        for line in ai:
            aai.append(line)
            if line in cid:
                cci.append(line)
            modelR.fit(X_train[:, aai], y_train[:, task], feature_name=[str(i) for i in aai],
               categorical_feature=[str(i) for i in aai if i in cid])
            y_pre = modelR.predict(X_test[:, aai])
            if task == 0:
                predictions = [round(value) for value in y_pre]
                MAE = accuracy_score(y_test[:, task], predictions)
            else:
                MAE = mean_absolute_error(y_test[:, task], y_pre)
            q = tp.Node(data=str(line), tag=str(aai), value=str(line) + ' : ' + str('%.3f'%MAE))
            tree.insert(p, q)
            aai.remove(line)
            if line in cci:
                cci.remove(line)
            if line == ai[0] or (MAE < MAEO and task !=0) or (MAE > MAEO and task ==0):
                MAEO = MAE
                t = q
                linet = line
            if MAE < minMAE:
                minMAE = MAE
            elif MAE > maxMAE:
                maxMAE = MAE
        p = t
        aai.append(linet)
        aaiMAE.append(MAEO)
        if linet in cid:
            cci.append(linet)
        ai.remove(linet)
    # 画局部树图
    # tree.show(20, round(minMAE, 3), (maxMAE - minMAE) / 19)
    return aai, aaiMAE

#特征全遍历树
def FTTree(Train, Test, header, catId):
    cid = catId.copy()
    attribNum = len(header) - 3
    hd = {header[i]: i for i in range(attribNum)}
    list_to_float1 = []
    list_to_float2 = []
    for line in Train:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))  # lambda x: float(x),
            list_to_float1.append(each_line)
    for line in Test:
        for line1 in line:
            each_line = list(map(lambda x: x, line1))
            list_to_float2.append(each_line)
    dataTra = np.array(list_to_float1)
    dataTes = np.array(list_to_float2)
    global ai,ak
    ai = [i for i in hd.values()]
    ak = [i for i in hd]
    X_train = dataTra[:, ai]
    y_train = dataTra[:, attribNum:attribNum + 3]
    X_test = dataTes[:, ai]
    y_test = dataTes[:, attribNum:attribNum + 3]

    modelR = lgb.LGBMRegressor()

    # 剩余时间特征选取全遍历
    aai = []
    aai.append(ai[0])
    cid = []
    cid.append(0)
    global TR,minIn
    TR = []
    minIn = []
    global tree
    tree = tp.Tree('0')
    timeS = time.time()
    fnAllTree(modelR,aai,1,X_train,y_train,X_test,y_test,catId,cid)
    # fnTree(modelR,aai,1,X_train,y_train,X_test,y_test,catId,cid)#加入特征类别
    # fnTree2(modelR, aai, 1, X_train, y_train, X_test, y_test)
    minVI = np.argmin(np.array(TR)[:, 0])
    timeE = time.time()
    print('特征选取时间：', timeE-timeS)
    print('剩余时间：', TR[minVI])
    # 画局部树图
    minV = min(np.array(TR)[:, 0])
    maxV = max(np.array(TR)[:, 0])
    tree.show(20,minV,(maxV-minV)/19)
    # if len(minIn)>1:
    #     myTree = {'0'+str(minIn[0][1]):plotLocalTree(1)}
    #     tp.createPlot(myTree)
    return TR[minVI]

def fnAllTree(modelR,aai,n,X_train,y_train,X_test,y_test,catId,cid):
    modelR.fit(X_train[:, aai], y_train[:, 2], feature_name=[str(aai[i]) for i in range(len(aai))],
               categorical_feature=[str(cid[i]) for i in range(len(cid))])
    y_pred = modelR.predict(X_test[:, aai])
    MAE = mean_absolute_error(y_test[:, 2], y_pred)
    TR.append([MAE, [ak[i] for i in aai], aai.copy()])
    if n == 1:
        tree.root.data = '0'
        tree.root.tag = '0'
        tree.root.value = '0 : '+str(round(MAE,3))
    else:
        p = tree.root
        line = []
        line.append(aai[0])
        for i in aai[1:]:
            line.append(i)
            q = tree.searchOne(p, str(i))
            if q is None:
                MAE = TR[list(map(lambda x:x[2], TR)).index(line)][0]
                q = tp.Node(data=str(i), tag=str(line), value=str(i)+' : '+str('%.3f'%MAE))#round(MAE,3)
                tree.insert(p, q)
            p = q
    if aai[-1] == ai[-1]:
        return tree
    else:
        for i in range(n, len(ai)):
            if aai[-1] >= ai[i]:
                continue
            if ai[i] in aai:
                break
            aai.append(ai[i])
            if ai[i] in catId:
                cid.append(ai[i])
            fnAllTree(modelR, aai, n+1, X_train, y_train, X_test, y_test,catId,cid)
            if aai[-1] in catId:
                cid.pop(-1)
            aai.pop(-1)
