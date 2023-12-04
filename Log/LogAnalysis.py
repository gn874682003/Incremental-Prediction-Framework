import math
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

def JS_divergence(p, q, bins):#JS散度：数据分布的相似度距离
    max0 = max(max(p), max(q))
    min0 = min(min(p), min(q))
    bins = np.linspace(min0-1e-4, max0-1e-4, num=bins)
    p = pd.cut(p, bins).value_counts()/len(p)
    q = pd.cut(q, bins).value_counts() / len(q)
    M = (p+q)/2
    return 0.5*scipy.stats.entropy(p,M,base=2)+0.5*scipy.stats.entropy(q,M,base=2)

def LogRecord(Log, variant = {}, activites = {}):#变体,活动
    for trace in Log.Train:
        activity = ''
        for act in trace:
            activity = activity + str(act[0]) + ' '
            if act[0] not in activites.keys():
                activites[act[0]] = []
                for fea, state in zip(act[1:-3], Log.State[1:]):
                    if state<3:
                        activites[act[0]].append([fea])
                    else:
                        activites[act[0]].append([fea, fea])
            else:
                feaNum = 0
                for fea, state in zip(act[1:-3], Log.State[1:]):
                    if state < 3:
                        if fea not in activites[act[0]][feaNum]:
                            activites[act[0]][feaNum].append(fea)
                    else:
                        if fea > activites[act[0]][feaNum][1]:
                            activites[act[0]][feaNum][1] = fea
                        if fea < activites[act[0]][feaNum][0]:
                            activites[act[0]][feaNum][0] = fea
                    feaNum += 1
        if activity not in variant.keys():
            variant[activity] = [trace[0][-1], trace[0][-1]]
        else:
            if trace[0][-1] > variant[activity][1]:
                variant[activity][1] = trace[0][-1]
            if trace[0][-1] < variant[activity][0]:
                variant[activity][0] = trace[0][-1]
    return variant, activites

def GeneralIndicator(DR, Data):
    traces = len(Data) # 轨迹数
    activitis = len(DR.ConvertReflact[0]) # 活动数
    max_case_length = 0
    variants = 0 # 变体数
    Variant = []
    Event = {} # 事件分布
    EventAvg = 0
    traceDuration = {} # 轨迹执行时间分布
    ActivityDuration = {} # 活动执行时间分布
    ActivityRemaintime = {} # 活动剩余时间分布
    onlyVariantRate = 0  # 唯一变体比
    onlyEventRate = 0  # 唯一事件比（每条轨迹的平均数）
    EventLenCV = 0  # 轨迹长度变异系数
    traceDurAvg = 0
    traceDurCV = 0 # 轨迹执行时间变异系数
    traceDurationCV = {}  # 轨迹执行时间变异系数（区分轨迹长度）
    ActivityDurationCV = {}  # 活动执行时间变异系数
    ActivityRemaintimeCV = {}  # 活动剩余时间变异系数
    for line in Data:
        tempA = []
        tempTD = line[0][-1]
        tempOE = 0
        if max_case_length < len(line):
            max_case_length = len(line)
        for line2 in line:
            if line2[0] not in tempA:
                tempOE += 1
            tempA.append(line2[0])
            if line2[0] in ActivityDuration.keys():
                ActivityDuration[line2[0]].append(line2[-2])
                ActivityRemaintime[line2[0]].append(line2[-1])
            else:
                ActivityDuration[line2[0]] = [line2[-2]]
                ActivityRemaintime[line2[0]] = [line2[-1]]
        onlyEventRate += tempOE/len(tempA)
        EventAvg += len(tempA)
        traceDurAvg += tempTD
        if tempA not in Variant:
            Variant.append(tempA)
            variants += 1
        if len(tempA) in Event.keys():
            Event[len(tempA)] += 1
            traceDuration[len(tempA)].append(tempTD)
        else:
            Event[len(tempA)] = 1
            traceDuration[len(tempA)] = [tempTD]
    # 轨迹包含事件数分布柱状图
    # Event = dict(sorted(Event.items(), key=lambda x: x[0]))
    # plt.rcParams["font.sans-serif"] = ['SimHei']
    # plt.rcParams["axes.unicode_minus"] = False
    # for eve in Event:
    #     plt.bar(str(eve), Event[eve])
    #     plt.text(str(eve), Event[eve], Event[eve], ha='center')
    # plt.title("轨迹包含事件数分布")
    # plt.xlabel("事件数（每条轨迹）")
    # plt.ylabel("轨迹数")
    # plt.show()
    # # 轨迹执行时间箱线图
    # traceDuration = dict(sorted(traceDuration.items(), key=lambda x: x[0]))
    # plt.grid(True)
    # plt.boxplot(traceDuration.values(), labels=traceDuration.keys(), sym="r+", showmeans=True)  # 绘制箱线图
    # plt.title("轨迹执行时间分布")
    # plt.xlabel("事件数（每条轨迹）")
    # plt.ylabel("轨迹执行时间（天）")
    # plt.show()
    # # 活动执行时间+等待时间箱线图
    # ActivityDuration = dict(sorted(ActivityDuration.items(), key=lambda x: x[0]))
    # plt.grid(True)
    # plt.boxplot(ActivityDuration.values(), labels=ActivityDuration.keys(), sym="r+", showmeans=True)  # 绘制箱线图
    # plt.title("活动执行时间分布")
    # plt.xlabel("活动索引")
    # plt.ylabel("活动执行时间（天）")
    # plt.show()
    # # 活动剩余时间箱线图
    # ActivityRemaintime = dict(sorted(ActivityRemaintime.items(), key=lambda x: x[0]))
    # plt.grid(True)
    # plt.boxplot(ActivityRemaintime.values(), labels=ActivityRemaintime.keys(), sym="r+", showmeans=True)  # 绘制箱线图
    # plt.title("活动剩余时间分布")
    # plt.xlabel("活动索引")
    # plt.ylabel("活动剩余时间（天）")
    # plt.show()

    onlyVariantRate = variants/traces
    onlyEventRate /= traces
    EventAvg /= traces
    traceDurAvg /= traces
    for eve in Event:
        EventLenCV += math.pow(eve - EventAvg, 2) * Event[eve]
    EventLenCV = math.sqrt(EventLenCV / (traces - 1)) / EventAvg
    for line in Data:
        traceDurCV += math.pow(line[0][-1] - traceDurAvg, 2)
    traceDurCV = math.sqrt(traceDurCV / (traces - 1)) / traceDurAvg
    for line in traceDuration:
        if len(traceDuration[line]) > 1:
            avg = sum(traceDuration[line]) / len(traceDuration[line])
            if avg == 0:
                traceDurationCV[line] = 0
                continue
            mae = 0
            for i in traceDuration[line]:
                mae += math.pow(i - avg, 2)
            traceDurationCV[line] = math.sqrt(mae / (len(traceDuration[line]) - 1)) / avg
        else:
            traceDurationCV[line] = 0
    # 轨迹包含事件数的变异系数图
    # traceDurationCV = dict(sorted(traceDurationCV.items(), key=lambda x: x[0]))
    # for i in traceDurationCV:
    #     plt.bar(str(i), round(traceDurationCV[i], 2))
    #     plt.text(str(i), round(traceDurationCV[i], 2), round(traceDurationCV[i], 2), ha='center')
    # plt.title("轨迹包含事件数的变异系数")
    # plt.xlabel("事件数（每条轨迹）")
    # plt.ylabel("变异系数")
    # plt.show()
    # # 活动数量分布图
    # ActivityDuration = dict(sorted(ActivityDuration.items(), key=lambda x: x[0]))
    # for i in ActivityDuration:
    #     plt.bar(str(i), len(ActivityDuration[i]))
    #     plt.text(str(i), len(ActivityDuration[i]), len(ActivityDuration[i]), ha='center')
    # plt.title("活动数量分布")
    # plt.xlabel("活动索引")
    # plt.ylabel("活动数")
    # plt.show()
    for line in ActivityDuration:
        if len(ActivityDuration[line]) > 1:
            avg = sum(ActivityDuration[line]) / len(ActivityDuration[line])
            if avg == 0:
                ActivityDurationCV[line] = 0
                continue
            mae = 0
            for i in ActivityDuration[line]:
                mae += math.pow(i - avg, 2)
            ActivityDurationCV[line] = math.sqrt(mae / (len(ActivityDuration[line]) - 1)) / avg
        else:
            ActivityDurationCV[line] = 0
    # 活动执行时间变异系数图
    # ActivityDurationCV = dict(sorted(ActivityDurationCV.items(), key=lambda x: x[0]))
    # for i in ActivityDurationCV:
    #     plt.bar(str(i), round(ActivityDurationCV[i], 2))
    #     plt.text(str(i), round(ActivityDurationCV[i], 2), round(ActivityDurationCV[i], 2), ha='center')
    # plt.title("活动执行时间变异系数")
    # plt.xlabel("活动索引")
    # plt.ylabel("变异系数")
    # plt.show()
    for line in ActivityRemaintime:
        if len(ActivityRemaintime[line]) > 1:
            avg = sum(ActivityRemaintime[line]) / len(ActivityRemaintime[line])
            if avg == 0:
                ActivityRemaintimeCV[line] = 0
                continue
            mae = 0
            for i in ActivityRemaintime[line]:
                mae += math.pow(i - avg, 2)
            ActivityRemaintimeCV[line] = math.sqrt(mae / (len(ActivityRemaintime[line]) - 1)) / avg
        else:
            ActivityRemaintimeCV[line] = 0
    # 活动剩余时间变异系数图
    # ActivityRemaintimeCV = dict(sorted(ActivityRemaintimeCV.items(), key=lambda x: x[0]))
    # for i in ActivityRemaintimeCV:
    #     plt.bar(str(i), round(ActivityRemaintimeCV[i], 2))
    #     plt.text(str(i), round(ActivityRemaintimeCV[i], 2), round(ActivityRemaintimeCV[i], 2), ha='center')
    # plt.title("活动剩余时间变异系数")
    # plt.xlabel("活动索引")
    # plt.ylabel("变异系数")
    # plt.show()
    # print('轨迹数：', traces)
    # print('活动数：', activitis)
    # print('变体数：', variants)
    # print('唯一变体比：', onlyVariantRate)
    # print('唯一事件比（每条轨迹的平均数）：', onlyEventRate)
    # print('轨迹长度变异系数：', EventLenCV)
    # print('轨迹执行时间变异系数：', traceDurCV)
    # print('轨迹执行时间变异系数（区分轨迹长度）：', traceDurationCV)
    # print('活动执行时间变异系数：', ActivityDurationCV)
    # print('活动剩余时间变异系数：', ActivityRemaintimeCV)
    return max_case_length

def PeriodicAnalysis0(Data, bins, Time):
    dt = 0  # 0 以轨迹开始时间划分，-1 以轨迹结束时间划分
    if Time == 'Month':
        index = -8
    elif Time == 'Day':
        index = -7
    elif Time == 'Week':
        index = -6
    elif Time == 'Hour':
        index = -5
    elif Time == 'Year':
        index = -4
    Periodic = {}
    for line in Data:
        if line[dt][index] in Periodic.keys():
            Periodic[line[dt][index]].append(line[dt][-1])
        else:
            Periodic[line[dt][index]] = [line[dt][-1]]
    Periodic = dict(sorted(Periodic.items(), key=lambda x: x[0]))
    p = []
    q = []
    s = []
    JS = 0
    Buckets = []
    temp = []
    count = 0
    for i in Periodic:
        x = [str(i) for j in range(len(Periodic[i]))]
        plt.scatter(x, Periodic[i], marker='.', color='k', s=10)
        plt.scatter(str(i), sum(Periodic[i]) / len(Periodic[i]), marker='_', color='r', s=50)
        if p == []:
            p = Periodic[i]
            s = Periodic[i]
        elif q == []:
            q = Periodic[i]
            JS = JS_divergence(p, q, bins)
        else:
            p = q
            q = Periodic[i]
            JS = JS_divergence(p, q, bins)
        # print(i, JS)
        if JS > 0.1 and count > 100:
            Buckets.append(temp)
            temp = []
            count = 0
        if temp == [] or i - 1 in Periodic.keys() or Buckets == []:
            temp.append(i)
            if JS > 0.1:
                count = len(Periodic[i])
            else:
                count += len(Periodic[i])
    Buckets.append(temp)
    JS = JS_divergence(q, s, bins)
    if (JS <= 0.1 or count < 100) and len(Buckets) > 1:
        for i in Buckets[-1]:
            Buckets[0].append(i)
        Buckets.pop(-1)
    print(Buckets)
    plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    plt.title(Time+"周期变化")
    plt.xlabel(Time)
    plt.ylabel("执行时间")
    plt.show()
    return Buckets

def PeriodicAnalysis(Data, bins, Time):
    if Time == 'Month':
        index = -8
    elif Time == 'Day':
        index = -7
    elif Time == 'Week':
        index = -6
    elif Time == 'Hour':
        index = -5
    elif Time == 'Year':
        index = -4
    Periodic = {}
    for line in Data:
        if line[0][index] in Periodic.keys():
            Periodic[line[0][index]].append(line[0][-1])
        else:
            Periodic[line[0][index]] = [line[0][-1]]
    Periodic = dict(sorted(Periodic.items(), key=lambda x: x[0]))
    p = []
    q = []
    s = []
    JS = 0
    Buckets = []
    temp = []
    count = 0
    for i in Periodic:
        x = [str(i) for j in range(len(Periodic[i]))]
        plt.scatter(x, Periodic[i], marker='.', color='k', s=10)
        plt.scatter(str(i), sum(Periodic[i])/len(Periodic[i]), marker='_', color='r', s=50)
        if p == []:
            p = Periodic[i]
            s = Periodic[i]
        elif q == []:
            q = Periodic[i]
            JS = JS_divergence(p, q, bins)
        else:
            p = q
            q = Periodic[i]
            JS = JS_divergence(p, q, bins)
        # print(i, JS)
        if JS > 0.1 and count > 100:
            Buckets.append(temp)
            temp = []
            count = 0
        if temp == [] or i-1 in Periodic.keys() or Buckets == []:
            temp.append(i)
            if JS > 0.1:
                count = len(Periodic[i])
            else:
                count += len(Periodic[i])
        else:
            for tt in temp:
                Buckets[-1].append(tt)
            temp = [i]
            count = len(Periodic[i])
    Buckets.append(temp)
    JS = JS_divergence(q, s, bins)
    if JS <= 0.1 and len(Buckets) > 2:
        for i in Buckets[-1]:
            Buckets[0].append(i)
        Buckets.pop(-1)
    print(Buckets)
    plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    plt.title(Time+"周期变化")
    plt.xlabel(Time)
    plt.ylabel("执行时间")
    plt.show()
    return Buckets

def PeriodicMerge(Data, bins, Time):
    dt = 0  # 0 以轨迹开始时间划分，-1 以轨迹结束时间划分
    traces = 100
    minJS = 0.03
    if Time == 'Month':
        index = -8
        # Buckets = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]]
        maxG = 2
    elif Time == 'Day':
        index = -7
        # Buckets = [[1], [2], [3], [4], [5], [6], [7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22],[23],[24],[25],[26],[27],[28],[29],[30],[31]]
        maxG = 2
    elif Time == 'Week':
        index = -6
        # Buckets = [[0],[1], [2], [3], [4], [5], [6]]
        maxG = 2
    elif Time == 'Hour':
        index = -5
    elif Time == 'Year':
        index = -4

    Periodic = {}
    for line in Data:
        if line[dt][index] in Periodic.keys():
            Periodic[line[dt][index]].append(line[dt][-1])
        else:
            Periodic[line[dt][index]] = [line[dt][-1]]
    Periodic = dict(sorted(Periodic.items(), key=lambda x: x[0]))
    # for i in Buckets:
    #     if i[0] not in Periodic.keys():
    #         Buckets.remove(i)
    Buckets = []
    for i in Periodic.keys():
        Buckets.append([i])
    # 只算一遍JS,然后循环合并
    temp = []
    p = []
    q = []
    s = []
    JS = 0
    count = []
    JSs = []
    for peri, i in zip(Buckets, range(len(Buckets))):
        periD = []
        for j in peri:
            periD.extend(Periodic[j])
        if p == []:
            p = periD
            s = periD
        elif q == []:
            q = periD
            JS = JS_divergence(p, q, bins)
            count.append(len(p))
        else:
            p = q
            q = periD
            JS = JS_divergence(p, q, bins)
            count.append(len(p))
        JSs.append(JS)
    JS = JS_divergence(q, s, bins)
    count.append(len(q))
    JSs[0] = JS
    # JS<0.1合并
    i = 0
    while i < len(Buckets) and len(Buckets) > 1:
        if JSs[i] < 0.1:
            if i == 0:
                Buckets[-1].extend(Buckets[0])
                Buckets.remove(Buckets[0])
                count[-1] += count[0]
                count.pop(0)
                JSs.pop(0)
            else:
                Buckets[i-1].extend(Buckets[i])
                Buckets.remove(Buckets[i])
                count[i-1] += count[i]
                count.pop(i)
                JSs.pop(i)
        else:
            i += 1
    # count<100合并
    i = 0
    while i < len(Buckets) and len(Buckets) > 1:
        if count[i] < 100:
            if i < len(Buckets)-1 and JSs[i] > JSs[i + 1]:
                Buckets[i].extend(Buckets[i + 1])
                Buckets.remove(Buckets[i + 1])
                count[i] += count[i + 1]
                count.pop(i + 1)
                JSs.pop(i + 1)
            elif i > 0 and i < len(Buckets)-1 and JSs[i] <= JSs[i+1]:
                Buckets[i - 1].extend(Buckets[i])
                Buckets.remove(Buckets[i])
                count[i - 1] += count[i]
                count.pop(i)
                JSs.pop(i)
            else:
                Buckets[-1].extend(Buckets[0])
                Buckets.remove(Buckets[0])
                count[-1] += count[0]
                count.pop(0)
                JSs.pop(0)
            # elif i == 0 and JSs[0] <= JSs[1]:
            #     Buckets[-1].extend(Buckets[0])
            #     Buckets.remove(Buckets[0])
            #     count[-1] += count[0]
            #     count.pop(0)
            #     JSs.pop(0)
            # elif i == len(Buckets)-1 and JSs[-1] > JSs[0]:
            #     Buckets[-1].extend(Buckets[0])
            #     Buckets.remove(Buckets[0])
            #     count[-1] += count[0]
            #     count.pop(0)
            #     JSs.pop(0)
        else:
            i += 1
    # 大于最大分组合并
    i = 0
    while len(Buckets) > maxG:
        i = JSs.index(min(JSs))  # count.index(min(count))
        if i == 0:
            Buckets[-1].extend(Buckets[0])
            Buckets.remove(Buckets[0])
            count[-1] += count[0]
            count.pop(0)
            JSs.pop(0)
        else:
            Buckets[i - 1].extend(Buckets[i])
            Buckets.remove(Buckets[i])
            count[i - 1] += count[i]
            count.pop(i)
            JSs.pop(i)
    # print(Buckets)
    # print(count)
    # print(JSs)
    return Buckets

def PeriodicMerge0(Data, bins, Time):
    dt = 0  # 0 以轨迹开始时间划分，-1 以轨迹结束时间划分
    traces = 100
    minJS = 0.03
    if Time == 'Month':
        index = -8
        Buckets = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]]
    elif Time == 'Day':
        index = -7
        Buckets = [[1], [2], [3], [4], [5], [6], [7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22],[23],[24],[25],[26],[27],[28],[29],[30],[31]]
    elif Time == 'Week':
        index = -6
        Buckets = [[0],[1], [2], [3], [4], [5], [6]]
    elif Time == 'Hour':
        index = -5
    elif Time == 'Year':
        index = -4
    Periodic = {}
    for line in Data:
        if line[dt][index] in Periodic.keys():
            Periodic[line[dt][index]].append(line[dt][-1])
        else:
            Periodic[line[dt][index]] = [line[dt][-1]]
    Periodic = dict(sorted(Periodic.items(), key=lambda x: x[0]))
    for i in Buckets:
        if i[0] not in Periodic.keys():
            Buckets.remove(i)
    # Buckets = []
    temp = []
    while len(Buckets)>1:
        p = []
        q = []
        s = []
        JS = 0
        count = []
        JSs = []
        for peri, i in zip(Buckets, range(len(Buckets))):
            periD = []
            for j in peri:
                periD.extend(Periodic[j])
            if p == []:
                p = periD
                s = periD
            elif q == []:
                q = periD
                JS = JS_divergence(p, q, bins)
                count.append(len(p))
            else:
                p = q
                q = periD
                JS = JS_divergence(p, q, bins)
                count.append(len(p))
            JSs.append(JS)
            # print(peri, JS)
            if JS < minJS and q != []:
                Buckets[i-1].extend(peri)
                Buckets.pop(i)
                p = Buckets[i-1]
        if len(Buckets) > 1:
            JS = JS_divergence(q, s, bins)
            count.append(len(q))
            JSs[0] = JS
            if JS < minJS:
                Buckets[0].extend(Buckets[-1])
                Buckets.pop(-1)
            # print(JSs)
            if temp == Buckets:
                break
            else:
                temp = Buckets.copy()
    # print(Buckets)
    while len(Buckets)>1:
        flag = 0
        for i in range(len(Buckets)):
            if count[i] < traces:
                flag = 1
                JSl = JSs[i]
                if i == len(Buckets) - 1:
                    JSn = JSs[0]
                else:
                    JSn = JSs[i+1]
                if JSl < JSn:
                    if i == 0:
                        Buckets[0].extend(Buckets[-1])
                        Buckets.remove(Buckets[-1])
                        count[0] += count[-1]
                        count.pop(-1)
                    else:
                        Buckets[i-1].extend(Buckets[i])
                        Buckets.remove(Buckets[i])
                        count[i-1] += count[i]
                        count.pop(i)
                else:
                    if i == len(Buckets) - 1:
                        Buckets[0].extend(Buckets[-1])
                        Buckets.remove(Buckets[-1])
                        count[0] += count[-1]
                        count.pop(-1)
                    else:
                        Buckets[i].extend(Buckets[i + 1])
                        Buckets.remove(Buckets[i + 1])
                        count[i] += count[i + 1]
                        count.pop(i + 1)
                break
        # print(count)
        if flag == 0:
            break
        p = []
        q = []
        s = []
        JS = 0
        count = []
        JSs = []
        for peri, i in zip(Buckets, range(len(Buckets))):
            periD = []
            for j in peri:
                periD.extend(Periodic[j])
            if p == []:
                p = periD
                s = periD
            elif q == []:
                q = periD
                JS = JS_divergence(p, q, bins)
                count.append(len(p))
            else:
                p = q
                q = periD
                JS = JS_divergence(p, q, bins)
                count.append(len(p))
            JSs.append(JS)
            # print(peri, JS)
            if JS < minJS and q != []:
                Buckets[i - 1].extend(peri)
                Buckets.pop(i)
                p = Buckets[i - 1]
        if len(Buckets) > 1:
            JS = JS_divergence(q, s, bins)
            count.append(len(q))
            JSs[0] = JS
            if JS < minJS:
                Buckets[0].extend(Buckets[-1])
                Buckets.pop(-1)
    print(Buckets)
    # print(count)
    # print(JSs)
    return Buckets