# Copyright (c) 2021 Dai HBG

"""
该代码定义的SignalGenerator类将解析一个公式，然后递归地计算signal值
v1.0
默认所有操作将nan替换成0

开发日志：
2021-09-13
-- 新增：csindneutral算子，获得行业中性的信号
-- 更新：为了方便算子计算，SignalGenerator类需要传入一个data类进行初始化
"""

import numba as nb
import numpy as np


class SignalGenerator:
    def __init__(self, data):
        """
        :param data: Data类的实例
        """
        self.operation_dic = {}
        self.get_operation()
        self.data = data

        # 单独注册需要用到额外信息的算子
        self.operation_dic['zscore'] = self.zscore
        self.operation_dic['csrank'] = self.csrank
        self.operation_dic['csindneutral'] = self.csindneutral
        self.operation_dic['csind'] = self.csind

        """
        截面算子，因为要调用top
        """

    def csrank(self, a):
        b = a.copy()  # 测试用，可以不用复制
        b[np.isnan(b)] = 0
        for i in range(len(a)):
            n = np.sum(b[i, self.data.top[i]] != 0)
            if n == 0:
                continue
            tmp = b[i, self.data.top[i]].copy()
            pos = tmp.argsort()
            for j in range(len(tmp)):
                tmp[pos[j]] = j
            tmp /= (len(tmp) - 1)
            b[i, self.data.top[i]] = tmp
        return b

    def zscore(self, a):
        b = a.copy()
        b[np.isnan(b)] = 0
        for i in range(len(a)):
            if np.sum(b[i][self.data.top[i]] != 0) <= 1:
                continue
            b[i][self.data.top[i]] -= np.mean(b[i][self.data.top[i]])
            b[i][self.data.top[i]] /= np.std(b[i][self.data.top[i]])
            b[i][(self.data.top[i]) & (b[i] > 3)] = 3
            b[i][(self.data.top[i]) & (b[i] < -3)] = -3
        return b

    def csindneutral(self, a):  # 截面中性化，暂时先使用申万二级行业，之后需要加入可选行业中性化
        s = a.copy()
        ind = self.data.industry['sws']  # 申万二级行业的位置
        for i in range(len(s)):
            ind_num_dic = {}  # 存放行业总数
            ind_sum_dic = {}  # 存放行业总值
            for j in list(set(ind[i])):
                ind_num_dic[j] = np.sum(ind[i] == j)
                ind_sum_dic[j] = np.sum(a[i, ind[i] == j])
            for key in ind_sum_dic.keys():
                ind_sum_dic[key] /= ind_num_dic[key]
            for j in range(s.shape[1]):
                s[i, j] = a[i, j] - ind_sum_dic[ind[i, j]]  # 减去行业平均，如果是没有出现过的行业，那么就是0
        return s

    def csind(self, a):  # 截面替换成所处行业的均值
        s = a.copy()
        ind = self.data.industry['sws']  # 申万二级行业的位置
        for i in range(len(s)):
            ind_num_dic = {}  # 存放行业总数
            ind_sum_dic = {}  # 存放行业总值
            for j in list(set(ind[i])):
                ind_num_dic[j] = np.sum(ind[i] == j)
                ind_sum_dic[j] = np.sum(a[i, ind[i] == j])
            for key in ind_sum_dic.keys():
                ind_sum_dic[key] /= ind_num_dic[key]
            for j in range(s.shape[1]):
                s[i, j] = ind_sum_dic[ind[i, j]]  # 减去行业平均，如果是没有出现过的行业，那么就是0
        return s

    def get_operation(self):
        def neg(a):
            return -a

        self.operation_dic['neg'] = neg

        """
        1_num型运算符
        """

        def tsdelay(a, num):
            s = np.zeros(a.shape)
            s[num:] = a[:-num].copy()
            return s

        self.operation_dic['tsdelay'] = tsdelay

        def tsdelta(a, num):
            s = np.zeros(a.shape)
            s[num:] = a[num:] - a[:-num]
            return s

        self.operation_dic['tsdelta'] = tsdelta

        @nb.jit
        def tsstd(a, num):
            s = np.zeros(a.shape)
            for i in range(len(a)):
                if i < num - 1:
                    continue
                for j in range(a.shape[1]):
                    s[i, j] = np.std(a[i - num + 1:i + 1, j])
            return s

        self.operation_dic['tsstd'] = tsstd

        @nb.jit
        def tsmean(a, num):
            s = np.zeros(a.shape)
            for i in range(len(a)):
                if i < num - 1:
                    continue
                for j in range(a.shape[1]):
                    s[i, j] = np.mean(a[i - num + 1:i + 1, j])
            return s

        self.operation_dic['tsmean'] = tsmean

        @nb.jit
        def tskurtosis(a, num):
            s = np.zeros(a.shape)
            for i in range(len(a)):
                if i < num - 1:
                    continue
                for j in range(a.shape[1]):
                    if np.std(a[i - num + 1:i + 1, j]) == 0:
                        continue
                    s[i, j] = np.mean((a[i - num + 1:i + 1,
                                       j] - np.mean(a[i - num + 1:i + 1,
                                                    j])) ** 4) / (np.std(a[i - num + 1:i + 1, j]) ** 4) - 3
            return s

        self.operation_dic['tskurtosis'] = tskurtosis

        @nb.jit
        def tsskew(a, num):
            s = np.zeros(a.shape)
            for i in range(len(a)):
                if i < num - 1:
                    continue
                for j in range(a.shape[1]):
                    s[i, j] = np.mean((a[i - num + 1:i + 1, j] - np.mean(a[i - num + 1:i + 1, j])) ** 3) / \
                              (np.std(a[i - num + 1:i + 1, j])) ** 3
            return s

        self.operation_dic['tsskew'] = tsskew

        @nb.jit
        def wdirect(a, num):  # 过去一段时间中心化之后时序加权
            s = np.zeros(a.shape)
            w = np.array([i for i in range(1, num + 1)])
            for i in range(len(a)):
                if i < num - 1:
                    continue
                for j in range(a.shape[1]):
                    if np.std(a[i - num + 1:i + 1, j]) == 0:
                        continue
                    s[i, j] = np.sum(w * (a[i - num + 1:i + 1, j] - np.mean(a[i - num + 1:i + 1, j]))) / \
                              np.std(a[i - num + 1:i + 1, j])
            return s

        self.operation_dic['wdirect'] = wdirect

        @nb.jit
        def tsrank(a, num):
            s = np.zeros(a.shape)

            for i in range(len(a)):
                if i < num - 1:
                    continue
                for j in range(a.shape[1]):
                    k = 0
                    tar = a[i, j]
                    if np.isnan(tar):
                        tar = 0
                    for c in a[i - num + 1:i + 1, j]:
                        if np.isnan(c):
                            continue
                        if c < tar:
                            k += 1
                    s[i, j] = k / (num - 1)
            return s

        self.operation_dic['tsrank'] = tsrank

        """
        2型运算符
        """

        def add(a, b):
            return a + b

        self.operation_dic['add'] = add

        def minus(a, b):
            return a - b

        self.operation_dic['minus'] = minus

        def prod(a, b):
            c = a * b
            c[np.isnan(c)] = 0
            c[np.isinf(c)] = 0
            return a * b

        self.operation_dic['prod'] = prod

        def div(a, b):
            c = a / b
            c[np.isnan(c)] = 0
            c[np.isinf(c)] = 0
            return c

        self.operation_dic['div'] = div

        """
        2_num型运算符
        """

        @nb.jit
        def tscorr(a, b, num):
            s = np.zeros(a.shape)
            for i in range(len(a)):
                if i < num - 1:
                    continue
                for j in range(a.shape[1]):
                    s[i, j] = np.corrcoef(a[i - num + 1:i + 1, j], b[i - num + 1:i + 1, j])[0, 1]
            return s

        self.operation_dic['tscorr'] = tscorr

        """
        1_num_num型运算符
        """

        @nb.jit
        def tsautocorr(a, delta, num):
            s = np.zeros(a.shape)
            for i in range(len(a)):
                if i < delta + num - 1:
                    continue
                for j in range(a.shape[1]):
                    s[i, j] = np.corrcoef(a[i - num + 1:i + 1, j], a[i - num + 1 - delta:i + 1 - delta, j])[0, 1]
            return s

        self.operation_dic['tsautocorr'] = tsautocorr

        # @nb.jit
        def condition(a, b, c):
            """
            :param a: 条件，一个布尔型矩阵
            :param b: 真的取值
            :param c: 假的取值
            :return: 信号
            """
            s = np.zeros(a.shape)
            for i in range(len(a)):
                if type(b) == int or type(b) == float:
                    s[i, a[i]] = b
                else:
                    s[i, a[i]] = b[i, a[i]]
                if type(c) == int or type(c) == float:
                    s[i, ~a[i]] = c
                else:
                    s[i, ~a[i]] = c[i, ~a[i]]
            return s

        self.operation_dic['condition'] = condition

        def lt(a, b):
            return a < b

        self.operation_dic['lt'] = lt

        def le(a, b):
            return a <= b

        self.operation_dic['le'] = le

        def gt(a, b):
            return a > b

        self.operation_dic['gt'] = gt

        def ge(a, b):
            return a >= b

        self.operation_dic['ge'] = ge
