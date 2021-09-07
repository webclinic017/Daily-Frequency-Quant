# Copyright (c) 2021 Dai HBG

"""
该代码定义的SignalGenerator类将解析一个公式，然后递归地计算signal值
v1.0
默认所有操作将nan替换成0
"""

import numba as nb
import numpy as np


class SignalGenerator:
    def __init__(self, top):
        """
        :param top: 计算截面时需要使用
        """
        self.operation_dic = {}
        self.get_operation()
        self.top = top

        self.operation_dic['zscore'] = self.zscore
        self.operation_dic['csrank'] = self.csrank

        """
        截面算子，因为要调用top
        """

    def csrank(self, a):
        b = a.copy()  # 测试用，可以不用复制
        b[np.isnan(b)] = 0
        for i in range(len(a)):
            n = np.sum(b[i][self.top[i]] != 0)
            if n == 0:
                continue
            tmp = b[i][self.top[i]].copy()
            pos = tmp.argsort()
            for j in range(len(tmp)):
                tmp[pos[j]] = j
            tmp /= (len(self.top[i]) - 1)
            b[i][self.top[i]] = tmp
        return b

    def zscore(self, a):
        b = a.copy()
        b[np.isnan(b)] = 0
        for i in range(len(a)):
            if np.sum(b[i][self.top[i]] != 0) <= 1:
                continue
            b[i][self.top[i]] -= np.mean(b[i][self.top[i]])
            b[i][self.top[i]] /= np.std(b[i][self.top[i]])
            b[i][(self.top[i]) & (b[i] > 3)] = 3
            b[i][(self.top[i]) & (b[i] < -3)] = -3
        return b

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
            return a

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
            for i in range(len(a)):
                if i < num - 1:
                    continue
                for j in range(a.shape[1]):
                    tmp = a[i - num + 1:i + 1, j].sort()
                    a[i, j] = np.where(tmp == a[i, j])[0]
                    a[i, j] /= (num - 1)
            return a

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
