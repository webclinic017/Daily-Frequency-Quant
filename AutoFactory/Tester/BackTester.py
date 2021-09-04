# Copyright (c) 2021 Dai HBG

"""
BackTester类根据传入信号以及交易逻辑进行交易
"""
import numpy as np
import datetime


class BackTester:
    def __init__(self, data, signal=None):
        """
        :param signal: 信号矩阵
        :param date: Data类
        """
        self.signal = signal
        self.data = data
        self.pnl = []  # pnl序列
        self.cumulated_pnl = []  # 累计pnl

    def long_short(self, start_date=None, end_date=None, n=0):  # 多空策略
        """
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param n: 进入股票数量，0表示使用top
        :return:
        """
        self.pnl = []
        self.cumulated_pnl = []

        if start_date is None:
            start_date = str(self.data.start_date)
        if end_date is None:
            end_date = str(self.data.end_date)
        tmp_start = start_date.split('-')
        i = 0
        while True:
            s = datetime.date(int(tmp_start[0]), int(tmp_start[1]), int(tmp_start[2])) + datetime.timedelta(days=i)
            try:
                start = self.data.date_position_dic[s]
                break
            except KeyError:
                i += 1
        i = 0
        tmp_end = end_date.split('-')
        while True:
            s = datetime.date(int(tmp_end[0]), int(tmp_end[1]), int(tmp_end[2])) + datetime.timedelta(days=i)
            try:
                end = self.data.date_position_dic[s]
                break
            except KeyError:
                i += 1

        if n != 0:  # 暂时不管
            return
        else:
            for i in range(start, end + 1):
                tmp = self.signal[i].copy()
                tmp[self.data.top[i]] -= np.mean(tmp[self.data.top[i]])
                tmp[self.data.top[i] & (tmp > 0)] /= np.sum(tmp[self.data.top[i] & (tmp > 0)])
                tmp[self.data.top[i] & (tmp < 0)] /= -np.sum(tmp[self.data.top[i] & (tmp < 0)])
                self.pnl.append(np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1, self.data.top[i]]) / 2)
                if not self.cumulated_pnl:
                    self.cumulated_pnl.append(np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1,
                                                                                           self.data.top[i]]) / 2)
                else:
                    self.cumulated_pnl.append(self.cumulated_pnl[-1] \
                                              + np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1,
                                                                                             self.data.top[i]]) / 2)

    def gen_signal(self, model, signals_dic, start_date=None, end_date=None):
        """
        :param model: 一个模型
        :param signals_dic: 使用的原始信号字典
        :param start_date: 得到信号的开始日期
        :param end_date: 得到信号的结束日期
        :return:
        """
        # 支持传入模型预测得到signal测试，为了方便必须返回一个形状完全一致的signal矩阵，只不过可以只在对应位置有值
        if start_date is None:
            start_date = str(self.data.start_date)
        if end_date is None:
            end_date = str(self.data.end_date)
        signal = np.zeros(self.data.data_dic['close'].shape)

        """
        这段代码重复了，需要包装成一个函数
        """
        i = 0
        tmp_start = start_date.split('-')
        while True:
            s = datetime.date(int(tmp_start[0]), int(tmp_start[1]), int(tmp_start[2])) + datetime.timedelta(days=i)
            try:
                start = self.data.date_position_dic[s]
                break
            except KeyError:
                i += 1
        i = 0
        tmp_end = end_date.split('-')
        while True:
            s = datetime.date(int(tmp_end[0]), int(tmp_end[1]), int(tmp_end[2])) + datetime.timedelta(days=i)
            try:
                end = self.data.date_position_dic[s]
                break
            except KeyError:
                i += 1

        for i in range(start, end + 1):
            tmp_x = []
            for j in signals_dic.keys():
                tmp = signals_dic[j][i].copy()
                tmp[np.isnan(tmp)] = 0
                tmp[self.data.top[i]] -= np.mean(tmp[self.data.top[i]])
                if np.sum(tmp[self.data.top[i]] != 0) >= 2:
                    tmp[self.data.top[i]] /= np.std(tmp[self.data.top[i]])
                tmp_x.append(tmp)
            tmp_x = np.vstack(tmp_x).T  # 用于预测
            signal[i, self.data.top[i]] = model.predict(tmp_x[self.data.top[i], :])  # 只预测需要的部分

        self.signal = signal

        return signal
