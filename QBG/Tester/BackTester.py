# Copyright (c) 2021 Dai HBG

import numpy as np
import datetime

"""
BackTester类根据传入信号以及交易逻辑进行交易
"""

"""
开发日志
2021-09-07
-- 更新：BackTester类统计pnl序列的平均日收益，最大回撤，标准差，夏普比，最长亏损时间
2021-09-11
-- 修复：回测时剔除涨停板
"""


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
        self.market_pnl = []  # 如果纯多头，这里存市场的pnl，以比较超额收益
        self.market_cumulated_pnl = []

        self.max_dd = 0  # 统计pnl序列的最大回撤
        self.mean_pnl = 0  # 平均pnl
        self.std = 0  # 统计pnl序列的标准差
        self.sharp_ratio = 0  # 夏普比
        self.max_loss_time = 0  # 最长亏损时间

        self.log = []  # 记录每一个具体的交易日给出的股票

    def cal_stats(self):
        self.std = np.std(self.pnl)
        self.mean_pnl = np.mean(self.pnl)
        self.sharp_ratio = self.mean_pnl / self.std

        max_dd = 0
        max_pnl = 0
        max_loss_time = 0
        loss_time = 0
        for i in self.cumulated_pnl:
            if i > max_pnl:
                max_pnl = i
                loss_time = 0
            else:
                if max_pnl - i > max_dd:
                    max_dd = max_pnl - i
                loss_time += 1
                if loss_time > max_loss_time:
                    max_loss_time = loss_time
        self.max_dd = max_dd
        self.max_loss_time = max_loss_time

    def long_short(self, start_date=None, end_date=None, n=0):  # 多空策略
        """
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param n: 进入股票数量，0表示使用top
        :return:
        """
        self.pnl = []
        self.cumulated_pnl = []
        self.market_pnl = []
        self.market_cumulated_pnl = []

        if start_date is None:
            start_date = str(self.data.start_date)
        if end_date is None:
            end_date = str(self.data.end_date)

        start, end = self.data.get_real_date(start_date, end_date)

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
                    self.cumulated_pnl.append(
                        self.cumulated_pnl[-1] + np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1,
                                                                                              self.data.top[i]]) / 2)

    def long(self, start_date=None, end_date=None, n=0):  # 多头策略
        """
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param n: 进入股票数量，0表示使用top
        :return:
        """
        self.pnl = []
        self.cumulated_pnl = []
        self.market_pnl = []
        self.market_cumulated_pnl = []

        if start_date is None:
            start_date = str(self.data.start_date)
        if end_date is None:
            end_date = str(self.data.end_date)

        start, end = self.data.get_real_date(start_date, end_date)

        if n != 0:  # 暂时不管
            return
        else:
            for i in range(start, end + 1):
                tmp = self.signal[i].copy()
                tmp[self.data.top[i]] -= np.mean(tmp[self.data.top[i]])
                tmp[self.data.top[i] & (tmp > 0)] /= np.sum(tmp[self.data.top[i] & (tmp > 0)])
                tmp[self.data.top[i] & (tmp < 0)] = 0
                self.pnl.append(np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1, self.data.top[i]]) -
                                np.mean(self.data.ret[i + 1, self.data.top[i]]))
                if not self.cumulated_pnl:
                    self.cumulated_pnl.append(np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1,
                                                                                           self.data.top[i]]) -
                                              np.mean(self.data.ret[i + 1, self.data.top[i]]))
                else:
                    self.cumulated_pnl.append(
                        self.cumulated_pnl[-1] + np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1,
                                                                                              self.data.top[i]]) -
                        np.mean(self.data.ret[i + 1, self.data.top[i]]))

    def long_top_n(self, start_date=None, end_date=None, n=0):  # 做多预测得分最高的n只股票
        """
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param n: 做多多少只股票，默认按照top做多
        :return:
        """
        self.pnl = []
        self.cumulated_pnl = []
        self.market_pnl = []
        self.market_cumulated_pnl = []

        # 验证用功能：记录每一天具体的给出的股票代码和实际的收益率
        self.log = []

        if start_date is None:
            start_date = str(self.data.start_date)
        if end_date is None:
            end_date = str(self.data.end_date)

        start, end = self.data.get_real_date(start_date, end_date)

        pos = np.array([i for i in range(len(self.data.top[0]))])  # 测试用
        if n != 0:
            for i in range(start, end + 1):
                tmp = self.signal[i].copy()
                tmp[self.data.top[i]] -= np.mean(tmp[self.data.top[i]])
                tmp[self.data.top[i] & (tmp > 0)] /= np.sum(tmp[self.data.top[i] & (tmp > 0)])
                tmp[self.data.top[i] & (tmp < 0)] = 0
                a = tmp[self.data.top[i] & (tmp > 0)].argsort()[-n:]
                self.log.append((self.data.position_date_dic[i],
                                 self.data.order_code_dic[pos[self.data.top[i] & (tmp > 0)][a][0]]))
                """
                self.pnl.append(np.sum(tmp[self.data.top[i] & (tmp > 0)][a] *
                                       self.data.ret[i + 1, self.data.top[i] & (tmp > 0)][a]) /
                                np.sum(tmp[self.data.top[i] & (tmp > 0)][a]))
                """
                ret_tmp = self.data.ret[i + 1, self.data.top[i] & (tmp > 0)][a].copy()
                # ret_tmp[self.data.ret[i, self.data.top[i] & (tmp > 0)][a] >= 0.099] = 0  # 剔除涨停板
                # if np.sum(ret_tmp == 0) >= 1:
                # ret_tmp = np.zeros(ret_tmp.shape)
                self.pnl.append(np.mean(ret_tmp))
                self.market_pnl.append(np.mean(self.data.ret[i + 1, self.data.top[i]]))
                if not self.cumulated_pnl:
                    """
                    self.cumulated_pnl.append(np.sum(tmp[self.data.top[i] & (tmp > 0)][a] *
                                                     self.data.ret[i + 1, self.data.top[i] & (tmp > 0)][a]) /
                                              np.sum(tmp[self.data.top[i] & (tmp > 0)][a]))
                                              """
                    self.cumulated_pnl.append(np.mean(self.data.ret[i + 1, self.data.top[i] & (tmp > 0)][a]))
                    self.market_cumulated_pnl.append(np.mean(self.data.ret[i + 1, self.data.top[i]]))
                else:
                    """
                    self.cumulated_pnl.append(np.sum(tmp[self.data.top[i] & (tmp > 0)][a] *
                                                     self.data.ret[i + 1, self.data.top[i] & (tmp > 0)][a]) /
                                              np.sum(tmp[self.data.top[i] & (tmp > 0)][a]))
                                              """
                    self.cumulated_pnl.append(self.cumulated_pnl[-1] +
                                              np.mean(ret_tmp))
                    self.market_cumulated_pnl.append(self.market_cumulated_pnl[-1] +
                                                     np.mean(self.data.ret[i + 1, self.data.top[i]]))
        else:
            for i in range(start, end + 1):
                tmp = self.signal[i].copy()
                tmp[self.data.top[i]] -= np.mean(tmp[self.data.top[i]])
                tmp[self.data.top[i] & (tmp > 0)] /= np.sum(tmp[self.data.top[i] & (tmp > 0)])
                tmp[self.data.top[i] & (tmp < 0)] = 0
                self.pnl.append(np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1, self.data.top[i]]))
                self.market_pnl.append(np.mean(self.data.ret[i + 1, self.data.top[i]]))
                if not self.cumulated_pnl:
                    self.cumulated_pnl.append(np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1,
                                                                                           self.data.top[i]]))
                    self.market_cumulated_pnl.append(np.mean(self.data.ret[i + 1, self.data.top[i]]))
                else:
                    self.cumulated_pnl.append(
                        self.cumulated_pnl[-1] + np.sum(tmp[self.data.top[i]] * self.data.ret[i + 1,
                                                                                              self.data.top[i]]))
                    self.market_cumulated_pnl.append(self.market_cumulated_pnl[-1] +
                                                     np.mean(self.data.ret[i + 1, self.data.top[i]]))
        return self.log

    def long_stock_predict(self, date=None, n=1):  # 非回测模式，直接预测最新交易日的股票
        """
        :param date: 预测的日期，默认是最新的日期
        :param n: 需要预测多少只股票
        :return: 返回预测的股票代码以及他们的zscore分数
        """
        pos = np.array([i for i in range(len(self.data.top[0]))])
        if date is None:
            start, end = self.data.get_real_date(str(self.data.start_date), str(self.data.end_date))
        else:
            start, end = self.data.get_real_date(date, date)
        for i in range(end, end + 1):
            tmp = self.signal[i].copy()
            tmp[self.data.top[i]] -= np.mean(tmp[self.data.top[i]])
            tmp[self.data.top[i] & (tmp > 0)] /= np.sum(tmp[self.data.top[i] & (tmp > 0)])
            tmp[self.data.top[i] & (tmp < 0)] = 0
            a = tmp[self.data.top[i] & (tmp > 0)].argsort()[-n:]
            return (self.data.position_date_dic[i],
                    [self.data.order_code_dic[pos[self.data.top[i] & (tmp > 0)][a][j]] for j in range(n)],
                    tmp[self.data.top[i] & (tmp > 0)][a])

    def generate_signal(self, model, signals_dic, start_date=None, end_date=None):
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

        start, end = self.data.get_real_date(start_date, end_date)

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
