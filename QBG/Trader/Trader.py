"""
Trader是一个定制策略的回测类，为了统一框架，一定是传入一个Data类，需要定制的算法是给出收益序列
"""

import numpy as np


class Trader:
    def __init__(self, data, trade_func=None):
        self.data = data
        if trade_func is None:
            self.trade_func = self.default_trade_func
        else:
            self.trade_func = trade_func

        self.pnl = None
        self.cumulated_pnl = None

    def default_trade_func(self):  # 默认交易函数，为盘中有涨停就等权买入，第二天收盘卖出
        """
        :return:
        """
        high_close = np.zeros(self.data.ret.shape)
        high_close[:-1] = self.data.data_dic['close'][1:] / self.data.data_dic['high'][:-1] - 1  # 今最高到明天收盘收益
        high_close[np.isinf(high_close)] = 0
        high_close[high_close > 0.2] = 0.2
        close_high = np.zeros(self.data.ret.shape)
        close_high[:-1] = self.data.data_dic['high'][1:] / self.data.data_dic['close'][:-1] - 1  # 今收盘到明天最高价的收益
        close_low = np.zeros(self.data.ret.shape)
        close_low[:-1] = self.data.data_dic['low'][1:] / self.data.data_dic['close'][:-1] - 1
        p = []
        all_p = []
        c_p = [0]
        c = []
        for i in range(len(high_close) - 1):
            p.append(np.mean(high_close[i + 1, AF.data.top[i] & (close_high[i] >= 0.099) & (close_low[i] < 0.099)]))
            all_p += list(high_close[i + 1, AF.data.top[i] & (close_high[i] >= 0.099) & (close_low[i] < 0.099)])
            c_p.append(c_p[-1] + p[-1])
            c.append(np.sum((close_high[i] >= 0.099) & (close_low[i] < 0.099)))
        p = np.array(p)
        # c = np.array(c)
        self.pnl = p
        self.cumulated_pnl = c_p
