# Copyright (c) 2021 Dai HBG

"""
BackTester类根据传入信号以及交易逻辑进行交易
"""
import numpy as np
import datetime


class BackTester:
    def __init__(self, signal, data):
        """
        :param signal: 信号矩阵
        :param date: Data类
        """
        self.data = data
        self.top = top

    def long_short(self, n=0):  # 多空策略
        """
        :param n: 进入多少股票，默认top的股票全部进入
        :return: 返回
        """
        pass
