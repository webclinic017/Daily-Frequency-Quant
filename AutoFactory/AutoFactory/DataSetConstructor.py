# Copyright (c) 2021 Dai HBG

import numpy as np
"""
DataSetConstructor是一个根据起止日期生成按时间顺序排列的X和Y
"""


class DataSetConstructor:  # 构造给定起始日期的模型训练数据集
    def __init__(self, signals_dic, ret, top, date_position_dic, shift=1):
        """
        :param signals_dic: 信号字典，值都是和ret
        :param ret: 待预测收益率
        :param top: 选择矩阵，表示当前选出哪些股票
        :param date_position_dic:
        :param shift: 待预测的信号的漂移量，1表示今天的信号预测明天的收益率
        """
        self.signals_dic = signals_dic
        self.ret = ret
        self.top = top
        self.data_position_dic = date_position_dic

    def construct(self, start_date, end_date):
        """
        :param start_date: 数据开始日期
        :param end_date: 数据结束日期
        :return: 返回一个np.array形式的X，Y
        """
        start = self.data_position_dic[start_date]
        end = self.data_position_dic[end_date]
        x = []
        y = []
        for i in range(start, end+1):
            x_tmp = []
            for j in self.signals_dic.keys():
                x_tmp.append(self.signals_dic[j][self.top[i]])
            x.append(np.vstack(x_tmp).T)
            y.append(self.ret[self.top[i]])
        x = np.vstack(x)
        y = np.hstack(y)
        return x, y