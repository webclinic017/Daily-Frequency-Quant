# Copyright (c) 2021 Dai HBG

"""
DataSetConstructor是一个根据起止日期生成按时间顺序排列的X和Y
"""

"""
更新日志：
2021-09-04
-- 更新：信号不再直接传入，而是传入一个地址，然后初始化的时候读
"""

import numpy as np
import os
import pickle


class DataSetConstructor:  # 构造给定起始日期的模型训练数据集
    def __init__(self, data, signal_path=None, shift=1):
        """
        :param data: Data类
        :param signal_path: 信号矩阵路径
        :param shift: 平移的天数，1代表着今天的信号预测明天的收益率
        """

        if signal_path is None:
            signal_path = 'F:/Documents/AutoFactoryData/Signal/{}-{}'.format(data.start_date, data.end_date)
        self.signal_path = signal_path
        self.data = data
        self.shift = shift

        print('reading signal...')
        signals_dic = {}
        lst = os.listdir(self.signal_path)
        num = 0
        for s in lst:
            with open('{}/{}'.format(self.signal_path, s), 'rb') as file:
                tmp = pickle.load(file)
                signals_dic[num] = tmp
            num += 1
        self.signals_dic = signals_dic
        print('done.')

    def construct(self, start_date=None, end_date=None):
        """
        :param start_date: 数据开始日期
        :param end_date: 数据结束日期
        :return: 返回一个np.array形式的X，Y
        """
        if start_date is None:
            start_date = self.data.start_date
        if end_date is None:
            end_date = self.data.end_date

        start = self.data.data_position_dic[start_date]
        end = self.data.data_position_dic[end_date]

        x = []
        y = []
        for i in range(start, end + 1):
            x_tmp = []
            for j in self.signals_dic.keys():
                x_tmp.append(self.signals_dic[j][self.top[i]])
            x.append(np.vstack(x_tmp).T)
            y.append(self.ret[self.top[i]])
        x = np.vstack(x)
        y = np.hstack(y)
        return x, y
