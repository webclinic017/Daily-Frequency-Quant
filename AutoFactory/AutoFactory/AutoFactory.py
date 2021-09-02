# Copyright (c) 2021 Dai HBG

"""
AutoFactory类是一个总体的集成类，通过调用其他类实现以下功能模块：
1. 测试因子，该功能通过AutoFormula类定义的方法实现
2. 自动搜寻因子，该功能通过调用AutoFormula类定义的方法实现
3. 模型训练，该功能通过调用Model类定义的各种结构化模型实现
4. 模拟交易，该功能通过调用Trader类定义的方法进行信号交易，评估效果
5. 股票生成，该功能通过调用real模式的Trader，根据日期给出股票信号，从高到低排列

结构说明：
1. AutoFactory类是核心，其初始化的时候将调用DataLoader读取数据，然后得到data_dic等保存在类属性中
2. AutoFormula文件夹下定义的都是和formula相关的类，因此所有和公式有关的方法，包括树的生成，解析，信号提取等，都应该在里面定义
3. Tester文件夹下定义的类都是和信号评价相关的类，因此所有和信号相关的方法都应该在里面定义
4. Model文件夹下定义的类都是和模型线相关的
"""

import sys

sys.path.append('../Tester')
sys.path.append('../DataLoader')
sys.path.append('../AutoFormula')

import BackTester
import FactorTester
import DataLoader
import AutoFormula


class AutoFactory:
    def __init__(self, start_date, end_date):
        """
        :param start_date: 总体的开始日期
        :param end_date: 总体的结束日期
        """
        self.start_date = start_date
        self.end_date = end_date
        self.back_tester = BackTester()  # 模拟交易回测
        self.autoformula = AutoFormula

        self.dataloader = DataLoader()
        self.data = self.dataloader.get_matrix_data()

    def factortest(self, formula, start_date, end_date):  # 测试因子
        """
        :param formula: 回测的公式
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        :return: 无返回值
        """

        self.autoformula.test_formula(formula, start_date, end_date)

    def backtest(self, signal, start_date, end_date):
        pass

    def stock_predict(self):
        pass

    def train(self):
        pass
