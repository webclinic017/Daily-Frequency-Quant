# Copyright (c) 2021 Dai HBG

"""
该代码定义一个调用FormulaTree类生成公式树的自动化公式生成器，然后返回一个公式
"""

import os
os.path.append('../BackTester')
from FormulaTree import FormulaTree
from SignalGenerator import SignalGenerator


class AutoFormula:
    def __init__(self, height=3, symmetric=False):
        self.height = height
        self.symmetric = symmetric
        self.tree = FormulaTree()
        self.tree.init_tree(height=self.height, symmetric=self.symmetric)
        self.operation = SignalGenerator()

    @staticmethod
    def cal_formula(tree, data_dic):  # 根据公式树递归计算得到因子值的方法
        """
        :param tree: 公式树
        :param data_dic: 原始数据字典
        :return: 返回信号值，形状和原始数据一致
        """
        if tree.height == 1:
            if tree.operation_type == '1':
                return self.operation.operation
