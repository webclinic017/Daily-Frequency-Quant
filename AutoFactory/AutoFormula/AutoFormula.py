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

    def cal_formula(self, tree, data_dic):  # 递归计算公式树的值
        """
        :param tree: 需要计算的公式树
        :param data_dic: 原始数据的字典，可以通过字段读取对应的矩阵
        :return: 返回计算好的signal矩阵
        """
        if tree.variable_type == 'data':
            return data_dic[tree.name].copy()  # 当前版本需要返回一个副本
        else:
            if tree.operation_type == '1':
                return self.operation[tree.name](tree.left)
            if tree.operation_type == '1_num':
                return self.operation[tree.name](tree.left, tree.num)
            if tree.operation_type == '2':
                return self.operation[tree.name](tree.left, tree.right)
            if tree.operation_type == '2_num':
                return self.operation[tree.name](tree.left, tree.right, tree.num)

    def change_data(self, tree, p=0.7):
        """
        :param tree: 需要被改变叶子节点数据的树
        :param p: 每个数据单独被改变的概率
        :return: 没有返回值，直接修改
        """
        if tree.variable_type == 'data':
            if np.random.uniform() < p:
                if type(tree.name) == np.int64:
                    if tree.father_name in []:
                        pass
                    else:
                        pass
                else:
                    tree.name = np.random.choice(self.datas)[0]
        else:
            if tree.left in not None:
                self.change_data(tree.left)
            if tree.num in not None:
                self.change_data(tree.num)
            if tree.right in not None:
                self.change_data(tree.right)


