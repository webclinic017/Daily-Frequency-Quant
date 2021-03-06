# Copyright (c) 2021 Dai HBG

"""
该代码定义一个调用FormulaTree类生成公式树的自动化公式生成器，然后返回一个公式

开发日志：
2021-09-13
-- 更新：AutoFormula类初始化需要传入一个data类
2021-09-20
-- 更新：新增多个算子
"""
import numpy as np
import sys
import datetime

sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/AutoFactory/Tester')

from AutoTester import AutoTester
from FormulaTree import FormulaTree, Node, FormulaParser
from SignalGenerator import SignalGenerator


class AutoFormula:
    def __init__(self, start_date, end_date, data, height=3, symmetric=False):
        """
        :param start_date: 该公式树
        :param end_date:
        :param data: Data实例
        :param height: 最大深度
        :param symmetric: 是否对称
        """
        self.height = height
        self.symmetric = symmetric
        self.start_date = start_date
        self.end_date = end_date
        self.tree_generator = FormulaTree()
        self.tree = self.tree_generator.init_tree(height=self.height, symmetric=self.symmetric)
        self.operation = SignalGenerator(data=data)
        self.formula_parser = FormulaParser()
        self.AT = AutoTester()

    def cal_formula(self, tree, data_dic, return_type='signal'):  # 递归计算公式树的值
        """
        :param tree: 需要计算的公式树
        :param data_dic: 原始数据的字典，可以通过字段读取对应的矩阵
        :param return_type: 返回值形式
        :return: 返回计算好的signal矩阵
        """
        if return_type == 'signal':
            if tree.variable_type == 'data':
                if type(tree.name) == int or type(tree.name) == float:
                    return tree.name
                return data_dic[tree.name].copy()  # 当前版本需要返回一个副本
            else:
                if tree.operation_type == '1':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type))
                if tree.operation_type == '1_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   tree.num.name)
                if tree.operation_type == '2':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   self.cal_formula(tree.right, data_dic, return_type))
                if tree.operation_type == '2_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   self.cal_formula(tree.right, data_dic, return_type),
                                                                   self.cal_formula(tree.num, data_dic, return_type))
                if tree.operation_type == '3':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   self.cal_formula(tree.middle, data_dic, return_type),
                                                                   self.cal_formula(tree.right, data_dic, return_type))
        if return_type == 'str':
            if tree.variable_type == 'data':
                return tree.name  # 返回字符串
            else:
                if tree.operation_type == '1':
                    return tree.name + '{' + (self.cal_formula(tree.left, data_dic, return_type)) + '}'
                if tree.operation_type == '1_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + str(
                        tree.num.name) + '}'
                if tree.operation_type == '2':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + '}'
                if tree.operation_type == '2_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.num, data_dic, return_type) + '}'
                if tree.operation_type == '3':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.middle, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + '}'

    def test_formula(self, formula, data, start_date=None, end_date=None, prediction_mode=False):
        """
        :param formula: 需要测试的因子表达式，如果是字符串形式，需要先解析成树
        :param data: Data类
        :param start_date: 如果不提供则按照Data类默认的来
        :param end_date: 如果不提供则按照Data类默认的来
        :param prediction_mode: 是否是最新预测模式，是的话不需要测试，只生成signal
        :return: 返回统计值以及该因子产生的信号矩阵
        """
        if not prediction_mode:
            if type(formula) == str:
                formula = self.formula_parser.parse(formula)
            signal = self.cal_formula(formula, data.data_dic)  # 暂时为了方便，无论如何都计算整个回测区间的因子值

            if start_date is None:
                start_date = str(data.start_date)
            if end_date is None:
                end_date = str(data.end_date)

            start, end = data.get_real_date(start_date, end_date)
            # return signal,start,end
            return self.AT.test(signal[start:end + 1], data.ret[start + 1:end + 2], top=data.top[start:end + 1]), signal
        else:
            if type(formula) == str:
                formula = self.formula_parser.parse(formula)
            return self.cal_formula(formula, data.data_dic)
