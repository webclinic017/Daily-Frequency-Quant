# Copyright (c) 2021 Dai HBG

"""
该代码定义一个调用FormulaTree类生成公式树的自动化公式生成器，然后返回一个公式
"""

from FormulaTree import FormulaTree


class AutoFormula:
    def __init__(self, height=3, symmetric=False):
        self.height = height
        self.symmetric = symmetric
        self.tree = FormulaTree()
        self.tree.init_tree(height=self.height, symmetric=self.symmetric)

    def cal_formula(self):  # 定义根据公式树计算得到因子值的方法
        pass