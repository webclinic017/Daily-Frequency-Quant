# Copyright (c) 2021 Dai HBG

"""
FormulaTree类定义了公式树变异的方法
"""


class Node:
    def __init__(self, name, variable_type, operation_type,
                 left=None, right=None, num=None):
        """
        :param name: 操作或者数据的名字
        :param variable_type: 变量类型，data指的是数据，operation指的是算符
        :param operation_type: 算符类型，数字指的是多少目运算符，num指的是需要传入一个确定数字而不是运算结果
        :param left: 左子树，可以挂载一颗公式树或者数据节点
        :param right: 右子树，可以挂载一颗公式树或者数据节点
        :param num: 数字，如果算符需要传入一个数字
        """
        self.name = name
        self.variable_type = variable_type
        if self.variable_type == 'operation':
            self.operation_type = operation_type
        self.left = left
        self.right = right
        self.num = num


class FormulaTree:
    def __init__(self, height=2, symmetric=False):
        self.height = height
        self.symmetric = symmetric
        self.datas = ['open', 'high', 'low', 'close', 'vwap', 'tvr_ratio']

        for i in range(1, 31):
            for data in self.datas[:6]:
                tree = Node(name='tsdelay', variable_type='operation',
                                   operation_type='1_num')
                tree.left = Node(name=data, variable_type='data')
                self.datas.append(tree)

    def init_tree(self, height, symmetric=False):
        if symmetric:
            if height == 1:
                operation_type = np.random.choice(['1', '1_num', '2', '2_num'],
                                                  p=[1 / 10, 4 / 10, 4 / 10, 1 / 10])[0]