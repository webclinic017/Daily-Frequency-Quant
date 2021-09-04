# Copyright (c) 2021 Dai HBG

"""
FormulaTree类定义了公式树变异的方法
"""


import numpy as np


class Node:
    def __init__(self, name, variable_type, operation_type=None,
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


class FormulaParser:
    def __init__(self):
        self.operation_dic = {'1': ['csrank', 'zscore', 'neg'],
                              '1_num': ['wdirect', 'tsrank', 'tskurtosis', 'tsskew',
                                        'tsmean', 'tsstd', 'tsdelay', 'tsdelta'],
                              '2': ['add', 'prod', 'minus', 'div'],
                              '2_num': ['tscorr']}

        dic = {}
        for key, value in self.operation_dic.items():
            for v in value:
                dic[v] = key
        self.operation_type_dic = dic

    def parse(self, s):
        """
        :param s: 待解析字符串
        :return: 返回一棵树
        """
        if '{' not in s:
            try:
                s = int(s)
            except ValueError:
                try:
                    s = float(s)
                except ValueError:
                    pass
            return Node(name=s, variable_type='data')
        else:
            # 定位到名称
            a = 0
            while s[a] != '{':
                a += 1
            name = s[:a]
            node = Node(name=name, variable_type='operation', operation_type=self.operation_type_dic[name])
            if self.operation_type_dic[name] == '1':
                node.left = self.parse(s[a + 1:-1])
                return node
            if self.operation_type_dic[name] == '1_num':
                # 定位中间的逗号
                left_num = 0
                right_num = 0
                b = len(s) - 2
                while True:
                    if s[b] == '}':
                        right_num += 1
                    if s[b] == '{':
                        left_num += 1
                    if left_num == right_num:
                        break
                    b -= 1
                if left_num == 0:
                    b += 1
                c = b - 1  # 此时c的位置是算子最后一位符号
                while s[c] != ',':
                    c -= 1
                num = int(s[c + 1:b])
                node.num = Node(name=num, variable_type='data')
                node.left = self.parse(s[a + 1:c])
                return node
            if self.operation_type_dic[name] == '2':
                # 定位中间的逗号
                left_num = 0
                right_num = 0
                b = len(s) - 2
                while True:
                    if s[b] == '}':
                        right_num += 1
                    if s[b] == '{':
                        left_num += 1
                    if left_num == right_num:
                        break
                    b -= 1
                if left_num == 0:
                    b += 1
                c = b - 1  # 此时c的位置是算子最后一位符号
                while s[c] != ',':
                    c -= 1
                node.right = self.parse(s[c + 1:len(s)-1])
                node.left = self.parse(s[a + 1:c])
                return node
            if self.operation_type_dic[name] == '2_num':
                # 定位第二个逗号
                left_num = 0
                right_num = 0
                b = len(s) - 2
                while True:
                    if s[b] == '}':
                        right_num += 1
                    if s[b] == '{':
                        left_num += 1
                    if left_num == right_num:
                        break
                    b -= 1
                if left_num == 0:
                    b += 1
                c = b - 1  # 此时c的位置是算子最后一位符号
                while s[c] != ',':
                    c -= 1
                num = int(s[c + 1:b])
                node.num = Node(name=num, variable_type='data')

                b = c - 1
                while True:
                    if s[b] == '}':
                        right_num += 1
                    if s[b] == '{':
                        left_num += 1
                    if left_num == right_num:
                        break
                    b -= 1
                if left_num == 0:
                    b += 1
                d = b - 1  # 此时d的位置是算子最后一位符号
                while s[d] != ',':
                    d -= 1
                node.right = self.parse(s[d + 1:c])
                node.left = self.parse(s[a + 1:d])
                return node


class FormulaTree:
    def __init__(self, height=2, symmetric=False):
        """
        :param height: 树的最大深度
        :param symmetric: 是否需要对称
        """
        self.height = height
        self.symmetric = symmetric

        self.datas = ['open', 'high', 'low', 'close', 'vwap', 'tvr_ratio']
        for i in range(1, 31):
            for data in self.datas[:6]:
                tree = Node(name='tsdelay', variable_type='operation',
                            operation_type='1_num')
                tree.left = Node(name=data, variable_type='data')
                tree.num = i
                self.datas.append(tree)

        self.operation_dic = {'1': ['csrank', 'zscore'],
                              '1_num': ['wdirect', 'tsrank', 'tskurtosis', 'tsskew',
                                        'tsmean', 'tsstd', 'tsdelay', 'tsdelta'],
                              '2': ['add', 'prod', 'minus', 'div'],
                              '2_num': ['tscorr']}

        dic = {}
        for key, value in self.operation_dic.items():
            for v in value:
                dic[v] = key
        self.operation_type_dic = dic

        self.p = {'1': [1/2, 1/2], '1_num': [1/15, 4/15, 2/15, 2/15, 2/15, 2/15, 1/15, 1/15],
                  '2': [1/7, 3/7, 1/7, 2/7], '2_num': [1]}  # 不同操作被选中的概率

    def init_tree(self, height, symmetric=False):
        """
        :param height: 树的高度
        :param symmetric: 是否需要对称的树，默认非对称，这样生成的树多样性更好
        :return: 返回一个公式树
        """
        operation_type = np.random.choice(['1', '1_num', '2', '2_num'], 1,
                                          p=[1 / 10, 4 / 10, 4 / 10, 1 / 10])[0]
        operation = np.random.choice(self.operation_dic[operation_type], 1, p=self.p[operation_type])[0]  # 随机选取一个操作
        node = Node(name=operation, variable_type='operation', operation_type=operation_type)  # 无论如何先生成一个节点
        node.height = height  # 需要记录该节点代表的树的深度，以便之后的树的变异方法的使用
        if height == 1:  # 如果高度是1，直接生成叶子节点就可以返回
            data = np.random.choice(self.datas, 1)[0]
            node.left = Node(name=data, variable_type='data')
            if operation_type in ['1_num', '2_num']:
                if operation in ['tsdelta', 'tsdelay']:
                    num = np.random.choice([i for i in range(1, 11)], 1, p=[(15 - i) / 95 for i in range(1, 11)])[0]
                else:
                    num = np.random.choice([i for i in range(2, 32)], 1, p=[(32 - i) / 465  for i in range(2, 32)])[0]
                node.num = Node(name=num, variable_type='data')
                node.num.father_name = operation
            if operation_type in ['2']:
                data = np.random.choice(self.datas, 1)[0]
                node.right = Node(name=data, variable_type='data')
                return node
        else:
            if symmetric:  # 否则如果是对称，就根据操作类型递归生成子节点
                node.left = self.init_tree(height-1, symmetric=symmetric)
                if operation_type in ['1_num', '2_num']:
                    if operation in ['tsdelta', 'tsdelay']:
                        num = np.random.choice([i for i in range(1, 11)], 1,
                                               p=[(15 - i) / 95 for i in range(1, 11)])[0]
                    else:
                        num = np.random.choice([i for i in range(2, 32)], 1,
                                               p=[(32 - i) / 465 for i in range(2, 32)])[0]
                    node.num = Node(name=num, variable_type='data')
                    node.num.father_name = operation
                if operation_type in ['2']:
                    node.right = self.init_tree(height-1, symmetric=symmetric)
                    return node
            else:  # 如果不对称，且运算符是双目的，则需要随机选取一边满足高度的约束
                if operation_type in ['1', '1_num']:
                    node.left = self.init_tree(height - 1, symmetric=symmetric)
                    if operation_type == '1_num':
                        if operation in ['tsdelta', 'tsdelay']:
                            num = np.random.choice([i for i in range(1, 11)], 1,
                                                   p=[(15 - i) / 95 for i in range(1, 11)])[
                                0]
                        else:
                            num = np.random.choice([i for i in range(2, 32)], 1,
                                                   p=[(32 - i) / 465 for i in range(2, 32)])[
                                0]
                        node.num = Node(name=num, variable_type='data')
                        node.num.father_name = operation
                else:
                    left_or_right = np.random.choice([0, 1])
                    if left_or_right == 0:
                        node.left = self.init_tree(height - 1, symmetric=symmetric)
                        right_height = np.random.choice([i for i in range(1, height)], 1)[0]
                        node.right = self.init_tree(right_height, symmetric=symmetric)
                    else:
                        node.right = self.init_tree(height - 1, symmetric=symmetric)
                        left_height = np.random.choice([i for i in range(1, height)], 1)[0]
                        node.left = self.init_tree(left_height, symmetric=symmetric)
                    if operation_type == '2_num':
                        if operation in ['tsdelta', 'tsdelay']:
                            num = np.random.choice([i for i in range(1, 11)], 1,
                                                   p=[(15 - i) / 95 for i in range(1, 11)])[
                                0]
                        else:
                            num = np.random.choice([i for i in range(2, 32)], 1,
                                                   p=[(32 - i) / 465 for i in range(2, 32)])[
                                0]
                        node.num = Node(name=num, variable_type='data')
                        node.num.father_name = operation
                return node

    def change_data(self, tree, p=0.7):
        """
        :param tree: 需要被改变叶子节点数据的树
        :param p: 每个数据单独被改变的概率
        :return: 没有返回值，直接修改
        """
        if tree.variable_type == 'data':
            if np.random.uniform() < p:
                if type(tree.name) == np.int64:
                    if node.father_name in ['tscorr', 'tsmean', 'tskurtosis', 'tsskew', 'tsrank']:  # 根据父节点的操作类型决定数据范围
                        node.name = 1 + np.random.choice([i for i in range(1, 31)], 1,
                                                         p=[(31 - i + 10) / 765 for i in range(1, 31)])[0]
                    else:
                        node.name = 1 + np.random.choice([i for i in range(1, 11)], 1,
                                                         p=[(11 - i + 4) / 95 for i in range(1, 11)])[0]
                else:
                    tree.name = np.random.choice(self.datas)[0]
        else:
            if tree.left is not None:
                self.change_data(tree.left, p=p)
            if tree.num is not None:
                self.change_data(tree.num, p=p)
            if tree.right is not None:
                self.change_data(tree.right, p=p)

    def change_structure(self):  # 改变树的结构，可以选择是否局部更改
        pass



