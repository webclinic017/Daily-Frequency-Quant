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

    def init_tree(self, height, symmetric=False):
        """
        :param height: 树的高度
        :param symmetric: 是否需要对称的树，默认非对称，这样生成的树多样性更好
        :return: 返回一个公式树
        """
        operation_type = np.random.choice(['1', '1_num', '2', '2_num'],
                                          p=[1 / 10, 4 / 10, 4 / 10, 1 / 10])[0]
        operation = np.random.choice(self.operation_dic[operation_type])[0]  # 随机选取一个操作
        node = Node(name=operation, variable_type='operation', operation_type=operation_type)  # 无论如何先生成一个节点
        node.height = height  # 需要记录该节点代表的树的深度，以便之后的树的变异方法的使用
        if height == 1:  # 如果高度是1，直接生成叶子节点就可以返回
            data = np.random.choice(self.datas)[0]
            node.left = Node(name=data, variable_type='data')
            if operation_type in ['1_num', '2_num']:
                if operation in ['tsdelta', 'tsdelay']:
                    num = np.random.choice([i for i in range(1, 11)], p=[(15 - i) / 95 for i in range(1, 11)])[0]
                else:
                    num = np.random.choice([i for i in range(2, 32)], p=[(32 - i) / 480 for i in range(2, 32)])[0]
                node.num = Node(name=num, variable_type='data')
                node.num.father_name = operation
            if operation_type in ['2']:
                data = np.random.choice(self.datas)[0]
                node.right = Node(name=data, variable_type='data')
                return node
        else:
            if symmetric:  # 否则如果是对称，就根据操作类型递归生成子节点
                node.left = self.init_tree(height-1, symmetric=symmetric)
                if operation_type in ['1_num', '2_num']:
                    if operation in ['tsdelta', 'tsdelay']:
                        num = np.random.choice([i for i in range(1, 11)], p=[(15 - i) / 95 for i in range(1, 11)])[0]
                    else:
                        num = np.random.choice([i for i in range(2, 32)], p=[(32 - i) / 480 for i in range(2, 32)])[0]
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
                            num = np.random.choice([i for i in range(1, 11)], p=[(15 - i) / 95 for i in range(1, 11)])[
                                0]
                        else:
                            num = np.random.choice([i for i in range(2, 32)], p=[(32 - i) / 480 for i in range(2, 32)])[
                                0]
                        node.num = Node(name=num, variable_type='data')
                        node.num.father_name = operation
                else:
                    left_or_right = np.random.choice([0, 1])[0]
                    if left_or_right == 0:
                        node.left = self.init_tree(height - 1, symmetric=symmetric)
                        right_height = np.random.choice([i for i in range(1, height)])[0]
                        node.right= self.init_tree(right_height, symmetric=symmetric)
                    else:
                        node.right = self.init_tree(height - 1, symmetric=symmetric)
                        left_height = np.random.choice([i for i in range(1, height)])[0]
                        node.left = self.init_tree(left_height, symmetric=symmetric)
                    if operation_type == '2_num':
                        if operation in ['tsdelta', 'tsdelay']:
                            num = np.random.choice([i for i in range(1, 11)], p=[(15 - i) / 95 for i in range(1, 11)])[
                                0]
                        else:
                            num = np.random.choice([i for i in range(2, 32)], p=[(32 - i) / 480 for i in range(2, 32)])[
                                0]
                        node.num = Node(name=num, variable_type='data')
                        node.num.father_name = operation
                return node

    def change_structure(self):  # 改变树的结构，可以选择是否局部更改

