from functools import reduce
import random
import numpy
import math


def sigmoid(inX):
    """
    定义sigmoid函数
    :param inX: 输入
    :return: sigmoid输出
    """
    return 1.0 / (1 + math.exp(-inX))


class Node(object):
    """
    Node节点对象计算和记录节点自身的信息，如（输出值a，误差项δ）等，以及与这个节点相关的向下游的连接
    """
    def __init__(self, layer_index, node_index):
        """
        构造节点对象
        :param layer_index: 节点所属的层的编号
        :param node_index: 节点的编号
        """
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self,output):
        """
        设置节点的输出值，如果节点属于输入层，会用到这个函数。
        :param output: 输出
        :return: 返回输入层节点的输出（即输入）
        """
        self.output = output

    def append_downstream_connection(self,conn):
        """
        添加一个到下游节点的连接
        :param conn: 连接
        :return: 增加一个下游连接
        """
        self.downstream.append(conn)

    def append_upstream_append(self,conn):
        """
        添加一个到上游节点的连接
        :param conn: 连接
        :return: 增加一个上游连接
        """
        self.upstream.append(conn)

    def calc_output(self):
        """
        根据：y=sigmoid(W·X) 计算节点的输出
        :return: 返回output
        """
        output = reduce(
            lambda ret, conn: ret + conn.upstream_node.output*conn.weight,
            self.upstream, 0)
        # reduce 的第三个参数是初始值
        # reduce(function, iterable[, initializer])
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        """
        几点属于隐藏层时，根据式子4计算delta
        :return: 该节点的delta
        """
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        """
        节点是输出层时，根据式子3计算delta
        :param label: 样本标记
        :return: 输出层节点delta
        """
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        """
        打印节点信息
        :return: 
        """
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(object):
    """
    为了实现一个输出横位1的节点，计算偏置项Wb时需要
    """
    def __init__(self, layer_index, node_index):
        """
        构造节点对象
        :param layer_index: 节点所属层的编号
        :param node_index: 节点的编号
        """
        self.lay_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1
        self.delta = 0

    def append_downstream_connection(self, conn):
        """
        添加一个到下游节点的连接
        :param conn:节点 
        :return: 添加downstream
        """
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        """
        节点属于隐藏层的时候，根据式4计算delta
        :return: 
        """
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        """
        打印节点信息
        :return: 
        """
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Layer(object):
    """
    层对象，由多个节点组成
    """
    def __init__(self, layer_index, node_count):
        """
        初始化一层
        :param layer_index: 层编号
        :param node_count: 层所包含节点的个数
        """
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        # 加一个偏置项
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        """
        设置层的输出。当层是输入层时用到
        :param data: 
        :return: 
        """
        for i in range(len(data)):
            self.nodes[i].set_ouput(data[i])

    def calc_output(self):
        """
        计算层的输出向量
        :return: 
        """
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        """
        打印层的信息
        :return: 
        """
        for node in self.nodes:
            print(node)


class Connection(object):
    """
    更新，记录连接的权重，以及这个链接所关联的上下游节点
    """
    def __init__(self, upstream_node, downstream_node):
        """
        初始化连接，权重初始化为是一个很小的随机数
        :param upstream_node: 连接的上游节点
        :param downstream_node: 连接的下游节点
        """
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        """
        计算梯度
        :return: 
        """
        self.gradient = self.downstream_node.delta*self.upstream_node.output

    def get_gradient(self):
        """
        获取当前梯度
        :return: 
        """
        return self.gradient

    def update_weight(self, rate):
        """
        根据梯度下降法更新权重
        :param rate: 
        :return: 
        """
        self.calc_gradient()
        self.weight += rate*self.gradient

    def __str__(self):
        """
        打印连接信息
        :return: 
        """
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)


class Connections(object):
    def __init__(self):
        """
        提供connection集合操作
        """
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)


class Network(object):
    """
    提供API
    """
    def __init__(self, layers):
        """
        初始化一个全连接神经网络
        :param layers: 二维数组，描述神经网络每层节点数
        """
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count-1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)

    def train(self, labels, data_set, rate, iteration):
        """
        训练神经网络
        :param labels: 数组，训练样本的标签。每个元素是一个样本的标签
        :param data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
        :param rate: 学习率
        :param iteration: 
        :return: 
        """
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_example(labels[d], data_set[d], rate)

    def train_one_example(self, label, sample, rate):
        """
        内部函数，用一个样本训练网络
        :param label: 
        :param sample: 
        :param rate: 
        :return: 
        """
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        """
        内部函数，计算每个节点的delta
        :param label: 
        :return: 
        """
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        """
        内部函数，更新每个连接权重
        :param rate: 
        :return: 
        """
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        """
        内部函数，计算每个连接的梯度
        :return: 
        """
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        """
        获得网络在一个样本下，每个连接上的梯度
        :param label: 样本标签
        :param sample: 样本输入
        :return: 
        """
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        """
        根据输入的样本预测输出值
        :param sample: 
        :return: 
        """
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def dump(self):
        """
        打印网络信息
        :return: 
        """
        for layer in self.layers:
            layer.dump()

