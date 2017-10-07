# _*_ coding: utf-8 _*_
# @Time     : 2017/10/6 21:51
# @Author    : Ligb
# @File     : KNN.py
"""构造一个简单的KNN分类器"""

from numpy import *
import operator


def create_data_set():
    """创建数据集和标签"""
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inx, data_set, labels, k):
    # 返回矩阵的第一维度的大小（行）
    data_set_size = data_set.shape[0]

    # 将输入的未知点处理成4*2的矩阵，便于计算距离
    # 不必循环，直接矩阵运算更方便
    diff_mat = tile(inx, (data_set_size, 1)) - data_set

    # 将矩阵的每行的值求和，求欧拉距离
    sqrt_diff_mat = diff_mat ** 2
    sqrt_distance = sqrt_diff_mat.sum(axis=1)
    distances = sqrt_distance ** 0.5

    # 将元素的索引值按照元素的大小排序
    sorted_distances = distances.argsort()
    class_count = {}

    # 获取距离最小的前k个点，并统计其中各个类的出现频率
    for i in range(k):
        vote_label = labels[sorted_distances[i]]

        # get（）不存在该键时返回0，存在该键时返回键的值
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    # operator.itemgetter(1)是一个函数赋值给key，决定以第一个域进行排序（即频率），返回一个列表
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

groups, labels = create_data_set()
print(classify([1, 1], groups, labels, 3))