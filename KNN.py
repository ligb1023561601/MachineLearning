# _*_ coding: utf-8 _*_
# @Time     : 2017/10/6 21:51
# @Author    : Ligb
# @File     : KNN.py
"""构造一个简单的KNN分类器"""

from numpy import *
import operator
import matplotlib.pyplot as plt


def create_data_set():
    """创建数据集和标签"""
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inx, data_set, labels, k):
    # 返回矩阵的第一维度的大小（行）
    data_set_size = data_set.shape[0]

    # 将输入的未知点处理成相同大小的矩阵，便于计算距离
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


def file_to_matrix(filename):
    """从文本文件解析数据"""
    index = 0
    with open(filename) as txt:
        data_lines = txt.readlines()
        return_matrix = zeros((len(data_lines), 3))
        class_label = []
        for data_line in data_lines:
            data_line = data_line.strip()
            list_for_one_line = data_line.split('\t')

            # 逐行将文本中的数据添加至矩阵
            return_matrix[index, :] = list_for_one_line[0:3]
            index += 1

            # 添加标签
            class_label.append(int(list_for_one_line[-1]))
    return return_matrix, class_label


def data_visualization(x_label, y_label, x_data, y_data, data_matrix, class_label, title=None):
    """
    将文本中的数据可视化，可设置横纵轴数据源
    0：航空里程
    1:玩视频游戏所占时间，
    2：冰淇淋消耗量
    """

    # 数据可视处理
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    for i in range(len(class_label)):
        if class_label[i] == 1:  # 不喜欢
            type1_x.append(data_matrix[i, x_data])
            type1_y.append(data_matrix[i, y_data])

        if class_label[i] == 2:  # 魅力一般
            type2_x.append(data_matrix[i, x_data])
            type2_y.append(data_matrix[i, y_data])

        if class_label[i] == 3:  # 很有魅力
            type3_x.append(data_matrix[i, x_data])
            type3_y.append(data_matrix[i, y_data])

    type1 = ax.scatter(type1_x, type1_y, s=20, c='red')
    type2 = ax.scatter(type2_x, type2_y, s=40, c='blue')
    type3 = ax.scatter(type3_x, type3_y, s=60, c='yellow')
    # ax.scatter(data_matrix[:, 1], data_matrix[:, 2], 15.0*array(class_label), 15.0*array(class_label))
    ax.legend([type1, type2, type3], ['do not like', "generally like", "charming"], loc='best')
    ax.grid(True)
    plt.show()


def norm_data(data_set):
    """
    对数据进行归一化处理，
    以消除不同范围带来的影响
    注意这里的除法是矩阵对应的元素直接相除
    """
    min_value = data_set.min(0)
    max_value = data_set.max(0)
    value_range = max_value - min_value
    norm_date_set = data_set - tile(min_value, (data_set.shape[0], 1))
    norm_date_set /= tile(value_range, (data_set.shape[0], 1))
    return norm_date_set, min_value, value_range


def test_knn():
    """测试构造的简单KNN分类器"""
    groups, labels = create_data_set()
    print(classify([1, 1], groups, labels, 3))


def test_dating_class():
    """测试dating的结果,验证性能"""
    # 读取文本中的数据
    data_matrixs, class_labels = file_to_matrix(r"E:\PY_Coding\KNN\datas\datingTestSet2.txt")

    # 归一化处理
    data_matrixs, min_val, ranges = norm_data(data_matrixs)
    # data_visualization('GameTime', 'IceCreamConsumption', 1, 2, data_matrixs, class_labels)
    data_visualization('Miles', 'GameTime', 0, 1, data_matrixs, class_labels)

    error_count = 0
    test_sample_ratio = 0.10
    test_sample_numbers = int(test_sample_ratio * data_matrixs.shape[0])
    for i in range(test_sample_numbers):
        classified_result = classify(data_matrixs[i, :], data_matrixs[test_sample_numbers:, :],
                                     class_labels[test_sample_numbers:], 25)
        print('分类器结果为: ' + str(classified_result) + ' 真实结果为：' + str(class_labels[i]))
        if classified_result != class_labels[i]:
            error_count += 1.0
    print('错误率为： ' + str(error_count/float(test_sample_numbers)))


def classify_person():
    """对输入的信息进行判断"""
    labels = ['do not like', 'generally like', 'perfectly like']
    travel_miles = float(input('Please input the miles you travel in a year:'))
    percent_gaming_time = float(input('Please input the proportion of your gaming time in one day: '))
    ice_cream = float(input('Please input your ice_cream consumption:'))
    input_person_array = array([travel_miles, percent_gaming_time, ice_cream])

    data_matrixs, class_labels = file_to_matrix(r"E:\PY_Coding\KNN\datas\datingTestSet2.txt")
    data_matrixs, min_val, ranges = norm_data(data_matrixs)

    # 对输入数据做归一化处理
    result = classify((input_person_array - min_val) / ranges, data_matrixs, class_labels, 25)
    print('You will probably ' + labels[result - 1] + ' this man!!')

if __name__ == '__main__':
    test_knn()
    test_dating_class()
    classify_person()






