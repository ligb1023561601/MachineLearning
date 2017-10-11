# _*_ coding: utf-8 _*_
# @Time     : 2017/10/10 9:27
# @Author    : Ligb
# @File     : HandWritingNumberRecognize.py

from numpy import *
from os import listdir

import KNN


def image2vector(image_path):
    """
    将文本中的二维的图像信息处理为向量
    图片为32*32像素
    """
    image_vector = zeros((1, 1024))
    with open(image_path) as ip:
        image_lines = ip.readlines()
        i = 0
        for line in image_lines:
            line = line.rstrip()
            for j in range(len(line)):
                image_vector[0, i * 32 + j] = int(line[j])
            i += 1
    return image_vector


def get_sample_set(dir_path):
    """
    获取指定路径下的所有样本
    :param dir_path: 测试样本文件夹or训练样本文件夹
    :return: 样本集和标签列表
    """
    image_names = listdir('datas/'+ dir_path)
    image_numbers = len(image_names)
    sample_set_arr = zeros((image_numbers, 1024))
    sample_labels = []
    for image_index in range(image_numbers):
        sample_set_arr[image_index, :] = image2vector('datas/' + dir_path + '/' + image_names[image_index])[:]
        sample_labels.append(image_names[image_index].split('.')[0].split('_')[0])
    return sample_set_arr, sample_labels


def test_hand_writing_recognition():
    """
    测试手写数字识别分类器的性能
    :return:
    """
    error_count = 0
    test_array, test_labels = get_sample_set('testDigits')
    training_array,training_labels = get_sample_set('trainingDigits')
    for test_index in range(len(test_labels)):
        test_result = KNN.classify(test_array[test_index, :], training_array, training_labels, 3)
        if test_result != test_labels[test_index]:
            error_count += 1
        print('分类器返回的结果是：' + str(test_result) + ' 真实的结果是：' + str(test_labels[test_index]))
    print('测试样本总数为：' + str(len(test_labels)) + ',错误总数为：' + str(error_count))
    print('错误率为：' + str(error_count / len(test_labels)))


def test_one_number(file_path):
    """
    测试一个手写数字
    用法：利用计算机的画图软件手写一个数字（黑白，
    且尺寸为32*32，保存为png格式，执行脚本CharImage.py
    生成对应的TXT文件，然后调用此测试函数
    :param file_path: 保存的txt文件的路径
    :return:
    """
    image = image2vector(file_path)
    training_array, training_labels = get_sample_set('trainingDigits')
    print(KNN.classify(image, training_array, training_labels, 3))


test_hand_writing_recognition()
print(test_one_number(r'E:\PY_Coding\KNN\datas\0.txt'))






