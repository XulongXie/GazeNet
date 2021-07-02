import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from ..utils.irisDetiction import Iris_Detec as detector

def XY_Cord(array):
    X = []
    Y = []
    # what is returned to us in the array is the (column, row) coordinate point
    for item in array:
        x = item[0]
        y = item[1]
        X.append(x)
        Y.append(y)
    return X, Y


def plot(X, Y):
    # set the step length and start and end positions of the x-axis and y-axis
    # plt.figure(figsize=(10, 15), dpi=100)
    # figure as a canvas, can it be used or not
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax = 224, xmin = 0)
    plt.ylim(ymax = 224, ymin = 0)
    '''
    plt.xticks(range(0, 20, 1))
    plt.yticks(range(0, 20, 1))
    '''
    # the color of the point, can be queried：https://blog.csdn.net/w576233728/article/details/86538060
    colors1 = '#FF0000'
    colors2 = '#FFA500'
    colors3 = '#008000'
    colors4 = '#0000FF'
    colors5 = '#A0522D'
    colors6 = '#800080'
    colors7 = '#000000'
    colors8 = '#FFFF00'
    colors9 = '#FFC0CB'
    # point area, may also need to be changed
    area = np.pi * 1 ** 2
    # draw a scatter plot, where s is the size of the point and alpha is the transparency of the point
    plt.scatter(X[0], Y[0], s = area, c = colors1, alpha = 0.4, label = 'Class A')
    plt.scatter(X[1], Y[1], s = area, c = colors2, alpha = 0.4, label = 'Class B')
    plt.scatter(X[2], Y[2], s = area, c = colors3, alpha = 0.4, label = 'Class C')
    plt.scatter(X[3], Y[3], s = area, c = colors4, alpha = 0.4, label = 'Class D')
    plt.scatter(X[4], Y[4], s = area, c = colors5, alpha = 0.4, label = 'Class E')
    plt.scatter(X[5], Y[5], s = area, c = colors6, alpha = 0.4, label = 'Class F')
    plt.scatter(X[6], Y[6], s = area, c = colors7, alpha = 0.4, label = 'Class G')
    plt.scatter(X[7], Y[7], s = area, c = colors8, alpha = 0.4, label = 'Class H')
    plt.scatter(X[8], Y[8], s = area, c = colors9, alpha = 0.4, label = 'Class I')
    # draw a straight line, maybe not need it then
    # plt.plot([0, 9.5], [9.5, 0], linewidth = '0.5', color = '#000000')
    # label location
    plt.legend(loc="upper right")
    # path can be changed according to request
    plt.savefig('./first_dataset/Right.jpg', dpi = 300)
    plt.show()

def fun(srcPath):
    # get the images
    img_list = os.listdir(srcPath)
    x = []
    y = []
    for img in img_list:
        Path = srcPath + '/' + img
        src_img = cv2.imread(Path)
        left_x, left_y, right_x, right_y, x_center, y_center = detector(src_img)
        x.append(right_x)
        y.append(right_y)
    x = np.array(x)
    y = np.array(y)
    return x, y

if __name__ == '__main__':

    X = []
    Y = []
    Path = "../Dataset/first_dataset/train"
    # read all kinds of folders in the data set folder
    dir_list = os.listdir(Path)
    for dir_name in dir_list:
        srcPath = Path + "/" + dir_name
        # find the centroid of each class
        x, y = fun(srcPath)
        X.append(x)
        Y.append(y)

    plot(X, Y)
