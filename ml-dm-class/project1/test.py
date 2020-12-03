import random
import numpy as np
import matplotlib.pyplot as plt

def get_points():
    point_list = []
    for i in range(0, 20):
        x1 = random.random() * 100
        x2 = random.random() * 100
        point_list.append([x1, x2])
    # print(type(point_list))
    point_list = np.array(point_list)
    # print(point_list)
    # print(type(point_list))
    # print(point_list[:, 0])
    return point_list

def get_target():
    w_list = []
    w0 = random.random() * 10
    w1 = random.random() * 10
    w2 = random.random() * (-10)
    w_list.append(w0)
    w_list.append(w1)
    w_list.append(w2)
    return w_list

def plt_point_target(point_list1, point_list2, tar_w_list):
    x = np.arange(-1, 100, 1)
    y = -(tar_w_list[0] + tar_w_list[1]*x)/tar_w_list[2]
    plt.plot(x, y, "red")
    plt.legend(["target"], loc=0)
    plt.scatter(point_list1[:, 0], point_list1[:, 1], marker='+')
    plt.scatter(point_list2[:, 0], point_list2[:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig("points_target.png")
    plt.show()


def plt_tar_hypo(point_list1, point_list2, tar_w_list, w_list):
    x = np.arange(-1, 100, 1)
    y1 = -(tar_w_list[0] + tar_w_list[1] * x) / tar_w_list[2]
    y2 = -(w_list[0] + w_list[1]*x)/w_list[2]
    plt.plot(x, y1, "red")
    plt.plot(x, y2, "green")
    plt.legend(["target", "hypothesis"], loc=0)
    plt.scatter(point_list1[:, 0], point_list1[:, 1], marker="+")
    plt.scatter(point_list2[:, 0], point_list2[:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig("points_target_hypothesis.png")
    # plt.show()

def get_label(p, w_list):
    label = -1
    y = -(w_list[0] + w_list[1] * p[0]) / w_list[2]
    if(p[1] > y):
        label = 1
    else:
        label = -1
    return label


if __name__ == "__main__":
    lr = 0.0001                          # learning rate
    points = get_points()           # data set
    print("points:", points)
    tar_w_list = get_target()       # get target function
    print("target function: %f + %f*x1 + %f*x2" % (tar_w_list[0], tar_w_list[1], tar_w_list[2]))
    # 得到样本集的标签 [1, -1]
    labels = []
    points_list1 = []
    points_list2 = []
    for i in range(0, points.shape[0]):
        t = get_label(points[i, :], tar_w_list)
        labels.append(t)
        if(t == 1):
            points_list1.append(points[i, :])
        else:
            points_list2.append(points[i, :])
    points_list1 = np.array(points_list1)
    points_list2 = np.array(points_list2)
    print(labels)
    print(points_list1.shape)
    # 画出数据集和目标函数
    plt_point_target(points_list1, points_list2, tar_w_list)
    # perceptron initialization
    w_list = []
    w0 = random.random() * 10
    w1 = random.random() * 10
    w2 = random.random() * (-10)
    w_list.append(w0)
    w_list.append(w1)
    w_list.append(w2)

    # 感知机开始迭代
    iter = True
    iter_num = 0
    while(iter):
        print('_____________________________')
        out_list = []
        for j in range(0, points.shape[0]):
            t = labels[j]
            o = get_label(points[j, :], w_list)
            out_list.append(o)
            # 参数更新过程
            w_list[0] = w_list[0] + lr * (t - o)
            w_list[1] = w_list[1] + lr * (t - o) * points[j, 0]
            w_list[2] = w_list[2] + lr * (t - o) * points[j, 1]
        print(labels)
        print(out_list)
        iter_num = iter_num + 1
        # 若迭代超过1000次还未正确分类，说明学习率过大，参数无法更新到合适的值
        if(iter_num > 1000):
            plt_tar_hypo(points_list1, points_list2, tar_w_list, w_list)
            break
        # 全分类正确，感知机学习结束
        if(labels == out_list):
            print(points)
            print("iter_num:", iter_num)
            plt_tar_hypo(points_list1, points_list2, tar_w_list, w_list)
            print("target function: %f + %f*x1 + %f*x2" % (tar_w_list[0], tar_w_list[1], tar_w_list[2]))
            print("hypothesis: %f + %f*x1 + %f*x2" % (w_list[0], w_list[1], w_list[2]))
            iter = False



