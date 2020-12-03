import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

def load_data(path):
    data = np.loadtxt(path, dtype=np.float32, delimiter=",")
    print(data.shape)
    return data

def get_tranform_data(data):
    # x1, x2 ——>  c, x1, x2, x1*x2, x1^2, x2^2
    constant = np.ones((data.shape[0]))
    x1_x2 = data[:, 0] * data[:, 1]
    square_x1 = data[:, 0] * data[:, 0]
    square_x2 = data[:, 1] * data[:, 1]
    transformed_data = np.insert(data, 0, values=constant, axis=1)
    transformed_data = np.insert(transformed_data, 3, values=square_x2, axis=1)
    transformed_data = np.insert(transformed_data, 3, values=square_x1, axis=1)
    transformed_data = np.insert(transformed_data, 3, values=x1_x2, axis=1)
    # print(transformed_data.shape)
    # print(transformed_data[0, :])
    return transformed_data

def split_data_x_y(data):
    # 分开数据和标签
    split_data_x_y = np.split(data, [data.shape[1]-1], 1)
    return split_data_x_y[0], split_data_x_y[1]

def split_data(data):
    # 根据最后一列（label）分数据集, 先找划分点
    column = data.shape[1]-1
    split_point = np.where(data[:, column] == 0)
    split_data = np.split(data, (split_point[0][0], ))
    return split_data[0], split_data[1]

def get_pred_label(fx):
    pred_label = np.round(fx)     # 四舍五入，因为0.5为界
    return pred_label

def get_accuracy(pred_label, true_label):
    same = (pred_label == true_label)       # 得到一个同维度的boolean矩阵
    num = np.sum(same)     # True被认为是1，False被认为是0，sum就是计算其中true的数量，用numpy.count_nonzero()计算非0元素的个数也行
    acc = 1.0 * num / true_label.shape[0]
    print("acc num: %d, acc: %f" % (num, acc))
    return acc

def plot_acc(train_acc_list, test_acc_list):
    time = np.arange(0, len(train_acc_list))
    plt.plot(time, train_acc_list)
    plt.plot(time, test_acc_list)
    plt.legend(["train acc", "test acc"], loc=0)
    plt.savefig("acc.png")
    # plt.show()


def plot_point(list1, list2):
    # 绘制数据集
    plt.scatter(list1[:, 0], list1[:, 1])
    plt.scatter(list2[:, 0], list2[:, 1])
    plt.legend(["y=1", "y=0"], loc=0)
    # plt.show()

def plot_decision_border(data, weights):
    list1, list2 = split_data(data)
    plt.scatter(list1[:, 0], list1[:, 1])
    plt.scatter(list2[:, 0], list2[:, 1])
    plt.legend(["y=1", "y=0"], loc=0)
    x1Min, x1Max, x2Min, x2Max = data[:, 0].min(), data[:, 0].max(), data[:, 1].min(), data[:, 1].max()     # 找范围
    xx1, xx2 = np.meshgrid(np.linspace(x1Min, x1Max), np.linspace(x2Min, x2Max))
    h = sigmoid(get_tranform_data(np.c_[xx1.ravel(), xx2.ravel()]).dot(weights))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], colors='red')
    plt.savefig("decision.png")
    # plt.show()

def sigmoid(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid

def sklearn_result(train_x, test_x, train_y, test_y):
    classifier = LogisticRegression()
    classifier.fit(train_x, train_y.ravel())
    train_pred = classifier.predict(train_x)
    train_acc = accuracy_score(train_y, train_pred)
    test_pred = classifier.predict(test_x)
    # print(test_y.T)
    # print(test_pred)
    # cm = confusion_matrix(test_y, test_pred)
    # print(cm)
    test_acc = accuracy_score(test_y, test_pred)
    # print(acc)
    f1 = f1_score(np.append(train_y, test_y, axis=0), np.append(train_pred, test_pred, axis=0), average="weighted")
    print("sklearn || train_acc: %f, test_acc: %f, f1_score: %f" % (train_acc, test_acc, f1))

def gradAscent(train_x, test_x, train_y, test_y, lr, decay, epoch):
    m, n = train_x.shape  # m：样本个数，n：属性个数
    weights = np.ones((n, 1))  # 这是要求的参数，初始化
    train_acc_list = []
    test_acc_list = []
    f1_list = []
    # train_x = np.mat(train_x)     # 这是转换成numpy.matrix
    # weights = np.mat(weights)
    for i in range(0, epoch):
        # print("这是第几次：", i)
        fx = sigmoid(np.matmul(train_x, weights))  # 预测值
        # fx = sigmoid(train_x * weights)  # 预测值
        error = train_y - fx      # m*1的
        # print("error:", error.T)
        grad = np.matmul(train_x.T, error)   # n*m 和 m*1  得到 n*1
        # grad = train_x.T * error
        lr_i = lr * 1.0 / (1.0 + decay * i)     # 学习率衰减
        weights = weights + lr_i * grad       # 梯度更新
        # 训练集
        pred_label = get_pred_label(fx)     # 预测的标签
        train_acc = accuracy_score(train_y, pred_label)
        train_acc_list.append(train_acc)
        # 测试集
        test_fx = sigmoid(np.matmul(test_x, weights))
        test_pred_label = get_pred_label(test_fx)  # 预测的标签
        test_acc = accuracy_score(test_y, test_pred_label)
        test_acc_list.append(test_acc)
        print(weights.T)
        f1 = f1_score(np.append(train_y, test_y, axis=0), np.append(pred_label, test_pred_label, axis=0), average="weighted")
        f1_list.append(f1_score)
        print("result || train_acc: %f, test_acc: %f, f1_score: %f" % (train_acc, test_acc, f1))
    # plot_acc(train_acc_list, test_acc_list)
    return weights

def main():
    path = "ex2data2.txt"
    data = load_data(path)
    transformed_data = get_tranform_data(data)
    lr = 0.1
    decay = 0.1
    epoch = 1000
    x, y = split_data_x_y(transformed_data)  # 数据和标签分开, random_state=0
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)  # train_x, test_x, train_y, test_y
    print(train_x.shape, train_y.shape)
    weights = gradAscent(train_x, test_x, train_y, test_y, lr, decay, epoch)
    sklearn_result(train_x, test_x, train_y, test_y)
    plot_decision_border(data, weights)

if __name__ == "__main__":
    main()