from src.utils import *
import random


def pegasos(data, label, C=0.1, iter_times=50):
    # weights = np.random.rand(n, 1)
    # bias = random.random()
    weights = np.zeros((n, 1))
    bias = 0
    my_lambda = 1 / (C * m)
    print(my_lambda)
    train_acc_list = []
    test_acc_list = []
    for j in range(1, iter_times+1):
        eta = 1 / (my_lambda * j)   # 学习率
        i = random.randint(0, m-1)
        print("第%d次选择的是第%d个" % (j, i))
        yi = label[i]
        xi = data[i, :]
        temp = yi * (np.dot(xi, weights) + bias)
        print(temp)
        if temp < 1:
            weights = weights - eta * (my_lambda * weights - yi * xi.reshape(n, 1))
            bias = bias - eta * (-yi)
        else:
            weights = weights - eta * my_lambda * weights
            bias = bias - eta * 0
        print(bias)
        print("Train dataset:")
        train_acc = get_acc(weights, bias, data, label)
        train_acc_list.append(train_acc)
        test_data, test_label = load_data('test')
        print("Test dataset:")
        test_acc = get_acc(weights, bias, test_data, test_label)
        test_acc_list.append(test_acc)
    return train_acc_list, test_acc_list

# 这玩意结果一点也不好，不用管了
def batch_pegasos(data, label, C=0.1, iter_times=50, batch=10):
    # weights = np.random.rand(n, 1)
    # bias = random.random()
    weights = np.zeros((n, 1))
    bias = 0
    my_lambda = 1 / (C * m)
    print(my_lambda)
    train_acc_list = []
    test_acc_list = []
    for j in range(1, iter_times+1):
        eta = 1 / (my_lambda * j)   # 学习率
        for k in range(batch):
            i = random.randint(0, m - 1)
            yi = label[i]
            xi = data[i, :]
            temp = yi * (np.dot(xi, weights) + bias)
            weights_delta = 0
            bias_delta = 0
            if temp < 1:
                weights_delta += yi * xi.reshape(n, 1)
                bias_delta += -yi
            weights = weights - eta * (my_lambda * weights - weights_delta / batch)
            bias = bias - eta * bias / batch
        print(bias)
        print("Train dataset:")
        train_acc = get_acc(weights, bias, data, label)
        train_acc_list.append(train_acc)
        test_data, test_label = load_data('test')
        print("Test dataset:")
        test_acc = get_acc(weights, bias, test_data, test_label)
        test_acc_list.append(test_acc)

    return train_acc_list, test_acc_list

data, label = load_data('train')
m, n = data.shape
# train_acc, test_acc = pegasos(data, label, iter_times=1000)
train_acc, test_acc = batch_pegasos(data, label, iter_times=1000, batch=20)
plot_acc(train_acc, test_acc)
