import scipy.io as scio
import numpy as np
import random
import matplotlib.pyplot as plt

def load_data(data_type):
    '''
    加载数据集, 训练集数据(4000, 1899)，测试集(1000, 1899)
    数据类型从int8改成int32, 为了将标签从0改成-1
    :param data_type:
    :return:
    '''
    train_path = '../data/spamTrain.mat'
    test_path = '../data/spamTest.mat'
    if data_type == 'train':
        train_data = scio.loadmat(train_path)
        # 因为读取的数据为int8, 0-1会溢出变成255，所以转成int32
        data = np.array(train_data['X'], dtype='int32')
        label = np.array(train_data['y'], dtype='int32')
        # 把标签中的0变成-1，1还是不变，只是为了与授课内容保持一致
        new_label = 2 * label - 1
    elif data_type == 'test':
        test_data = scio.loadmat(test_path)
        data = np.array(test_data['Xtest'], dtype='int32')
        label = np.array(test_data['ytest'], dtype='int32')
        new_label = 2 * label - 1
    return data, new_label


def get_acc(weights, bias, data, label):
    # weights = np.random.rand(1899, 1)
    # bias = random.random()
    temp = label * (np.dot(data, weights) + bias)
    num = sum(temp >= 1)
    acc = 1.0 * num / label.shape[0]
    print("correct_num: %d; accuracy: %f" % (num, acc))
    return acc

def plot_acc(train_acc_list, test_acc_list):
    time = np.arange(0, len(train_acc_list))
    plt.plot(time, train_acc_list)
    plt.plot(time, test_acc_list)
    plt.legend(["train acc", "test acc"], loc=0)
    plt.savefig("acc.png")
    plt.show()

def sklearn_svm():
    from sklearn import svm
    clf = svm.SVC(C=0.1, kernel='poly')
    # clf = svm.SVC(C=0.1)
    data, label = load_data('train')
    test_data, test_label = load_data('test')
    clf.fit(data, label.ravel())
    # print(clf.support_vectors_)
    acc = clf.score(data, label)
    test_acc = clf.score(test_data, test_label)
    print("train acc: %f, test acc: %f" % (acc, test_acc))

if "__name__ == __main__":
    sklearn_svm()
