import matplotlib.pyplot as plt
from src.utils import *
import random
import time


def SVD_PCA(X, k):
    """
    利用SVD加速得到特征值和特征向量
    :param X: 最终要得到X_XT的，先计算XT_X
    :param k: 降到k维
    :return:
    """
    XT = np.transpose(X)  # X的转秩
    XT_X = np.dot(XT, X)  # XT * X
    # SVD 得特征值和特征向量
    start = time.time()
    eigenvalue, feature_vector = np.linalg.eig(XT_X)
    end = time.time()
    print("SVD方法中400*400矩阵XT_X计算特征值和特征向量的时间为：%f秒" % (end-start))
    idx = np.argsort(-eigenvalue)  # 从小到大排序,返回索引, 加个负号就是从大到小
    eigenvalue = eigenvalue[idx]
    feature_vector = feature_vector[:, idx]
    # 取前k个
    LAMBDA = eigenvalue[0:k]
    V = feature_vector[:, 0:k]
    U = np.dot(X, V) / np.sqrt(LAMBDA)
    return U


def get_Z_SVD(U, X, X_mean, k, j):
    '''
    得到SVD分解方法，人脸的低维表示，并重构出图像保存
    :param U:
    :param X:
    :param X_mean:
    :param k: 降到了k维
    :param j: 文件夹中的第几个
    :return: Z,就是低维表示
    '''
    img = X[:, j]
    Z = np.dot(np.transpose(U), X[:, j])
    print("SVD Z:\n", Z)
    Z = Z.reshape((k, 1))
    image = (np.dot(U, Z) + X_mean).reshape(112, 92)
    str1 = "../IMG/SVD" + str(j + 1) + "_k_" + str(8) + ".png"
    plt.imshow(image, plt.cm.gray)
    plt.savefig(str1)
    return Z


def ML_PCA(X, k, p):
    S = np.cov(X, bias=True)  # bias = True表示有偏估计,分母N，无偏估计N-1
    start = time.time()
    eigenvalue, feature_vector = np.linalg.eig(S)
    end = time.time()
    print("ML方法中10304*10304矩阵S计算特征值和特征向量的时间为：%f秒" % (end - start))
    # np.linalg.eig以复数的形式运算，算法在收敛时，虚部可能还没有完全收敛到0，
    # 但是都已经很小了，计算的时候可以直接取实部
    eigenvalue = np.real(eigenvalue)
    feature_vector = np.real(feature_vector)
    idx = np.argsort(-eigenvalue)  # 排序,返回索引
    eigenvalue = eigenvalue[idx]
    feature_vector = feature_vector[:, idx]
    U_k = feature_vector[:, 0:k]  # S的前k个最大特征值对应的特征向量
    A_k = np.diag(eigenvalue[0:k])
    SEGMA2 = np.sum(eigenvalue[k:p]) / (p - k)
    W = np.dot(U_k, np.sqrt(A_k - SEGMA2 * np.eye(k)))
    return W, SEGMA2



def get_Z_ML(W, SEGMA2, X, X_mean, k, j):
    img = X[:, j]
    C = np.dot(np.transpose(W), W) + SEGMA2 * np.eye(k)
    C_1 = np.linalg.inv(C)
    Z = np.dot(np.dot(C_1, np.transpose(W)), img)
    print("ML Z:\n", Z)
    Z = Z.reshape((k, 1))
    image = (np.dot(W, Z) + X_mean).reshape(112, 92)
    str1 = "../IMG/ML" + str(j + 1) + "_k_" + str(8) + ".png"     # k=8重构图像X保存
    plt.imshow(image, plt.cm.gray)
    plt.savefig(str1)
    return Z



def EM_PCA(X, p, k):
    W = np.random.rand(p, k)  # 正态分布初始化W
    Z0 = np.zeros((k, n))
    for i in range(0, 1000):
        # E步
        WT = np.transpose(W)
        WT_W = np.dot(WT, W)
        WT_W_1 = np.linalg.inv(WT_W)
        Z = np.dot(np.dot(WT_W_1, WT), X)
        # M步
        ZT = np.transpose(Z)
        Z_ZT = np.dot(Z, ZT)
        Z_ZT_1 = np.linalg.inv(Z_ZT)
        W = np.dot(np.dot(X, ZT), Z_ZT_1)
        if (np.allclose(Z0, Z)):    # 两个矩阵足够相似就当作已经收敛
            print("EM算法在第%d次收敛" % (i+1))
            break
        else:
            Z0 = Z
    return W, Z

def get_Image_EM(W, Z, X_mean, k, j):
    print("EM Z:\n", Z[:, j])
    Z = Z[:, j].reshape((k, 1))
    image = (np.dot(W, Z) + X_mean).reshape(112, 92)
    str1 = "../IMG/EM" + str(j + 1) + "_k_" + str(8) + ".png"
    plt.imshow(image, plt.cm.gray)
    plt.savefig(str1)




if __name__ == "__main__":
    p = 10304
    n = 10
    k = 8
    # dir = random.randint(1, 40)  # 随机选择一个文件夹的人脸图像
    dir = 38
    print("本次PCA选择的是第%d个文件夹的人脸图像" % dir)
    X = get_original_X(p, n, dir)
    X_mean = np.mean(X, axis=1)
    X_mean = X_mean.reshape(p, 1)
    X = X - X_mean  # X居中

    # j = random.randint(0, n - 1)  # 随机选择一张图片去计算表示和重构
    j = 2
    print("本次选择重构的是s%d文件夹里第%d个人脸图像" % (dir, j+1))
    # SVD
    U = SVD_PCA(X, k)
    get_Z_SVD(U, X, X_mean, k, j)

    # Likelihood
    W_ML, SEGMA2 = ML_PCA(X, k, p)
    get_Z_ML(W_ML, SEGMA2, X, X_mean, k, j)

    # SEGMA2 = 0, EM
    W_EM, Z_EM = EM_PCA(X, p, k)
    get_Image_EM(W_EM, Z_EM, X_mean, k, j)