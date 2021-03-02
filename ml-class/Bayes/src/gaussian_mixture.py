import numpy as np
import math
import matplotlib.pyplot as plt

def gaussian(mean=0, std=1, x=1):
    return np.exp(-np.square(x - mean) / (2 * np.square(std))) / (np.sqrt(2 * math.pi) * std)

def get_color(index):
    color = []
    for i in range(index.shape[0]):
        c = 'c' if index[i]==0 else 'g'
        color.append(c)
    return color

def plot_res(mean1, sigma1, mean2, sigma2, X, color):
    Y = np.zeros(X.shape)
    plt.scatter(X, Y, c=color)
    x1 = np.linspace(mean1 - 6 * sigma1, mean1 + 6 * sigma1, 100)
    y1 = gaussian(mean1, sigma1, x1)
    x2 = np.linspace(mean2 - 6 * sigma2, mean2 + 6 * sigma2, 100)
    y2 = gaussian(mean2, sigma2, x2)
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.grid()
    plt.savefig("gmm.png")
    plt.show()

X = np.array([1.0, 1.3, 2.2, 2.6, 2.8, 5.0, 7.3, 7.4, 7.5, 7.7, 7.9])
n = 11
mean1 = 6    # ppt 6.63
mean2 = 7.5    # ppt 7.57
square_std1 = 1
square_std2 = 1
P_C = np.array([0.5, 0.5])
likelihood0 = 0
likelihood = 1
i = 1
while (np.abs(likelihood - likelihood0) > 0.0001):
    print("================这是第%d次迭代===============" % i)
    likelihood0 = likelihood
    i = i + 1
    # Expectation
    std1 = np.sqrt(square_std1)
    std2 = np.sqrt(square_std2)
    X_C1 = gaussian(mean1, std1, x=X)   # 条件概率 X|C
    X_C2 = gaussian(mean2, std2, x=X)
    P_XC1 = P_C[0] * X_C1   # 联合概率 P(X, C)
    P_XC2 = P_C[1] * X_C2
    P_XC = np.array([P_XC1, P_XC2])
    P_X = np.sum(P_XC, axis=0)
    P_X_tile = np.tile(P_X, (2, 1))
    P_C_X = P_XC / P_X_tile  # 后验概率P(C|X)， 有np.sum(P_C_X, axis=0)=1
    # print(P_C_X)

    # 计算似然函数的值
    sum_K = P_XC1 + P_XC2
    ln_sum_K = np.log(sum_K)
    likelihood = np.sum(ln_sum_K)
    # Maximization
    P_C = np.sum(P_C_X, axis=1) / n
    print("P(Ci):", P_C)
    mean1 = np.sum(np.multiply(X, P_C_X[0])) / np.sum(P_C_X[0])
    mean2 = np.sum(np.multiply(X, P_C_X[1])) / np.sum(P_C_X[1])
    print("均值：", mean1, mean2)
    square_std1 = np.sum(np.multiply(P_C_X[0], np.square(X-mean1))) / np.sum(P_C_X[0])
    square_std2 = np.sum(np.multiply(P_C_X[1], np.square(X-mean2))) / np.sum(P_C_X[1])
    print("方差：", square_std1, square_std2)

index = np.argmax(P_C_X, axis=0)
# print(index.shape)
color = get_color(index)
# print(color)
plot_res(mean1, np.sqrt(square_std1), mean2, np.sqrt(square_std2), X, color)