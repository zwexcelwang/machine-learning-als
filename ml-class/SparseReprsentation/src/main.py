import os
import numpy as np
import random
import time
from src.utils import read_pgm, write_Alpha_txt, write_B_txt, get_original_x, loss_function, plot_loss
import datetime
now_time = datetime.datetime.now()
print(now_time)
start = time.time()
p = 10304
n = 400
k = 40
my_lambda = 1
Alpha = np.zeros((k, n))
Alpha = np.mat(Alpha)
loss_list = []

path = "../orl_faces/"
X, B = get_original_x(path, p, n, k)


lam = my_lambda / 2
for i in range(0, 200):

    # 固定B，优化Alpha
    for j in range(0, n):
        alpha = Alpha[:, j]
        x = X[:, j]
        for m in range(0, k):
            for t in range(0, 10):
                alpha_m = float(alpha[m])
                B_m = B[:, m].reshape(1, -1)
                a_m = np.dot(x + np.dot(B[:, m], alpha_m).reshape(x.shape) - np.dot(B, alpha).reshape(x.shape), - B[:, m])
                b_m = np.dot(B_m, B[:, m])
                if (a_m > lam):
                    alpha[m] = - (a_m - lam) / b_m
                elif (a_m < - lam):
                    alpha[m] = - (a_m + lam) / b_m
                else:
                    alpha[m] = 0.0

    # 固定Alpha，优化B
    AlphaT = Alpha.I
    AA_T = np.dot(Alpha, AlphaT)
    det = np.linalg.det(AA_T)
    for t in range(0, 10):
        if(det == 0):
            # 不可逆
            I = np.identity(k)
            AA_T_I_T = AA_T + my_lambda * I
            B = np.dot(np.dot(X, AlphaT), AA_T_I_T.I)
        else:
            # 可逆
            B = np.dot(np.dot(X, AlphaT), AA_T.I)

    # 一次优化完成
    zero_num = sum(Alpha == 0)
    zero_num = np.sum(zero_num)
    loss = loss_function(X, B, Alpha, my_lambda)
    loss_list.append(loss)
    print("第%d次优化, 损失：%f, Alpha中0个数：%d" % (i, loss, zero_num))
    if(zero_num != 0):
        write_Alpha_txt(i + 1, Alpha, zero_num)
        write_B_txt(i + 1, B)
    elif((i+1) % 50 == 0):
        write_Alpha_txt(i + 1, Alpha, zero_num)

write_B_txt(i + 1, B)
end = time.time()
print("程序运行时间：", end-start)
now_time = datetime.datetime.now()
print(now_time)
plot_loss(loss_list)
