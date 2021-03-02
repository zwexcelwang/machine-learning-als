import numpy as np

def do_page_rank(pt, d, N, L):
    '''
    需要知道d,L,从L可以得到Dc(-1), B=(1-d)*e*e^T/N + d*L*Dc(-1)
    N是网页数, e是N长的列向量
    :return:
    '''
    e = np.ones((N, 1))
    B1 = (1-d) * e * e.T / N
    print(B1)
    c = np.sum(L, axis=0)
    Dc = np.diag(c)  # 根据输入生成对角矩阵
    print(Dc)
    Dc_1 = np.linalg.pinv(Dc)
    B2 = d * np.dot(L, Dc_1)
    print(B2)
    B = B1 + B2
    # pt = np.array([[1], [1], [1], [1]])  # 要使N个p的重要性的平均数是1
    is_unstable = True
    i = 1
    while is_unstable:
        p = np.dot(B, pt)
        print("这是第%d次迭代, 网页的重要性为%s" % (i, p.T))
        if (p == pt).all():
            is_unstable = False
        pt = p
        i = i + 1


if __name__ == "__main__":
    # d = 0.85
    # N = 4
    # pt = np.array([[1], [1], [1], [1]])
    # L = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]])
    # do_page_rank(pt, d, N, L)
    a = np.array([1, 3])

