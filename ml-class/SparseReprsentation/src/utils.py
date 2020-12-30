import re
import os
import random
import numpy as np
import matplotlib.pyplot as plt


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            # ).reshape((int(height)*int(width),))
                            ).reshape((int(height), int(width)))

def write_Alpha_txt(i, Alpha, zero_num):
    path = "../Alpha_B/"
    A_file = str(i) + "_Alpha"
    A_filepath = path + A_file + ".txt"
    # zero_num = sum(Alpha == 0)     # Alpha中0的个数
    # zero_num = np.sum(zero_num, axis=1)
    str1 = "Zero_Num in Alpha：" + str(zero_num)
    # print(i, str1)
    np.savetxt(A_filepath, Alpha, header=str1, encoding="utf-8")

def write_B_txt(i, B):
    path = "../Alpha_B/"
    B_file = str(i) + "_B"
    B_filepath = path + B_file + ".txt"
    np.savetxt(B_filepath, B)

def get_original_x(path, p, n, k):
    b_col = 0
    X = np.zeros((p, n))
    B = np.zeros((p, k))
    dir_list = os.listdir(path)
    for i in range(0, len(dir_list)):
        dir_path = path + dir_list[i]
        file_list = os.listdir(dir_path)
        random_file_num = random.randint(0, len(file_list) - 1)
        for j in range(0, len(file_list)):
            file_path = dir_path + '/' + file_list[j]
            image = read_pgm(file_path, byteorder='<')
            x_col = 10 * i + j
            X[:, x_col] = image
            if (j == random_file_num):
                B[:, b_col] = image
                # print(image)
                b_col = b_col + 1
    return X, B

def loss_function(X, B, A, lamda):
    BA = np.dot(B, A)
    X_BA = X - BA
    loss = np.sum(np.square(X_BA))
    loss = loss + lamda * np.sum(A)
    return loss

def plot_loss(loss_list):
    time = np.arange(0, len(loss_list))
    plt.plot(time, loss_list)
    plt.title('loss')
    plt.savefig("loss.png")
    plt.show()

if __name__ == "__main__":
    import numpy as np
    import codecs
    image = read_pgm("../orl_faces/s1/1.pgm", byteorder='<')
    filecp = codecs.open("../Alpha_B/100_Alpha.txt", encoding='utf-8')
    A = np.loadtxt(filecp)
    # A = np.loadtxt()
    B = np.loadtxt("../Alpha_B/200_B.txt")
    # plt.imshow(image, plt.cm.gray)
    # plt.savefig("ini_x.png")de
    # plt.show()
    # print(B.shape)
    after_sparse = np.dot(B, A[:, 0])
    after_sparse = after_sparse.reshape(112, 92)
    print(after_sparse.shape)
    plt.imshow(after_sparse, plt.cm.gray)
    plt.savefig("aft_spa.png")
    plt.show()


