import os
import re
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
                            ).reshape((int(height)*int(width),))
                            # ).reshape((int(height), int(width)))

def get_original_ALL(p, n, path="../orl_faces/"):
    X = np.zeros((p, n))
    dir_list = os.listdir(path)
    for i in range(0, len(dir_list)):
        dir_path = path + dir_list[i]
        file_list = os.listdir(dir_path)
        for j in range(0, len(file_list)):
            file_path = dir_path + '/' + file_list[j]
            image = read_pgm(file_path, byteorder='<')
            x_col = 10 * i + j
            X[:, x_col] = image
    return X

def get_original_X(p, n, dir=1):
    path = "../orl_faces/s" + str(dir)
    X = np.zeros((p, n))
    file_list = os.listdir(path)
    for j in range(0, len(file_list)):
        file_path = path + '/' + file_list[j]
        image = read_pgm(file_path, byteorder='<')
        x_col = j
        X[:, x_col] = image
    return X


def show_img(img):
    path = "../orl_faces/s" + img
    image = read_pgm(path, byteorder='<').reshape(112, 92)
    plt.imshow(image, plt.cm.gray)
    plt.savefig("../IMG/s10_SRC_9.png")


if __name__ == "__main__":
    img = "10/9.pgm"
    show_img(img)
