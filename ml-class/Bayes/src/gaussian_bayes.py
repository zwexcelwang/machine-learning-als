import math
import numpy as np

def naive_gaussian(mean=0, std=1, x=1):
    # return np.exp(-np.square(x - mean) / (2 * np.square(std))) / (np.sqrt(2 * math.pi) * std)
    return math.exp(-math.pow(x - mean, 2) / (2 * math.pow(std, 2))) / (np.sqrt(2 * math.pi) * std)

def gaussian(mean, cov, x):
    det = np.linalg.det(cov)
    a = 2 * math.pi * math.sqrt(det)
    tmp = x - mean
    tmp = np.array([tmp])
    b = - np.dot(np.dot(tmp, np.linalg.pinv(cov)), tmp.T) / 2
    return math.exp(b) / a

def normalization(data):
    sum = np.sum(data)
    return data / sum

# Naive Gaussian Bayes
def naive_gaussian_bayes(good_melon, bad_melon):
    good_mean = np.mean(good_melon, axis=1)
    bad_mean = np.mean(bad_melon, axis=1)
    # print(good_mean, bad_mean)
    good_std = np.std(good_melon, axis=1)
    bad_std = np.std(bad_melon, axis=1)
    # print(good_std, bad_std)
    good_de = naive_gaussian(good_mean[0], good_std[0], 0.5)
    good_su = naive_gaussian(good_mean[1], good_std[1], 0.3)
    good_pro = 8 / 17 * good_de * good_su
    bad_de = naive_gaussian(bad_mean[0], bad_std[0], 0.5)
    bad_su = naive_gaussian(bad_mean[1], bad_std[1], 0.3)
    bad_pro = 9 / 17 * bad_de * bad_su
    pro = normalization(np.array([good_pro, bad_pro]))
    return pro

# Gaussian Bayes
def gaussian_bayes(good_melon, bad_melon):
    good_cov = np.cov(good_melon)
    bad_cov = np.cov(bad_melon)
    good_mean = np.mean(good_melon, axis=1)
    bad_mean = np.mean(bad_melon, axis=1)
    x = np.array([0.5, 0.3])
    good_pro = 8 / 17 * gaussian(good_mean, good_cov, x)
    bad_pro = 9 / 17 * gaussian(bad_mean, bad_cov, x)
    pro = normalization(np.array([good_pro, bad_pro]))
    return pro

good_melon = np.array([[0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437],
                     [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211]])
bad_melon = np.array([[0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719],
                     [0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103]])

navie_pro = naive_gaussian_bayes(good_melon, bad_melon)
pro = gaussian_bayes(good_melon, bad_melon)
print(navie_pro, pro)




