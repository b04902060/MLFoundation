import numpy as np
import pandas as pd
import sys
import random
from numpy.linalg import inv
import matplotlib.pyplot as plt

def generateData(num=1000, tansform=1):
    x1 = np.random.uniform(-1, 1, num)
    x2 = np.random.uniform(-1, 1, num)
    X = np.c_[np.ones((num,1)), x1, x2]
    if tansform == 1:
        X = np.c_[X, x1*x2, x1*x1, x2*x2]
    Y = np.sign(np.power(x1, 2)+np.power(x2, 2)-0.6)
    Y[Y == 0] = -1
    for i in range(num):
        if random.random() < 0.1:
            Y[i]*= -1
    Y = Y.reshape((num, 1))
    return X, Y



def main():

    totalerr = 0
    Eout_list = []
    for i in range(1000):
        x,y = generateData()
        theta = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)
        ypred = np.sign(x.dot(theta))
        err = np.sum(ypred!=y)/1000.0
        Eout_list.append(err)
        totalerr += err
    print('Ein: ', totalerr/1000.0)
    plt.hist(Eout_list)
    plt.show()


if __name__ == '__main__':
    main() 