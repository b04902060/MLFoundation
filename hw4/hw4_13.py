import numpy as np
import pandas as pd
from numpy.linalg import inv
import sys
import matplotlib.pyplot as plt

def loadData(trainfile, testfile):
    data = pd.read_csv(trainfile, sep='\s+', header=None)
    data = data.as_matrix()
    col, row = data.shape
    X_train = np.c_[np.ones((col,1)), data[:, 0:-1]]
    Y_train = data[:, -1]
    Y_train = np.array(Y_train)

    data = pd.read_csv(testfile, sep='\s+', header=None)
    data = data.as_matrix()
    col, row = data.shape
    X_test = np.c_[np.ones((col,1)), data[:, 0:-1]]
    Y_test = data[:, -1]
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test



def ridge_regression(X, Y, l):
    row, col = X.shape
    #w = inv(np.matmul(X.transpose(),X) - l*np.identity(col)).dot( np.matmul(X.transpose(), Y) )
    w = np.matmul(inv(np.matmul(X.transpose(),X) + l*np.identity(col)),  np.matmul(X.transpose(), Y) )

    return w

def compute_error(theta, X, Y):
    Y_pred = X.dot(theta)
    Y_pred[Y_pred > 0] = 1
    Y_pred[Y_pred <= 0 ] = -1

    err = np.sum(Y_pred != Y)/float(len(Y))
    return err


def main():
    Ein_plot = []
    Eout_plot = []

    for i in range(2, -11, -1):
        lada = 10**i
        X_train, Y_train, X_test, Y_test = loadData(sys.argv[1], sys.argv[2])
    
        theta = ridge_regression(X_train, Y_train, lada)

        Ein = compute_error(theta, X_train, Y_train)
        Eout = compute_error(theta, X_test, Y_test)
        Ein_plot.append(Ein)
        Eout_plot.append(Eout)

        print "lambda = " + str(lada)
        print "Ein: "  + str(Ein)
        print "Eout: " + str(Eout)
        print "=================================="

    plot1 = plt.plot(range(2, -11, -1), Ein_plot, '-o', label='Ein')
    plot2 = plt.plot(range(2, -11, -1), Eout_plot, '-o', label='Eout')
    plt.xlabel('log lambda')# make axis labels
    plt.ylabel('error rate')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()