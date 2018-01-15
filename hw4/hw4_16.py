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


    X_val = X_train[120:]
    Y_val = Y_train[120:]
    X_train = X_train[0:120]
    Y_train = Y_train[0:120]

    data = pd.read_csv(testfile, sep='\s+', header=None)
    data = data.as_matrix()
    col, row = data.shape
    X_test = np.c_[np.ones((col,1)), data[:, 0:-1]]
    Y_test = data[:, -1]
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test



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
    Eval_plot = []

    for i in range(2, -11, -1):
        lada = 10**i
        X_train, Y_train, X_val, Y_val, X_test, Y_test = loadData(sys.argv[1], sys.argv[2])
    
        theta = ridge_regression(X_train, Y_train, lada)

        Ein = compute_error(theta, X_train, Y_train)
        Eval = compute_error(theta, X_val, Y_val)
        Ein_plot.append(Ein)
        Eval_plot.append(Eval)
    
        print "lambda = " + str(lada)
        print "Ein: "  + str(compute_error(theta, X_train, Y_train))
        print "Eval: " + str(compute_error(theta, X_val, Y_val))
        print "Eout: " + str(compute_error(theta, X_test, Y_test))
        print "=================================="

    plot1 = plt.plot(range(2, -11, -1), Ein_plot, '-o', label='Ein')
    plot2 = plt.plot(range(2, -11, -1), Eval_plot, '-o', label='Eval')
    plt.xlabel('log lambda')# make axis labels
    plt.ylabel('error rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()