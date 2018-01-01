import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

ITERATION = 2000
L_RATE = 0.01
SGD = 0

def loadData(trainfile, testfile):
    data = pd.read_csv(trainfile, sep='\s+', header=None)
    data = data.as_matrix()
    col, row = data.shape
    X_train = np.c_[np.ones((col,1)), data[:, 0:-1]]
    Y_train = data[:, -1]

    data = pd.read_csv(testfile, sep='\s+', header=None)
    data = data.as_matrix()
    col, row = data.shape
    X_test = np.c_[np.ones((col,1)), data[:, 0:-1]]
    Y_test = data[:, -1]
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8)) 

def logistic(X_train, Y_train, X_test, Y_test, iter=1000, l_rate=0.01, SGD=1, plot=1):
    row, col = X_train.shape
    theta = np.zeros((1,col))
    plot_Ein = []
    plot_Eout = []
    num = 0
    for i in range(iter):
        if SGD == 0:
            gradient = np.zeros((1,col))
            for j in range(row):
                gradient = gradient + np.transpose(sigmoid(-1*Y_train[j]*theta.dot(X_train[j,:]))*(-1)*Y_train[j]*X_train[j,:])
            gradient = gradient/row
            theta = theta-l_rate*gradient
        else:
            if num >= row:
                num = 0
            gradient = np.transpose(sigmoid(-1*Y_train[num]*theta.dot(X_train[num,:]))*(-1)*Y_train[num]*X_train[num,:])
            theta = theta-l_rate*gradient
            num = num + 1
        if plot==1:
            if i%50 == 0:           
                plot_Ein.append(test(X_train, Y_train, np.transpose(theta)))
                plot_Eout.append(test(X_test, Y_test, np.transpose(theta)))
    if plot == 1:
        if SGD == 0:
            plot1 = plt.plot(range(0, iter, 50), plot_Ein, '-o', label='Ein norm')
            #plot2 = plt.plot(range(0, iter, 50), plot_Eout, '-o', label='Eout norm')
        else:
            plot1 = plt.plot(range(0, iter, 50), plot_Ein, '-o', label='Ein SGD')
            #plot2 = plt.plot(range(0, iter, 50), plot_Eout, '-o', label='Eout SGD')
        plt.legend()
    return np.transpose(theta)

def test(X, Y, theta):
    Y = Y.reshape((len(Y),1))
    y_pred = X.dot(theta)
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0 ] = -1

    err = np.sum(y_pred != Y)/float(len(Y))

    return err

def main():
    X_train, Y_train, X_test, Y_test = loadData(sys.argv[1], sys.argv[2])
    theta = logistic(X_train, Y_train, X_test, Y_test, ITERATION, L_RATE, SGD)
    theta = logistic(X_train, Y_train, X_test, Y_test, ITERATION, L_RATE, 1)

    print "Eout: "+ str(test(X_test, Y_test, theta))
    plt.title('l_rate:'+str(L_RATE)+', iter:'+str(ITERATION))
    plt.xlabel('iteration')# make axis labels
    plt.ylabel('error rate')
    plt.show()

if __name__ == '__main__':
    main() 