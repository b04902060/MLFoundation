import numpy as np
import sys


def read_file(pathname): 
    file = open(pathname)
    X = []
    Y = []
    for line in file:
        tmp = [1]
        tmp.extend(line.split()[0:4])
        X.append(tmp)
        Y.append(line.split()[4:5])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    return X, Y

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def pla(X, Y):

    w = np.zeros(5)

    index = 0
    correct = 0
    isfinished = 0 
    step = 0 # to record the time that w has updated
    while(not isfinished):
        if sum(X[index]*w)*Y[index] <= 0: # wrong one
            w = w + Y[index]*X[index]
            correct = 0
            step  = step+1
        else: # correct
            correct = correct+1
        index = index+1
        if index == len(X)-1:
            index = 0
        if correct == len(X):
            isfinished = 1
    return step


def main():
    X, Y = read_file(sys.argv[1])

    total_step = 0
    for i in range(2000):
        X, Y = _shuffle(X,Y)
        step = pla(X,Y)
        print(str(i)+':'+str(step))
        total_step = total_step + step

    print (total_step/2000)



if __name__ == '__main__':
    main()