import numpy as np
import sys
import matplotlib.pyplot as plt


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
    iteration = 2000
    total_step = 0
    
    step_array = np.zeros(100, dtype=np.int)

    for i in range(iteration):
        X, Y = _shuffle(X,Y)
        step = pla(X,Y)
        step_array[step] = step_array[step]+1 
        print(str(i)+':'+str(step))
        total_step = total_step + step
    print (step_array)
    print (total_step/iteration)



    plt.xlabel('numbers of step', fontsize=14)
    plt.ylabel('times', fontsize=14)
    #plt.xticks(np.linspace(0, 300, 300))
    plt.bar(np.arange(100),step_array)
    plt.show()




if __name__ == '__main__':
    main()