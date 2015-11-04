from numpy import *
import numpy as np
import pylab as pl
import operator

trainingSampleNum = 5000
testingSampleNum = 1000

def knn(xtrain, ytrain, xtest, k):
    #xtrain, xtest have shape (2, number of data)
    dataSetSize = xtrain.shape[0]   # shape[0] stands for the num of row
    # calculate the Euclidean distance
    diffMat = tile(xtest, (dataSetSize,1)) - xtrain
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()    # return the indices that would sort an array in a ascending order

    classCount = {}
    for i in range(k):
        votelabel = ytrain[sortedDistIndicies[i]]   # choose the min k distance
        classCount[votelabel] = classCount.get(votelabel,0) + 1

    # the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            ytest = key

    #ytest are the predicted labels for xtest
    return ytest

def dataCreate(num, scale):        # create training data
    x1 = np.random.uniform(-1*scale,1*scale,num)
    x2 = np.random.uniform(-1,1,num)
    X = []
    Y = []

    for i in range(0,num):
        p1 = np.random.uniform(0,1,1)
        p2 = np.random.uniform(0,1,1)
        X.append([x1[i],x2[i]])
        if x2[i] >= 0:
            if p1 <= 0.75:
                Y.append((X[i],1))
            else:
                Y.append((X[i],-1))
        else:
            if p2 <= 0.75:
                Y.append((X[i],-1))
            else:
                Y.append((X[i],1))
    return array(X), array(Y)

def showplot(x1, y1, x2, y2,type):  # show the plot 
    if type == 1:
        pl.title("Training Point")
    elif type == 2:
        pl.title("Testing Point")
    elif type == 3:
        pl.title("Testing Point with deviation")
    pl.plot(x1,y1,'o')
    pl.plot(x2,y2,'or')
    pl.xlabel('x1 axis')
    pl.ylabel('x2 axis')
    pl.show()

def errorCalc(ytest,ytest_knn):  # calculate the zero-one error of the data
    dataSetSize = len(ytest)
    error = 0
    for i in range(dataSetSize):
        if ytest[i] != ytest_knn[i]:
            error += 1
    return float(error)/float(dataSetSize)

def standardDeviation(X):    # calculate the standard deviations of X 
    std1 = X[:,0].std()
    std2 = X[:,1].std()
    return std1, std2

def divideByStandardDeviation(x,std1,std2):  # devide X by the standard deviations of two dimension
    xDivide = []
    for i in range(x.shape[0]):
        xDivide.append([float(x[i][0])/std1,float(x[i][1])/std2])
    return np.array(xDivide)


def runKnn(scale, k):          # scale is the scale of the first component, k is the neighbor
    X, Y = dataCreate(trainingSampleNum, scale)
    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    labels = []

    for (value1, value2), value3 in Y:
        labels.append(value3)
        if value3 == 1:
            tmp1.append(value1)
            tmp2.append(value2)
        else:
            tmp3.append(value1)
            tmp4.append(value2)

    showplot(tmp1, tmp2, tmp3, tmp4 ,1)
    tmp1 = []    # clear the list tmp1, tmp2, tmp3, tmp4
    tmp2 = []
    tmp3 = []
    tmp4 = []

    xtest, ytest = dataCreate(testingSampleNum, scale)
    yArray = []
    for (value1, value2), value3 in ytest:
        yArray.append(value3)

    yTestArray = []
    for [x, y] in xtest:
        label = knn(X, labels, [x,y] , k)
        yTestArray.append(label)
        if label == 1:
            tmp1.append(x)
            tmp2.append(y)
        else:
            tmp3.append(x)
            tmp4.append(y)

    showplot(tmp1, tmp2, tmp3, tmp4, 2)
    errorScore = errorCalc(yArray,yTestArray)
    print "The error score is: ",errorScore
    std1, std2 = standardDeviation(X)
    print "The first dimension standard deviation: ",std1
    print "The second dimension standard deviation: ",std2

    std3, std4 = standardDeviation(xtest)
    XDevideByStd = divideByStandardDeviation(X, std1, std2)
    xtestDevideByStd = divideByStandardDeviation(xtest, std3, std4)
    std1, std2 = standardDeviation(XDevideByStd)
    print "The first dimension standard deviation: ",std1
    print "The second dimension standard deviation: ",std2
    std3, std4 = standardDeviation(xtestDevideByStd)
    print "The first dimension standard deviation: ",std3
    print "The second dimension standard deviation: ",std4

    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    del yTestArray[:]
    for [x, y] in xtestDevideByStd:
        label = knn(XDevideByStd, labels, [x,y] , k)
        yTestArray.append(label)
        if label == 1:
            tmp1.append(x)
            tmp2.append(y)
        else:
            tmp3.append(x)
            tmp4.append(y)

    showplot(tmp1, tmp2, tmp3, tmp4, 3)
    errorScore = errorCalc(yArray,yTestArray)
    print "The error score is: ",errorScore


print "Run The First Time"
runKnn(1, 100)
print ""
print "Run The Second Time"
runKnn(1000, 100)
