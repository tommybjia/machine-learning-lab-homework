import numpy as np
import operator
from scipy import *
import pylab as pl

dataNum = 100   # the sample number

def dataCreate(num):  # create the data used in lab
	x1 = np.random.normal(1, 1, num)
	x2 = np.random.normal(3, 1, num)
	X = np.zeros((2,num))
	X[0,:] = x1
	X[1,:] = x2
    return X  # X is an array in numpy

def showPlot(X):  # use this function to show plot
	pl.plot(X[0,:],X[1,:],'.')
	pl.show()
	

X = dataCreate(dataNum)
showPlot(X)  # the orignal data

A1 = np.mat('-1 0; 0 1')  # construct A1
A1X = np.dot(A1, X)
showPlot(A1X)

A2 = np.mat('0.5 0; 0 1') # construct A2
A2X = np.dot(A2, X)
showPlot(A2X)

A3 = np.mat([[np.sqrt(2)/2, np.sqrt(2)/2], [np.sqrt(2)/-2, np.sqrt(2)/2]]) # construct A3

A3X = np.dot(A3, X)
showPlot(A3X)

A4 = np.mat('1 0; 0 -1')  # construct A4


A5 = np.dot(np.dot(A2, A1), A4)  # # construct A5

A5X = np.dot(A5, X)
showPlot(A5X)
