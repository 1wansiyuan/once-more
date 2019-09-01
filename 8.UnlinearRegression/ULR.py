import numpy as np
import random

def genData(numPoints,bias,variance):
    x = np.zeros(shape=(numPoints,2))
    y = np.zeros(shape=(numPoints))
    for i in range(0,numPoints):
        x[i][0]=1
        x[i][1]=i
        y[i]=(i+bias)+random.uniform(0,1)+variance
    return x,y

def gradientDescent(x,y,theta,alpha,m,numIterations):
    xTran = np.transpose(x)
    for i in range(numIterations):
        hypothesis = np.dot(x,theta)
        loss = hypothesis-y#损失函数（Loss Function ）是定义在单个样本上的，算的是一个样本的误差。
        cost = np.sum(loss**2)/(2*m)#代价函数（Cost Function ）是定义在整个训练集上的，是所有样本误差的平均，也就是损失函数的平均。
        gradient=np.dot(xTran,loss)/m#对代价函数求导
        theta = theta-alpha*gradient
        print ("Iteration %d | cost :%f" %(i,cost))
    return theta
# a = np.array([[1, 2, 3],[2, 3, 4]])
# b = np.array([1, 2, 3])
# c = np.dot(a,b)
# print(c)
x,y = genData(100, 25, 10)
print ("x:")
print (x)
print ("y:")
print (y)

m,n = np.shape(x)
n_y = np.shape(y)

print("m:"+str(m)+" n:"+str(n)+" n_y:"+str(n_y))

numIterations = 100000
alpha = 0.0005
theta = np.ones(n)
theta= gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)