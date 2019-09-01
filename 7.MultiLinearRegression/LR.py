from numpy import genfromtxt
import numpy as np
from sklearn import linear_model

dataPath = r"F:\Python\7.MultiLinearRegression\Delivery.csv"
deliveryData = genfromtxt(dataPath,delimiter=',')

print ("data")
print (deliveryData)

x= deliveryData[:,:-1]
y = deliveryData[:,-1]

print (x)
print (y)

lr = linear_model.LinearRegression()
lr.fit(x, y)

print (lr)

print("coefficients:")
print (lr.coef_)

print("intercept:")
print (lr.intercept_)

xPredict = np.array([102,6]).reshape(1, -1)
yPredict = lr.predict(xPredict)
print("predict:")
print (yPredict)
