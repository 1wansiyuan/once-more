from numpy import genfromtxt
from sklearn import linear_model
import  numpy as np
import csv
datapath=open(r"F:\Python\7.MultiLinearRegression\Delivery_Dummy.csv",encoding = 'utf-8')
data = genfromtxt(datapath,delimiter=",")

x = data[1:,:-1]
y = data[1:,-1]
print (x)
print (y)

mlr = linear_model.LinearRegression()

mlr.fit(x, y)

print (mlr)
print ("coef:")
print (mlr.coef_)
print ("intercept")
print (mlr.intercept_)

xPredict =  np.array([90,2,0,0,1]).reshape(1, -1)
yPredict = mlr.predict(xPredict)

print ("predict:")
print (yPredict)