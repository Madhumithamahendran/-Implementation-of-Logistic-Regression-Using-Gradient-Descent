# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
Program to implement the the Logistic Regression Using Gradient Descent.

Developed by:MADHUMITHA M 

Register Number:212222220020  

import numpy as np

import matplotlib.pyplot as plt

from scipy import optimize

df=np.loadtxt("/content/ex2data1.txt",delimiter=',')

X=df[:,[0,1]]

y=df[:,2]

X[:5]

y[:5]

plt.figure()

plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")

plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")

plt.xlabel("Exam 1 score")

plt.ylabel("Exam 2 score")

plt.show()

def sigmoid(z):

  return 1/(1+np.exp(-z))

plt.plot()

X_plot = np.linspace(-10,10,100)

plt.plot(X_plot,sigmoid(X_plot))

plt.show()

def costFunction(theta,X,y):

  h=sigmoid(np.dot(X,theta))
  
  J=-(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  
  grad=np.dot(X.T,h-y)/X.shape[0]
  
  return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))

theta=np.array([0,0,0])

J,grad=costFunction(theta,X_train,y)

print(J)

print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))

theta=np.array([-24,0.2,0.2])

J,grad=costFunction(theta,X_train,y)

print(J)

print(grad)

def cost(theta,X,y):

  h=sigmoid(np.dot(X,theta))
  
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  
  return J

def gradient(theta,X,y):

  h=sigmoid(np.dot(X,theta))
  
  grad=np.dot(X.T,h-y) / X.shape[0]
  
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))

theta = np.array([0,0,0])

res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),
                        method='Newton-CG',jac=gradient)
                        
print(res.fun)

print(res.x)

def plotDecisionBoundary(theta,X,y):

  x_min,x_max=X[:,0].min() - 1,X[:,0].max()+1
  
  y_min,y_max=X[:,1].min() - 1,X[:,0].max()+1
  
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))
                    

  X_plot = np.c_[xx.ravel(),yy.ravel()]
  
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label='admitted')
  
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label='NOT admitted')
  
  plt.contour(xx,yy,y_plot,levels=[0])
  
  plt.xlabel("Exam 1 score")
  
  plt.ylabel("Exam 2 score")
  
  plt.legend()
  
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))

print(prob)

def predict(theta,X):

  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  
  prob = sigmoid(np.dot(X_train,theta))
  
  return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)

## Output:
ARRAY VALUE OF X:

![image](https://github.com/Madhumithamahendran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394403/2f262671-dc80-44d6-94b1-f790db387e5a)

ARRAY VALUE OF Y:

![image](https://github.com/Madhumithamahendran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394403/0d0dfba2-b8f4-4f9a-9a1e-645f57d33170)

EXAM 1 - SCORE GRAPH:

![image](https://github.com/Madhumithamahendran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394403/8163dedd-a1ed-47af-9cfa-ff9fbef4fdfe)

SIGMOID FUNCTION GRAPH

![image](https://github.com/Madhumithamahendran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394403/1921f6be-33eb-4302-aef3-53aa563a5323)

X TRAIN GRADE VALUE

![image](https://github.com/Madhumithamahendran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394403/49c92b9f-b636-4db7-8507-dbe3f9e1dcd9)

Y TRAIN GRADE VALUE

![image](https://github.com/Madhumithamahendran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394403/d9ea723b-c13f-4b57-9d96-13f016ff7f9f)

PRINT RES.X

![image](https://github.com/Madhumithamahendran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394403/5983854f-7fd4-42c8-a9b8-36c04d0ed944)

DECISION BOUNDARY - GRAPH FOR EXAM SCORE

![image](https://github.com/Madhumithamahendran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394403/b2c5a417-875f-494a-a9b3-f3f208425dba)

Probability value

![image](https://github.com/Madhumithamahendran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394403/e494aaae-a7d7-4bdd-ba01-f142dd975032)

Prediction value of mean

![image](https://github.com/Madhumithamahendran/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119394403/33f8b0bd-d67b-4a6c-aa74-7799524c76c3)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

