# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:20:22 2017

@author: yuan
"""

import numpy as np

def basic_sigmod(x):
    s=1/(1+np.exp(-x))
    return s

def sigmod_derivative(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)
    return ds

def image2vector(image):
    v=image.reshape((image.shape[0]*image.shape[1],image.shape[2]))
    return v

def normalizeRows(x):
    xs=np.linalg.norm(x,axis=1,keepdims=True)
    x_normalized=x/xs
    return x_normalized

def softmax(x):
    #s=exp(x)
    s=np.exp(x)
    p=np.ones(x.shape[1])
    #求s一行的和
    m=s.dot(p)
    #第二种求和方法
    m=np.array([s[0].sum(),s[1].sum()])
    m1=1/m
    #扩展p为2*5矩阵，方便求s*m2.T
    m2=np.tile(m1,(x.shape[1],1)) 
    n=s*m2.T
    return n

def L1(yhat,y):
    l1=np.abs(yhat-y).sum()
    return l1

def L2(yhat,y):
    l2=np.square(yhat-y).sum()
    return l2
    

if __name__=="__main__":
    print basic_sigmod(3)
    
    x=np.array([1,2,3])
    #print(np.exp(x))
    #print(x+3)
    print sigmod_derivative(x)
    
    image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],
       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],
       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
    
    print image2vector(image)
    
    x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
    
    print normalizeRows(x)
    
    x2 = np.array([
    [9, 2, 5, 0, 0,1],
    [7, 5, 0, 0 ,0,2]])
    
    print softmax(x2)
    
    yhat = np.array([0.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L1 = " + str(L1(yhat,y)))
    print("L2 = " + str(L2(yhat,y)))
    
    ss=x2.reshape(2,1,-1)
    print ss