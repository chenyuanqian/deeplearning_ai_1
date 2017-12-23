# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 23:15:49 2017

@author: yuan
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v3 import *
from dnn_utils_v2 import sigmoid,sigmoid_backward,relu,relu_backward

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))
    
    assert(W1.shape==(n_h,n_x))
    assert(b1.shape==(n_h,1))
    assert(W2.shape==(n_y,n_h))
    assert(b2.shape==(n_y,1))
    
    parameters={"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2}

    return parameters

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters={}
    L=len(layer_dims)
    
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters["b"+str(l)]=np.zeros((layer_dims[l],1))
        
    return parameters

def linear_forward(A,W,b):
    Z=W.dot(A)+b
    cache=(A,W,b)

    return Z,cache
        
def linear_activation_forward(A_prev,W,b,activation):
    Z,linear_cache=linear_forward(A_prev,W,b)
    if activation=="sigmoid":
        A,activation_cache=sigmoid(Z)
        cache=(linear_cache,activation_cache)
    elif activation=="relu":
        A,activation_cache=relu(Z)
        cache=(linear_cache,activation_cache)
    return A,cache

def L_model_forward(X,parameters):
    caches=[]
    A=X
    L=len(parameters)/2
    for l in range(1,L):
        A_prev=A
        A,cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],'relu')
        caches.append(cache)
    AL,cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],'sigmoid')
    caches.append(cache)
    return AL,caches  

def compute_cost(AL,Y):
    m=Y.shape[1]
    logprobs=np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),(1-Y))
    cost=-np.sum(logprobs)/m
#    cost=-1*((Y.dot(np.log(A2).T))+(1-Y).dot(np.log(1-A2).T))/m
    cost = np.squeeze(cost)
    return cost  

def linear_backward(dZ,cache):
    A_prev,W,b=cache
    m=A_prev.shape[1]
    
    dW=dZ.dot(A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=W.T.dot(dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation=="relu":
        dZ=relu_backward(dA, activation_cache)
        dA_prev, dW, db=linear_backward(dZ, linear_cache)
    elif activation=="sigmoid":
        dZ=sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db=linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

if __name__=="__main__":
    plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    #3.1 - 2-layer Neural Networ
#    parameters = initialize_parameters(3,2,1)
#    print("W1 = " + str(parameters["W1"]))
#    print("b1 = " + str(parameters["b1"]))
#    print("W2 = " + str(parameters["W2"]))
#    print("b2 = " + str(parameters["b2"]))
#    
#    #3.2 - L-layer Neural Network
#    parameters = initialize_parameters_deep([5,4,3])
#    print("W1 = " + str(parameters["W1"]))
#    print("b1 = " + str(parameters["b1"]))
#    print("W2 = " + str(parameters["W2"]))
#    print("b2 = " + str(parameters["b2"]))
#    #4.1 - Linear Forward
#    A, W, b = linear_forward_test_case()
#    Z, linear_cache = linear_forward(A, W, b)
#    print("Z = " + str(Z))
    #4.2 - Linear-Activation Forward
#    A_prev, W, b = linear_activation_forward_test_case()
#    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
#    print("With sigmoid: A = " + str(A))
#    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
#    print("With ReLU: A = " + str(A))
    #d) L-Layer Model
    X, parameters = L_model_forward_test_case_2hidden()
    AL, caches = L_model_forward(X, parameters)
    print("AL = " + str(AL))
    print("Length of caches list = " + str(len(caches)))
    
    #5 - Cost function
    Y, AL = compute_cost_test_case()

    print("cost = " + str(compute_cost(AL, Y)))
    #6.1 - Linear backward
    dZ, linear_cache = linear_backward_test_case()

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))
    
    #6.2 - Linear-Activation backward
    AL, linear_activation_cache = linear_activation_backward_test_case()

    dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
    print ("sigmoid:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db) + "\n")
    
    dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
    print ("relu:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))
    