# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 22:58:49 2017

@author: yuan
"""

import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import  plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

def layer_sizes(X,Y):
    n_x=X.shape[0]
    n_h=4
    n_y=2
    
    return(n_x,n_h,n_y)
    
def initialize_parameters(n_x,n_h,n_y):
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)
    b2=np.zeros((n_y,1))
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters={"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2}
    
    return parameters

def forward_propagation(X,parameters):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]    
    Z1=W1.dot(X)+b1
    A1=sigmoid(Z1)
    Z2=W2.dot(A1)+b2
    A2=sigmoid(Z2)
        
    assert(A2.shape==(1,X.shape[1]))
        
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
        
        
        
        


if __name__=="__main__":
    #2 - Dataset
    X,Y=load_planar_dataset()
    # Visualize the data:
    plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);
    
    shape_X=X.shape
    shape_Y=Y.shape
    m=shape_X[1]
    
    print ('The shape of X is: ' + str(shape_X))
    print ('The shape of Y is: ' + str(shape_Y))
    print ('I have m = %d training examples!' % (m))
    
    #3 - Simple Logistic Regression
    clf=sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T,Y.T)
    LR_predictions=clf.predict(X.T)
    print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")
    
    #4.1 - Defining the neural network structure
    X_assess, Y_assess = layer_sizes_test_case()
    (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
    print("The size of the input layer is: n_x = " + str(n_x))
    print("The size of the hidden layer is: n_h = " + str(n_h))
    print("The size of the output layer is: n_y = " + str(n_y))
    
    n_x, n_h, n_y = initialize_parameters_test_case()
    #4.2 - Initialize the model's parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    #4.3 - The Loop
    X_assess, parameters = forward_propagation_test_case()
    A2, cache = forward_propagation(X_assess, parameters)

    print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))