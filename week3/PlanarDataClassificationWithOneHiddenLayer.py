# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 22:58:49 2017

@author: yuan
"""

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import  plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

def layer_sizes(X,Y):
    n_x=X.shape[0]
    n_h=4
    n_y=Y.shape[0]
    
    return(n_x,n_h,n_y)
    
def initialize_parameters(n_x,n_h,n_y):
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
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
    A1=np.tanh(Z1)
    Z2=W2.dot(A1)+b2
    A2=sigmoid(Z2)
        
   # assert(A2.shape==(1,X.shape[1]))
        
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
        
def compute_cost(A2,Y,parameters):
    m=Y.shape[1]
    logprobs=np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),(1-Y))
    cost=-np.sum(logprobs)/m
#    cost=-1*((Y.dot(np.log(A2).T))+(1-Y).dot(np.log(1-A2).T))/m
    cost = np.squeeze(cost)
    return cost

def backward_propagation(parameters,cache,X,Y):
    m=X.shape[1]
    
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]    
    Z1=W1.dot(X)+b1
    A1=np.tanh(Z1)
    Z2=W2.dot(A1)+b2
    A2=sigmoid(Z2)
    
    dZ2=A2-Y
    dW2=dZ2.dot(A1.T)/m
    db2=np.sum(dZ2,axis=1,keepdims=True)/m
    
    dZ1=W2.T.dot(dZ2)*(1-np.power(A1,2))
    dW1=dZ1.dot(X.T)/m
    db1=np.sum(dZ1,axis=1,keepdims=True)/m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters,grads,learning_rate=1.2):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    
    dW1=grads["dW1"]
    db1=grads["db1"]
    dW2=grads["dW2"]
    db2=grads["db2"]
    
    W1=W1-learning_rate*dW1
    b1=b1-learning_rate*db1
    W2=W2-learning_rate*dW2
    b2=b2-learning_rate*db2
    
    parameters={"W1":W1,
                "b1": b1,
                "W2": W2,
                "b2": b2}
    return parameters

def nn_model(X,Y,n_h,num_iterations=10000,print_cost=False):
    np.random.seed(3)
    n_x=layer_sizes(X,Y)[0]
    n_y=layer_sizes(X,Y)[2]
    
    parameters=initialize_parameters(n_x,n_h,n_y)

    
    for i in range(0,num_iterations):
        A2,cache=forward_propagation(X,parameters)
        cost=compute_cost(A2,Y,parameters)
        grads=backward_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate=1.2)
        
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
#            print("W1 = " + str(parameters["W1"]))
#            print("b1 = " + str(parameters["b1"]))
#            print("W2 = " + str(parameters["W2"]))
#            print("b2 = " + str(parameters["b2"]))
#            print ("dW1 = "+ str(grads["dW1"]))
#            print ("db1 = "+ str(grads["db1"]))
#            print ("dW2 = "+ str(grads["dW2"]))
#            print ("db2 = "+ str(grads["db2"]))
    return parameters

def predict(parameters, X):
    m=X.shape[1]
    Y_prediction=np.zeros((1,m))
    A2,chche=forward_propagation(X,parameters)
    for i in range(A2.shape[1]):
        if A2[0][i]<=0.5:
            Y_prediction[0][i]=0
        else:
            Y_prediction[0][i]=1
    assert(Y_prediction.shape == (1, m))
    return Y_prediction

if __name__=="__main__":
    #2 - Dataset
    X,Y=load_planar_dataset()
#    # Visualize the data:
#    #plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);
#    
#    shape_X=X.shape
#    shape_Y=Y.shape
#    m=shape_X[1]
#    
#    print ('The shape of X is: ' + str(shape_X))
#    print ('The shape of Y is: ' + str(shape_Y))
#    print ('I have m = %d training examples!' % (m))
#    
#    #3 - Simple Logistic Regression
#    clf=sklearn.linear_model.LogisticRegressionCV()
#    clf.fit(X.T,Y.T)
#    LR_predictions=clf.predict(X.T)
#    print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
#       '% ' + "(percentage of correctly labelled datapoints)")
#    
#    #4.1 - Defining the neural network structure
#    X_assess, Y_assess = layer_sizes_test_case()
#    (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
#    print("The size of the input layer is: n_x = " + str(n_x))
#    print("The size of the hidden layer is: n_h = " + str(n_h))
#    print("The size of the output layer is: n_y = " + str(n_y))
#    
#    n_x, n_h, n_y = initialize_parameters_test_case()
#    #4.2 - Initialize the model's parameters
#    parameters = initialize_parameters(n_x, n_h, n_y)
#    print("W1 = " + str(parameters["W1"]))
#    print("b1 = " + str(parameters["b1"]))
#    print("W2 = " + str(parameters["W2"]))
#    print("b2 = " + str(parameters["b2"]))
    
#    #4.3 - The Loop
#    X_assess, parameters = forward_propagation_test_case()
#    A2, cache = forward_propagation(X_assess, parameters)
#
#    print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))
#    print cache['Z1'] ,cache['A1']
#    
#    A2, Y_assess, parameters = compute_cost_test_case()
#    print("cost = " + str(compute_cost(A2, Y_assess, parameters)))
#    
#    parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
#
#    grads = backward_propagation(parameters, cache, X_assess, Y_assess)
#    print ("dW1 = "+ str(grads["dW1"]))
#    print ("db1 = "+ str(grads["db1"]))
#    print ("dW2 = "+ str(grads["dW2"]))
#    print ("db2 = "+ str(grads["db2"]))
#    
#    parameters, grads = update_parameters_test_case()
#    parameters = update_parameters(parameters, grads)
#    print("W1 = " + str(parameters["W1"]))
#    print("b1 = " + str(parameters["b1"]))
#    print("W2 = " + str(parameters["W2"]))
#    print("b2 = " + str(parameters["b2"]))
    
    #4.4 - Integrate parts 4.1, 4.2 and 4.3 in nn_model()
#    X_assess, Y_assess = nn_model_test_case()
#    parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
#    print("W1 = " + str(parameters["W1"]))
#    print("b1 = " + str(parameters["b1"]))
#    print("W2 = " + str(parameters["W2"]))
#    print("b2 = " + str(parameters["b2"]))
    
#    parameters, X_assess = predict_test_case()
#
#    predictions = predict(parameters, X_assess)
#    print("predictions mean = " + str(np.mean(predictions)))
#    
#    # Build a model with a n_h-dimensional hidden layer
#    X, Y=nn_model_test_case()
#    parameters = nn_model( X, Y, n_h = 4, num_iterations = 10000, print_cost=True)
#    
#    # Plot the decision boundary
#    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0])
#    plt.title("Decision Boundary for hidden layer size " + str(4))
#    
#    # Print accuracy
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    parameters = nn_model( X, Y, n_h = 4, num_iterations = 10000, print_cost=True)
    predictions = predict(parameters, X)
    print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
    
    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i+1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations = 5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0])
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
        print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
        
#5) Performance on other datasets
        # Datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    
    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}
    
    ### START CODE HERE ### (choose your dataset)
    dataset = "noisy_moons"
    ### END CODE HERE ###
    
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    
    # make blobs binary
    if dataset == "blobs":
        Y = Y%2
    
    # Visualize the data
    plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral);