# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:05:32 2017

@author: yuan
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w=np.zeros((dim,1),float)
    b=0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))    
    return w, b

def propagate(w,b,X,Y):
    m=X.shape[1]
    A=w.T.dot(X)+b
    a=sigmoid(A)
    cost=-1*((Y.dot(np.log(a).T))+(1-Y).dot(np.log(1-a).T))/m
    dw=X.dot((a-Y).T)/m
    db=(a-Y).sum()/m
    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    cost=np.squeeze(cost)
    assert(cost.shape==())
    grads = {"dw": dw,
             "db": db}
    return grads, cost

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs=[]
    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y)
        dw=grads["dw"]
        db=grads["db"]
        w=w-dw*learning_rate
        b=b-db*learning_rate
        
        if i%100==0:
            costs.append(cost)
        if print_cost and i%100==0:
            print ("Cost after iteration %i:%f" %(i,cost))
    
    params={"w":w,
            "b":b}
    grads={"dw":dw,
           "db":db}
    
    return params,grads,costs


def predict(w,b,X):
    m=X.shape[1]
    Y_prediction=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)
    A=sigmoid(w.T.dot(X)+b)
    
    for i in range(A.shape[1]):
        if A[0][i]<0.5:
            Y_prediction[0][i]=0
        else:
            Y_prediction[0][i]=1
    assert(Y_prediction.shape == (1, m))
    return Y_prediction


def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=False):
    w,b=initialize_with_zeros(X_train.shape[0])
    parameters,grads,costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost=False)
    w=parameters["w"]
    b=parameters["b"]
    
    Y_prediction_test=predict(w,b,X_test)
    Y_prediction_train=predict(w,b,X_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d        
    
    

if __name__=="__main__":
    # 2-Overview of the Problem set
    
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    
    index=25
# =============================================================================
#     plt.imshow(train_set_x_orig[index])
# =============================================================================
    
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
    
    
    
    m_train = train_set_x_orig.shape[0]
    m_test =  test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

# =============================================================================
#     print ("Number of training examples: m_train = " + str(m_train))
#     print ("Number of testing examples: m_test = " + str(m_test))
#     print ("Height/Width of each image: num_px = " + str(num_px))
#     print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
#     print ("train_set_x shape: " + str(train_set_x_orig.shape))
#     print ("train_set_y shape: " + str(train_set_y.shape))
#     print ("test_set_x shape: " + str(test_set_x_orig.shape))
#     print ("test_set_y shape: " + str(test_set_y.shape))
# =============================================================================

    train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
    
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.
    
    print  train_set_x.shape,test_set_x_flatten.shape
    
    
# =============================================================================
#     
#     #3 - General Architecture of the learning algorithm
#     #4 - Building the parts of our algorithm
#     print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
#     
#     #4.2
#     dim = 2
#     w, b = initialize_with_zeros(dim)
#     print ("w = " + str(w))
#     print ("b = " + str(b))
#     #4.3
#     w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
#     grads, cost = propagate(w, b, X, Y)
#     print ("dw = " + str(grads["dw"]))
#     print ("db = " + str(grads["db"]))
#     print ("cost = " + str(cost))
#     #4.4
#     params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
# 
#     print ("w = " + str(params["w"]))
#     print ("b = " + str(params["b"]))
#     print ("dw = " + str(grads["dw"]))
#     print ("db = " + str(grads["db"]))
#     
#     
#     w = np.array([[0.1124579],[0.23106775]])
#     b = -0.3
#     X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
#     print ("predictions = " + str(predict(w, b, X)))
# =============================================================================
    

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    print d
    index = 1
# =============================================================================
#     plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
# =============================================================================
# =============================================================================
#     print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")
# =============================================================================

# =============================================================================
#     costs = np.squeeze(d['costs'])
#     plt.plot(costs)
#     plt.ylabel('cost')
#     plt.xlabel('iterations (per hundreds)')
#     plt.title("Learning rate =" + str(d["learning_rate"]))
#     plt.show()
# =============================================================================

    #6 
# =============================================================================
#     learning_rates = [0.01, 0.001, 0.0001]
#     models = {}
#     for i in learning_rates:
#         print ("learning rate is: " + str(i))
#         models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
#         print ('\n' + "-------------------------------------------------------" + '\n')
#     
#     for i in learning_rates:
#         plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
# 
#     plt.ylabel('cost')
#     plt.xlabel('iterations')
#     
#     legend = plt.legend(loc='upper center', shadow=True)
#     frame = legend.get_frame()
#     frame.set_facecolor('0.90')
#     plt.show()
# =============================================================================
    
    #7
    ## START CODE HERE ## (PUT YOUR IMAGE NAME) 
    my_image = "la_defense.jpg"   # change this to the name of your image file 
    ## END CODE HERE ##
    
    # We preprocess the image to fit your algorithm.
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)
    
    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
        