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


if __name__=="__main__":
    # 2-Overview of the Problem set
    
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    
    index=25
    plt.imshow(train_set_x_orig[index])
    
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
    
    
    
    m_train = train_set_x_orig.shape[0]
    m_test =  test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))

    train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
    
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.
    
    print  train_set_x,train_set_x_flatten.shape
    
    
    
    #3 - General Architecture of the learning algorithm
    #4 - Building the parts of our algorithm


