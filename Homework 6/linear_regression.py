# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 09:52:19 2016

@author: revan
"""

import numpy as np
import urllib

def get_data(url):
    f = urllib.urlopen(url)
    data = [[float(value) for value in line.strip('\n').split('\r')[0].split()] for line in f]
    training = [e[:-1] for e in data]
    labels = [[e[-1]] for e in data]
    return np.array(training), np.array(labels) 

def do_transformation(control):  
    """ This function reads and transforms the training and testing set
    input:   control - 0 for training set, 1 for testing set
    outputs: transformed - the transformed training/testing set: based on the control parameter
             labels - the labeling of the training/testing set: based on the control parameter
    """         
    
    training_set, training_labels = get_data("http://work.caltech.edu/data/in.dta")
    testing_set, testing_labels = get_data("http://work.caltech.edu/data/out.dta")
    
    if control == 0:
        transform = training_set
        labels = training_labels
    else:
        transform = testing_set
        labels = testing_labels        
        
    transform_size = len(transform)
    transformed = np.ones((transform_size, 8)) # we have 8 features after the transformations
    for i in range(transform_size):
        transformed[i, 1] = transform[i, 0]
        transformed[i, 2] = transform[i, 1]
        transformed[i, 3] = transform[i, 0] ** 2
        transformed[i, 4] = transform[i, 1] ** 2
        transformed[i, 5] = transform[i, 0] * transform[i, 1]
        transformed[i, 6] = np.abs(transform[i, 0] - transform[i, 1])
        transformed[i, 7] = np.abs(transform[i, 0] + transform[i, 1])
        
    return transformed, labels  
    
def compute_error():
    
    training_set, labels_training = do_transformation(0)
    testing_set, labels_testing = do_transformation(1)
    weights = np.dot(np.linalg.pinv(training_set), labels_training)
    
    # compute the training error
    training_set_length = len(training_set)
    output = np.dot(training_set, weights)
    number_of_errors = 0
    for i in range(training_set_length):
        if np.sign(output[i]) != labels_training[i]:
            number_of_errors += 1       
    error_rate_training = float(number_of_errors)/training_set_length 
    print "Training error: " + str(error_rate_training)
    
    # compute the test error
    testing_set_length = len(testing_set)
    output = np.dot(testing_set, weights)
    number_of_errors = 0
    for i in range(testing_set_length):
        if np.sign(output[i]) != labels_testing[i]:
            number_of_errors += 1       
    error_rate_testing = float(number_of_errors)/testing_set_length 
    print "Testing error: " + str(error_rate_testing)    

#compute_error()

def compute_error_regularized(regularization_parameter):
    
    training_set, labels_training = do_transformation(0)
    testing_set, labels_testing = do_transformation(1)
    
    training_set_length = len(training_set[0])
    identity = np.identity(training_set_length)
    weights = np.linalg.inv(np.dot(training_set.T, training_set) + regularization_parameter * identity)
    weights = np.dot(weights, training_set.T)
    weights = np.dot(weights, labels_training)
    
    # compute the training error
    output = np.dot(training_set, weights)
    number_of_errors = 0
    for i in range(training_set_length):
        if np.sign(output[i]) != labels_training[i]:
            number_of_errors += 1       
    error_rate_training = float(number_of_errors)/training_set_length 
    print "Training error: " + str(error_rate_training)
    
    # compute the test error
    testing_set_length = len(testing_set)
    output = np.dot(testing_set, weights)
    number_of_errors = 0
    for i in range(testing_set_length):
        if np.sign(output[i]) != labels_testing[i]:
            number_of_errors += 1       
    error_rate_testing = float(number_of_errors)/testing_set_length 
    print "Testing error: " + str(error_rate_testing)   
    
#compute_error_regularized(1e-3)  
compute_error_regularized(1e2)  
compute_error_regularized(1e1) 
compute_error_regularized(1e0) 
compute_error_regularized(1e-1) 
compute_error_regularized(1e-2)    