# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 10:57:38 2016

@author: revan
"""

import numpy as np
import urllib
from copy import deepcopy

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

        
    
def compute_error(k, num_training):
    
    training_and_validation_set, labels_training_and_validation = do_transformation(0)
    testing_set, labels_testing = do_transformation(1)
    
    # split the training set into training and validation
    training_set = training_and_validation_set[num_training:, :k]
    validation_set = training_and_validation_set[:num_training, :k]
    labels_training = labels_training_and_validation[num_training:]
    labels_validation = labels_training_and_validation[:num_training]
    
    # get only the needed features for the testing
    testing_set = testing_set[:, :k]
    
    weights = np.dot(np.linalg.pinv(training_set), labels_training)
    
    # compute the training error
    training_set_length = len(training_set)
    output = np.dot(training_set, weights)
    number_of_errors = 0
    for i in range(training_set_length):
        if np.sign(output[i]) != labels_training[i]:
            number_of_errors += 1       
    error_rate_training = float(number_of_errors)/training_set_length 
    
    # compute the validation error
    validation_set_length = len(validation_set)
    output = np.dot(validation_set, weights)
    number_of_errors = 0
    for i in range(validation_set_length):
        if np.sign(output[i]) != labels_validation[i]:
            number_of_errors += 1       
    error_rate_validation = float(number_of_errors)/validation_set_length 
    print "Validation error, k = " + str(k) + ":   " +  str(error_rate_validation)     
    
    # compute the test error
    testing_set_length = len(testing_set)
    output = np.dot(testing_set, weights)
    number_of_errors = 0
    for i in range(testing_set_length):
        if np.sign(output[i]) != labels_testing[i]:
            number_of_errors += 1       
    error_rate_testing = float(number_of_errors)/testing_set_length 
    print "Testing error, k = " + str(k) + ":   " + str(error_rate_testing)  
    
compute_error(3, 25)
compute_error(4, 25)
compute_error(5, 25)
compute_error(6, 25)
compute_error(7, 25)  
    
    