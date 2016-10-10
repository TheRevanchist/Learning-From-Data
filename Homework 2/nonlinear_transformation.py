# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 19:56:33 2016

@author: revan
"""

import numpy as np
import random

def create_points(number_of_points):
    """ create_points create number_of_points points where the number
    of points is the number of points we want in our dataset
    inputs: number_of_points - the number of points we want in our dataset
    output: points - a 2D numpy array which contain the points """
    
    x = []
    y = []
    
    for i in range(number_of_points):
        x_i = random.uniform(-1, 1)
        y_i = random.uniform(-1, 1)
        x.append(x_i)
        y.append(y_i)
     
    x = np.asarray(x)
    y = np.asarray(y)      
    
    points = [x, y]
    points = np.transpose(points)
    return points
    
    
def determine_output(points, number_of_points):
    """ This function decides which points are classified originally as -1 and
    which are classified as 1. 
    inputs:  points - the matrix of points
             number_of_points - the number of points in the dataset
    output:  y - the original classification of the points """         
    
    y = np.zeros(number_of_points)
    for i in range(number_of_points):
        y[i] = np.sign(points[i, 0] ** 2 + points[i, 1] ** 2 - 0.6)
    
    if number_of_points != 0:        
        to_flip = np.random.randint(0, 1000, 100)    
        for element in to_flip:
            y[element] *= -1 
    return y  

def add_bias(points, number_of_points):
    """ This function adds bias term in the dataset
    inputs:  points - the dataset which contains the points
             number_of_points - the number of points in the dataset
    output:  points - the augmented dataset """         
    
    points = np.insert(points, 0, 1, axis=1)
    return points
    
def add_nonlinear_transformation(points, number_of_points):
    """ This function adds the nonlinear transformations x1 * x2, x1 ^ 2 and x2 ^ 2 
        inputs: points - the dataset which contains the points
                number_of_points - the number of points in the dataset
        output: points - the new dataset which contains the new features """
    
    x1x2 = np.multiply(points[:, 1], points[:, 2])
    x1_square = np.multiply(points[:, 1], points[:, 1])
    x2_square = np.multiply(points[:, 2], points[:, 2])
    points = np.insert(points, 3, x1x2, axis=1)
    points = np.insert(points, 4, x1_square, axis=1)
    points = np.insert(points, 5, x2_square, axis=1) 
    return points    
        

def compute_error_linear_model(number_of_points_training):
    """ This function computes the error in the training set using a linear regression classifier
    inputs:  number_of_points_training - number of points in the training set
    outputs: error_rate_training - error rate in the training set """      
    
    # generate the training test
    points_training = create_points(number_of_points_training) 
    y = determine_output(points_training, number_of_points_training)
    points_training = add_bias(points_training, number_of_points_training)
    
    # compute the weights
    weights = np.dot(np.linalg.pinv(points_training), y)
    
    
    # compute the training error
    output = np.dot(points_training, weights)
    number_of_errors = 0
    for i in range(number_of_points_training):
        if np.sign(output[i]) != y[i]:
            number_of_errors += 1       
    error_rate_training = float(number_of_errors)/number_of_points_training    
    
    return error_rate_training

def linear_model():
    """ This function runs the linear regression classifier
    prints: final_error_rate_training - the training error """
    
    number_of_points_training = 1000
    final_error_rate_training = 0.0
        
    number_of_iterations = 1000
    for i  in range(number_of_iterations):
        error_rate_training = compute_error_linear_model(number_of_points_training)
        final_error_rate_training += error_rate_training
    final_error_rate_training /= number_of_iterations
        
    # print train/test error for the linear regression used for classification
    print "Error rate in the training set is: " + str(final_error_rate_training)
 
def compute_error_linear_model_augmented(number_of_points_training, number_of_points_testing):
    """ This function computes the error in the training set using a linear regression classifier
    inputs:  number_of_points_training - number of points in the training set
    outputs: error_rate_training - error rate in the training set 
             output - the output we got by performing the classifier
             points_training - the dataset 
             error_rate_testing - error rate in the test set """      
    
    # generate the training test
    points_training = create_points(number_of_points_training) 
    y = determine_output(points_training, number_of_points_training)
    points_training = add_bias(points_training, number_of_points_training)
    points_training = add_nonlinear_transformation(points_training, number_of_points_training)
    
    # compute the weights
    weights = np.dot(np.linalg.pinv(points_training), y)
    
    # compute the training error
    output = np.dot(points_training, weights)
    number_of_errors = 0
    for i in range(number_of_points_training):
        if np.sign(output[i]) != y[i]:
            number_of_errors += 1       
    error_rate_training = float(number_of_errors)/number_of_points_training

    # generate the testing set 
    points_testing = create_points(number_of_points_testing) 
    y_testing = determine_output(points_testing, number_of_points_testing)
    points_testing = add_bias(points_testing, number_of_points_testing)
    points_testing = add_nonlinear_transformation(points_testing, number_of_points_testing) 
    
    # compute the testing error
    output_testing = np.dot(points_testing, weights)
    number_of_errors_testing = 0
    for i in range(number_of_points_testing):
        if np.sign(output_testing[i]) != y_testing[i]:
            number_of_errors_testing += 1       
    error_rate_testing = float(number_of_errors_testing)/number_of_points_testing
    
    return error_rate_training, output, points_training, error_rate_testing
   
def linear_model_augmented():
    """ This function solves the exercise Nr.9 and Exercise Nr.10 """
    
    # define the number of training points and testing points
    number_of_points_training = 1000
    number_of_points_testing = 1000
    final_error_rate_training = 0.0
    
    number_of_iterations = 1000
    for i  in range(number_of_iterations):
        error_rate_training, output, points_training, final_error_rate_testing =\
            compute_error_linear_model_augmented(number_of_points_training, number_of_points_testing)
        final_error_rate_training += error_rate_training
    final_error_rate_training /= number_of_iterations
    
    # the given hypothesis'
    missmatches1 = 0
    missmatches2 = 0
    missmatches3 = 0
    missmatches4 = 0
    missmatches5 = 0

    # define the number of missmatches with our hypothesis for every given hypothesis
    for i in range(number_of_points_training):
        
        if np.sign(-1 * points_training[i, 0] - 0.05 * points_training[i, 1] +\
            0.08 * points_training[i, 2] + 0.13 * points_training[i, 3] +\
            1.5 * points_training[i, 4] + 1.5 * points_training[i, 5]) != np.sign(output[i]):
                missmatches1 += 1
                
        if np.sign(-1 * points_training[i, 0] - 0.05 * points_training[i, 1] +\
            0.08 * points_training[i, 2] + 0.13 * points_training[i, 3] +\
            1.5 * points_training[i, 4] + 15 * points_training[i, 5]) != np.sign(output[i]):
                missmatches2 += 1 
                
        if np.sign(-1 * points_training[i, 0] - 0.05 * points_training[i, 1] +\
            0.08 * points_training[i, 2] + 0.13 * points_training[i, 3] +\
            15 * points_training[i, 4] + 1.5 * points_training[i, 5]) != np.sign(output[i]):
                missmatches3 += 1  
                
        if np.sign(-1 * points_training[i, 0] - 1.5 * points_training[i, 1] +\
            0.08 * points_training[i, 2] + 0.13 * points_training[i, 3] +\
            0.05 * points_training[i, 4] + 0.05 * points_training[i, 5]) != np.sign(output[i]):
                missmatches4 += 1  
                
        if np.sign(-1 * points_training[i, 0] - 0.05 * points_training[i, 1] +\
            0.08 * points_training[i, 2] + 1.5 * points_training[i, 3] +\
            0.15 * points_training[i, 4] + 0.15 * points_training[i, 5]) != np.sign(output[i]):
                missmatches5 += 1                
                
    print "Testing error is: " + str(final_error_rate_training) 
    best_hypothesis = np.asarray([missmatches1, missmatches2, missmatches3, missmatches4, missmatches5])
    print "Best hypothesis is hypothesis No." + str(np.argmin(best_hypothesis) + 1)
    print "Testing error is: " + str(final_error_rate_testing) 
    
# run the algorithms   
print "Exercise 8: linear model - results!"    
linear_model()  
print "-----------------------------------"
print "Exercise 9 and 10: nonlinear transformation results!"
linear_model_augmented()   