# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 17:27:35 2016

@author: revan
"""

import numpy as np
import random

# The following functions are needed for exercises 5, 6 and 7

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
    
 
def choose_two_points(number_of_points, points):
    """ The function randomly chooses two points (needed later to create)
    the hypothesis which seperates the data into positive and negative
    examples.
    inputs:  number_of_points - the number of points in the dataset
             points - the 2D array which contains the points
    outputs: point1 - 1D array which contains the coordinates of the first point
             point2 - 1D array which contains the coordinates of the second point"""          
    
    x1 = random.uniform(-1, 1)
    y1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y2 = random.uniform(-1, 1)
    point1 = np.asarray([x1, y1])
    point2 = np.asarray([x2, y2])

    return point1, point2 
    
def determine_output(points, point1, point2, number_of_points):
    """ This function decides which points are classified originally as -1 and
    which are classified as 1. 
    inputs:  points - the matrix of points
             point1 - the randomly chosen first point
             point2 - the randomly chosen second point
             number_of_points - the number of points in the dataset
    output:  y - the original classification of the points """         
    
    y = np.zeros(number_of_points)
    slope = (point2[1] - point1[1])/(point2[0] - point1[0])
    for i in range(number_of_points):
        if points[i, 1] > slope * (points[i, 0] - point1[0]) + point1[1]:
            y[i] = 1.0
        else:
            y[i] = -1.0
    return y 
    
def add_bias(points, number_of_points):
    """ This function adds bias term in the dataset
    inputs:  points - the dataset which contains the points
             number_of_points - the number of points in the dataset
    output:  points - the augmented dataset """         
    
    points = np.insert(points, 0, 1, axis=1)
    return points    

# the following function solves exercise 5 and 6
    
def compute_error(number_of_points_training, number_of_points_testing):
    """ This function computes the error in the training set and in the testing set
    inputs:  number_of_points_training - number of points in the training set
             number_of_points_testing - number of points in the testing set
    outputs: error_rate_training - error rate in the training set
             error_rate_testing - error rate in the testing set 
             weights - the vector of weights 
             points_training - the matrix of points
             y - the labelling of the points """      
    
    # generate the training test
    points_training = create_points(number_of_points_training) 
    point1, point2 = choose_two_points(number_of_points_training, points_training)
    y = determine_output(points_training, point1, point2, number_of_points_training)
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
    
    # generate the testing set
    points_testing = create_points(number_of_points_testing)
    y_test = determine_output(points_testing, point1, point2, number_of_points_testing)
    points_testing = add_bias(points_testing, number_of_points_testing)
    
    # compute the testing error
    
    output_test = np.dot(points_testing, weights)
    number_of_errors_test = 0
    for i in range(number_of_points_testing):
        if np.sign(output_test[i]) != y_test[i]:
            number_of_errors_test += 1  
    
    if number_of_points_testing != 0:       
        error_rate_testing = float(number_of_errors_test)/number_of_points_testing
    else:
        error_rate_testing = 0.0
    
    return error_rate_training, error_rate_testing, weights, points_training, y
    
# the following function solves exercise 7
    
def run_iteration_perceptron(points, number_of_points, y, weights):
    """ This function runs an entire iteration of the PLA
    if the iteration is successul it returns 1, otherwise it updates the weights
    and returns 0
    inputs:  points - the dataset containing the points
             number_of_points - the number of points in the dataset
             y - the labelling of the points
             weights - the weight vector
    outputs: weights - the updated weight vector
             has_not_converged - 0/1 boolean value (coverged/not converged) 
             a - our classification hypothesis """
    
    has_not_converged = 0      
    a = np.sign(np.dot(points, weights)) # the computed sign for each point
    missclassified = []
    choice = -1
        
    for i in range(number_of_points):       
        if a[i] != y[i]:
            missclassified.append(i)
            has_not_converged = 1
            
    if has_not_converged:        
        choice = np.random.choice(missclassified)  
        weights = weights + y[choice] * points[choice] 
 
    return weights, has_not_converged, a 
    
    
def perceptron(points, number_of_points, y, weights):
    """ The function runs the PLA algorithm until convergence 
    inputs:  same as previous function
    output:  number_of_iterations - the number of iterations needed to converge
             weights - the final weights """
    
    control = 1
    number_of_iterations = 0
    while (control):
        weights, has_not_converged, a = run_iteration_perceptron(points, number_of_points, y, weights)
        number_of_iterations += 1
        if has_not_converged == 0:
            control = 0
         
    return number_of_iterations, weights     

def run_simulation():
    """ This function runs the simulations for exercises 5, 6 and 7, computing the
    training error, the test error for the linear regression used for classification,
    and the average number of iterations for the PLA
    prints: the training error
            the testing error 
            number of iterations needed for PLA to converge """
    
    number_of_points_training = 100
    number_of_points_testing = 1000
    final_error_rate_training = 0.0
    final_error_rate_testing = 0.0
        
    number_of_iterations = 1000
    for i  in range(number_of_iterations):
        error_rate_training, error_rate_testing, weights, points_training, y = compute_error(number_of_points_training, number_of_points_testing)
        final_error_rate_training += error_rate_training
        final_error_rate_testing += error_rate_testing
    final_error_rate_training /= number_of_iterations
    final_error_rate_testing /= number_of_iterations
        
    # print train/test error for the linear regression used for classification
    print "Error rate in the training set is: " + str(final_error_rate_training)
    print "Error rate in the testing set is: " + str(final_error_rate_testing)
    
    number_of_points_training = 10
    number_of_points_testing = 0
    final_error_rate_training = 0.0
    final_error_rate_testing = 0.0
        
    number_of_iterations = 1000
    iterations_to_converge_mean = 0
    for i  in range(number_of_iterations):
        error_rate_training, error_rate_testing, weights, points_training, y = compute_error(number_of_points_training, number_of_points_testing)
        iterations_to_converge, weights = perceptron(points_training, number_of_points_training, y, weights)
        iterations_to_converge_mean += iterations_to_converge
    iterations_to_converge_mean /= float(number_of_iterations)
    
    # print the average number of iterations needed for PLA to converge
    print "The average number of iterations needed for PLA to converge is: " + str(iterations_to_converge_mean)       
    
run_simulation()    
    
        
    
    
    

