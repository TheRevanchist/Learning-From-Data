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
    
def initialize_weights(number_of_points, dimensions):
    """ The function initializes the weights to 0 
    inputs:   number_of_points - the number of points in the dataset
             dimensions - the dimension of the dataset
    output:  weights - the array which contains the weights """  
    
    weights = np.zeros(dimensions + 1) # add 1 dimension for the bias term
    return weights
    
def run_iteration(points, number_of_points, y, weights):
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
        weights, has_not_converged, a = run_iteration(points, number_of_points, y, weights)
        number_of_iterations += 1
        if has_not_converged == 0:
            control = 0
         
    print "Perceptron converged in: " + str(number_of_iterations) + " iterations!"
    return number_of_iterations, weights
    
def compute_test_error(number_of_test_points, test_points, test_y, weights):
    """ The function computes the 'test_error" by generating random points and
    seeing if they have been classified correctly
    inputs:  number_of_test_points - the number of test points
             points - the points we used for the training
             weight - the weights of the perceptron
             point1, point2 - points needed to labell the testing points
    output: the error rate """         
    
    computed_y = np.sign(np.dot(test_points, weights))
    incorrectly_classified = 0.0
    
    for i in range(number_of_test_points):
        if test_y[i] != computed_y[i]:
            incorrectly_classified += 1
            
    return incorrectly_classified/number_of_test_points        
            

def run_simulation(number_of_points, dimensions): 
    """ This function runs 1000 simulations to compute the average number of 
    iterations needed for perceptron to converge, in addition to the average
    'test error' of the perceptron """
    
    
    number_of_simulations = 1000
    number_of_test_points = 500
    suma = []
    suma2 = 0
    sum_test_error = 0
    
    for i in range(number_of_simulations):
        
        points = create_points(number_of_points) # generate dataset    
        point1, point2 = choose_two_points(number_of_points, points) # choose two random points
        y = determine_output(points, point1, point2, number_of_points) # labell the points
        points = add_bias(points, number_of_points) # add bias in the dataset
        weights = initialize_weights(number_of_points, dimensions) # initialize the weights to 0
        
        number_of_iterations, weights = perceptron(points, number_of_points, y, weights) # run perceptron
        
        test_points = create_points(number_of_test_points)
        test_y = determine_output(test_points, point1, point2, number_of_test_points)
        test_points = add_bias(test_points, number_of_test_points)
        test_error = compute_test_error(number_of_test_points, test_points, test_y, weights) # compute the test error
        suma.append(number_of_iterations)
        suma2 += number_of_iterations
        sum_test_error += test_error
        
    average = suma2/number_of_simulations
    median = np.median(suma)
    return average, median, test_error

# the following parameter control the number of points we want in the training
# set and the number of points we want in the testing set
number_of_points = 10

# the following parameter controls the dimensionality of the data, for this 
# assignment is not needed (because the dimension of the data is always 2) but
# by changing the parameter we can use the perceptron to classify data in any 
# dimensional space (the code on create_points and choose_two_points needs a
# slight adjustement
dimensions = 2

average, median, test_error = run_simulation(number_of_points, dimensions)
print "Average number of iterations: " + str(average)
print "Median number of iterations: " + str(median)
print "Average test error is: " + str(test_error)