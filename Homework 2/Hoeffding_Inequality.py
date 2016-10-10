# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 19:37:07 2016

@author: revan
"""

import numpy as np

def simulate_coin():
    """ Simulate a random (fair) coin """
    return np.random.randint(2, size=1)[0]
    
def flip_coin():
    """ Return the percentage of heads when we throw the coin 10 times """
    heads = 0
    for i in range(10):
        is_heads = simulate_coin()
        if is_heads == 1:
            heads += 1
            
    return heads/10.0

def one_simulation():
    """ Do one simulation of 1000 virtual coins being throwed """

    # generate 1000 coins
    coins = []
    for i in range(1000):
        coin = flip_coin()
        coins.append(coin)
        
    # the first coin
    v1 = coins[0]

    # the random coin 
    v_rand = coins[np.random.randint(10)] 
    
    # the coin with minimal value
    v_min = np.min(coins)
    
    return v1, v_rand, v_min
    
def full_simulation():
    """ Do the full simulation and see the v1, v_rand and v_min values """
    
    v1 = []
    v_rand = []
    v_min = []
    
    for i in range(100000):
        c1, c_rand, c_min = one_simulation()
        v1.append(c1)
        v_rand.append(c_rand)
        v_min.append(c_min)
        if i % 1000 == 0:
            print i
        
    return np.mean(v1), np.mean(v_rand), np.mean(v_min)    
    
v1, v_rand, v_min = full_simulation() 
print v1, v_rand, v_min

# The experiment needs a lot of CPU time to be performed. There are 100000 simulations,
# each of which has 1000 coins being thrown 10 times. In other words there are one 
# billion coins being thrown, so if you use this code, run it, go for a coffe/tea
# and then come back to see the results. For every 1000 simulations, I printed
# the current number of simulations (the i in full_simulation() function) for the user
# to see how close is the experiment to end


