import numpy as np
import matplotlib.pyplot as plt

def equi_dist_circ(center, radius, n):
    return np.full((n,2), center) + np.array([[radius*np.cos(2*np.pi*(i/n)), radius*np.sin(2*np.pi*(i/n))] for i in range(1,n+1)])

def zero_error_two_circles():
    center1, radius1 = [0,0], 1
    center2, radius2 = [2,2], 0.5
    n = 100
    
    data = np.concatenate([equi_dist_circ(center1, radius1, n), equi_dist_circ(center2, radius2, n)])
    np.random.default_rng().shuffle(data)
    
    return data

def zero_error_three_circles():
    center1, radius1 = [0,0], 0.5
    center2, radius2 = [1.25,1], 0.5
    center3, radius3 = [-1.25,2], 0.5
    n = 100
    
    data = np.concatenate([equi_dist_circ(center1, radius1, n), equi_dist_circ(center2, radius2, n), equi_dist_circ(center3, radius3, n)])
    np.random.default_rng().shuffle(data)
    
    return data

def unavoidable_error_three_circles():
    center1, radius1 = [0,0], 1
    center2, radius2 = [1,2], 1.1
    center3, radius3 = [-1.75, 2.5], 1
    n = 100
    
    data = np.concatenate([equi_dist_circ(center1, radius1, n), equi_dist_circ(center2, radius2, n), equi_dist_circ(center3, radius3, n)])
    np.random.default_rng().shuffle(data)
    
    return data

def zero_error_y_cut():
    center1, radius1 = [0,0], 1
    center2, radius2 = [0,2], 0.5
    center3, radius3 = [0,4], 0.5
    n = 100
    
    data = np.concatenate([equi_dist_circ(center1, radius1, n), equi_dist_circ(center2, radius2, n), equi_dist_circ(center3, radius3, n)])
    np.random.default_rng().shuffle(data)
    
    return data

def zero_error_x_cut():
    center1, radius1 = [0,0], 1
    center2, radius2 = [2,0], 0.5
    center3, radius3 = [4,0], 0.5
    n = 100
    
    data = np.concatenate([equi_dist_circ(center1, radius1, n), equi_dist_circ(center2, radius2, n), equi_dist_circ(center3, radius3, n)])
    np.random.default_rng().shuffle(data)
    
    return data