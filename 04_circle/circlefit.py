import scipy.io as sio
import numpy as np
from math import pi
from matplotlib import pyplot as plt

def quad_to_center(d,e,f):
    x0 = -d/2
    y0 = -e/2
    r = np.sqrt(d**2 + e**2 - 4*f)/2
    return x0, y0, r

def fit_circle_nhom(X):
    A = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    b = -np.sum(X**2, axis=1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x

def fit_circle_hom(X):
    d = 1
    e = 2
    f = 0.5
    return d,e,f

def dist(X, x0, y0, r):
    """Compute the distance between the points in X and the circle with center (x0,y0) and radius r
    :param X: 2D array of shape (N,2) containing the points
    :param x0: x-coordinate of the center of the circle
    :param y0: y-coordinate of the center of the circle
    :param r: radius of the circle
    :return: 1D array of shape (1,N) containing the distances
    """
    center = np.array([x0, y0])[np.newaxis, ...]
    center_distance = (X - center)**2
    center_distance = np.sum(center_distance, axis=1)
    center_distance = np.sqrt(center_distance)
    return center_distance - r

def fit_circle_ransac(X, num_iter, threshold):
    d,e,f = fit_circle_nhom(X)
    x0, y0, r = quad_to_center(d,e,f)
    return x0, y0, r

def plot_circle(x0,y0,r, color, label):
    t = np.arange(0,2*pi,0.01)
    X = x0 + r*np.cos(t)
    Y = y0 + r*np.sin(t)
    plt.plot(X,Y, color=color, label=label)

if(__name__ == '__main__'):
    data = sio.loadmat('data.mat')
    X = data['X'] # only inliers
    A = data['A'] # X + outliers

    def_nh = fit_circle_nhom(X)
    x0y0r_nh = quad_to_center(*def_nh)
    dnh = dist(X, *x0y0r_nh)

    def_h = fit_circle_hom(X)
    x0y0r_h = quad_to_center(*def_h)
    dh = dist(X, *x0y0r_h)

    results = {'def_nh':def_nh, 'def_h':def_h, 
               'x0y0r_nh' : x0y0r_nh, 'x0y0r_h': x0y0r_nh,
               'dnh': dnh, 'dh':dh}
    
    GT = sio.loadmat('GT.mat')
    for key in results:
        print('max difference',  np.amax(np.abs(results[key] - GT[key])), 'in', key)


    x = fit_circle_ransac(A, 2000, 0.1)

    plt.figure(1)
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], marker='.', s=3)
    plot_circle(*x0y0r_h, 'r', 'hom')
    plot_circle(*x0y0r_nh, 'b', 'nhom')
    plt.legend()
    plt.axis('equal')    
    plt.subplot(122)
    plt.scatter(A[:,0], A[:,1], marker='.', s=2)
    plot_circle(*x, 'y', 'ransac')
    plt.legend()
    plt.axis('equal')
    plt.show()
    
