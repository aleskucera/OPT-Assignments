import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from typing import Callable

import matplotlib

# matplotlib.use('TkAgg')


def LM_iter(X: np.array, x0: float, y0: float, r0: float, mu: float) -> tuple:
        """ makes one iteration of the Levenberg-Marquardt method
        Args:
            X: np.array: input points
            x0, y0: float: coordinates of the circle center
            r0: float: circle diameter
            mu: float: parameter mu of LM algorithm

        Return:
            tuple: (x,y,r, succ), where x,y,r are circle parameters
                 succ is True iff the value of criterion f is decreased 
                                  after the update
        Shape:
        - Input: (N,2)
        """
        gtg_inverse = np.linalg.inv(first_derivative(X, x0, y0, r0).T.dot(first_derivative(X, x0, y0, r0)) + mu * np.eye(3))
        direction = gtg_inverse.dot(first_derivative(X, x0, y0, r0).T.dot(dist(X, x0, y0, r0)))
        x = x0 - direction[0]
        y = y0 - direction[1]
        r = r0 - direction[2]
        if get_objective_function(X)([x, y, r]) < get_objective_function(X)([x0, y0, r0]):
            return x, y, r, True
        else:
            return x0, y0, r0, False

def GN_iter(X: np.array, x0: float, y0: float, r0: float) -> tuple:
        """ makes one iteration of the Gauss-Newton method
        Args:
            X: np.array: input points
            x0, y0: float: coordinates of the circle center
            r0: float: circle diameter

        Return:
            tuple: (x,y,r), where x,y,r are circle parameters


        Shape:
        - Input: (N,2)
        """
        x = x0 - np.linalg.pinv(first_derivative(X, x0, y0, r0)).dot(dist(X, x0, y0, r0))[0]
        y = y0 - np.linalg.pinv(first_derivative(X, x0, y0, r0)).dot(dist(X, x0, y0, r0))[1]
        r = r0 - np.linalg.pinv(first_derivative(X, x0, y0, r0)).dot(dist(X, x0, y0, r0))[2]

        return x, y, r


def grad_iter(X: np.array, x0: float, y0: float, r0: float, a: float) -> tuple:
        """ makes one iteration of the gradient method
        Args:
            X: np.array: input points
            x0, y0: float: coordinates of the circle center
            r0: float: circle diameter
            a: float: step size

        Return:
            tuple: (x,y,r), where x,y,r are circle parameters


        Shape:
        - Input: (N,2)
        """
        print(first_derivative(X, x0, y0, r0).shape)
        x = x0 - a * np.sum(first_derivative_f(X, x0, y0, r0)[:, 0])
        y = y0 - a * np.sum(first_derivative_f(X, x0, y0, r0)[:, 1])
        r = r0 - a * np.sum(first_derivative_f(X, x0, y0, r0)[:, 2])
        return x, y, r

def first_derivative(X: np.array, x0: float, y0: float, r0: float) -> np.array:
        """ compute gradient of the objective function
        Args:
            X: np.array: input points
            x0, y0: float: coordinates of the circle center
            r0: float: circle diameter

        Return:
            np.array: gradient of the objective function
        """
        distance = np.linalg.norm(X - np.array([x0, y0])[np.newaxis, ...], axis=1, ord=2)
        derivative = np.ones((X.shape[0], 3))
        for i in range(X.shape[0]):
            derivative[i, 0] = (x0 - X[i, 0]) / distance[i]
            derivative[i, 1] = (y0 - X[i, 1]) / distance[i]
            derivative[i, 2] = -1
        return derivative

def first_derivative_f(X: np.array, x0: float, y0: float, r0: float) -> np.array:
        """ compute gradient of the objective function
        Args:
            X: np.array: input points
            x0, y0: float: coordinates of the circle center
            r0: float: circle diameter

        Return:
            np.array: gradient of the objective function
        """
        distance = np.linalg.norm(X - np.array([x0, y0])[np.newaxis, ...], axis=1, ord=2)
        derivative = np.ones((X.shape[0], 3))
        for i in range(X.shape[0]):
            derivative[i, 0] = 2 * (X[i, 0] - x0) * ((r0 / distance[i]) - 1)
            derivative[i, 1] = 2 * (X[i, 1] - y0) * ((r0 / distance[i]) - 1)
            derivative[i, 2] = 2 * (r0 - distance[i])
        return derivative


def dist(X: np.array, x0: float, y0: float, r: float) -> np.array:
        """ compute oriented distance o points form circumference.
            the distance should be positive outside the circle and 
            negative inside
        Args:
            X: np.array: input points
            x0, y0: float: coordinates of the circle center
            r: float: circle diameter

        Return:
            np.array: oriented distances from circumference


        Shape:
        - Input: (N,2)
        - Output: (N,)
        """
        # n = X.shape[0]
        # ret = np.zeros((n,1)) + x0 + y0 + r

        center = np.array([x0, y0])[np.newaxis, ...]
        center_distance = np.linalg.norm(X - center, axis=1)
        distance = center_distance - r
        return distance



def get_objective_function(X: np.array) -> Callable:
        """ returns a function f such that 
            f(c) = np.sum(dist(X,c[0], c[1], c[2])**2),
            where c = [x0, y0, r] like
        Args:
            X: np.array: input points


        Return:
            Callable: objecitve function


        Shape:
        - Input: (N,2)
        """
        return lambda c: np.sum(dist(X,c[0], c[1], c[2])**2)

def plot_circle(ax, x0, y0, r, color):
    theta = np.arange(0, 2*np.pi, 0.01)
    xs = r*np.cos(theta) + x0
    ys = r*np.sin(theta) + y0
    ax.plot(xs,ys, color)


if(__name__ == '__main__'):

    # either get points using ginput, or generate them
    use_user_input_points = False

    if(use_user_input_points):
        try:
            X = sio.loadmat('data.mat')['X']
        except:
            num_points = 10
            plt.title('Select ' + str(num_points) + ' X')
            plt.draw()
            X = np.asarray(plt.ginput(num_points))
            sio.savemat('points.mat', {'X':X})
    else:
        x0 = 5
        y0 = 6
        r = 7
        theta = np.random.rand(150)*2*np.pi
        n = len(theta)
        X = np.zeros((n,2))
        X[:,0] = r*np.cos(theta) + x0
        X[:,1] = r*np.sin(theta) + y0
        X = X + np.random.randn(n,2)*3

    # set initial parameters to r=1 and x0,y0 = mean(X)
    params = np.ones((3,3)) 
    params[:,:2] = np.mean(X,0) 

    # calculate initiall ssd. assumes same init for all methods
    d = np.sum(dist(X, *params[0])**2)
    dists = [[d],[d],[d]]
    mu = 1
    num_iter = 10
    # step size
    a = 0.01
    
    res = minimize(get_objective_function(X), x0=params[0])
    print('get_objective_function:')
    print(' objective', res.fun)
    print(' circle params x0, y0, r:', res.x)
    print()
    for i in range(num_iter):
        if(i):            
            # after timeout sec it will continue by itself
            # otherwise will wait for button press
            timeout = 0.1
            plt.waitforbuttonpress(timeout)

            # if you close the figure, program should stop
            if(not plt.get_fignums()):
                break

        plt.clf()
        ax = plt.subplot(2,2,1)
        ax.title.set_text('LM')
        ax.plot(X[:,0], X[:,1], '.')
        plot_circle(ax, *params[0], 'r')
        x,y,r,succ = LM_iter(X, *params[0], mu)
        dists[0].append(np.sum(dist(X, x,y,r)**2))
        if(succ):
            params[0] = np.array([x,y,r])
            mu /= 3
        else:
            mu *= 2
        ax.axis('equal')

        ax = plt.subplot(2,2,2)
        ax.title.set_text('GN')
        ax.plot(X[:,0], X[:,1], '.')
        plot_circle(ax, *params[1], 'g')
        x,y,r = GN_iter(X, *params[1])
        dists[1].append(np.sum(dist(X, x,y,r)**2))
        params[1] = np.array([x,y,r])
        ax.axis('equal')


        ax = plt.subplot(2,2,3)
        ax.title.set_text('grad')
        ax.plot(X[:,0], X[:,1], '.')
        plot_circle(ax, *params[2],'k')
        x,y,r = grad_iter(X, *params[2], a)
        new_dist = np.sum(dist(X, x,y,r)**2)
        if(new_dist > dists[2][-1]):
            a /= 2
            new_dist =  dists[2][-1]
        else:
            a *= 2
            params[2] = np.array([x,y,r])
        dists[2].append(new_dist)
        ax.axis('equal')

        ax = plt.subplot(4,2,6)
        ax.title.set_text('error progression')
        ax.plot(dists[0], 'r-')
        ax.plot(dists[1], 'g--')
        ax.plot(dists[2], 'k-.')
        ax.ticklabel_format(useOffset=False)
        
        ax = plt.subplot(4,2,8)
        ax.title.set_text('error of last 5 iters')
        ax.plot(dists[0][-5:], 'r-' )
        ax.plot(dists[1][-5:], 'g--' )
        ax.plot(dists[2][-5:], 'k-.' )
        ax.ticklabel_format(useOffset=False)

        plt.tight_layout()
        plt.draw()
        print('Iter %d:' %(i+1))
        print('LM: f=%f, success=%d, mu=%f' %(dists[0][-1], succ, mu))
        print('GN: f=%f' %dists[1][-1] )
        print('GM: f=%f, a: %a' %(dists[2][-1], a))
        print()
    plt.show()