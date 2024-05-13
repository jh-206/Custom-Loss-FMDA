# Set of functions to use for loss functions and model evaluation

import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error

# Construct Idealized ROS curve from eyeballing plot and connecting points with cubic splines
x = np.array([0, 5, 10, 15, 20, 25, 30, 35])
y = np.array([7.5, 4.3, 3.1, 2.6, 2.1, 1.4, 0, 0])*10**-3
xvals = np.linspace(start=0, stop=35, num=100)
ros_f = CubicSpline(x, y)

def ros(fm):
    r = ros_f(fm)
    r[fm>30]=0
    return r

# RMSE Function
def rmse(observed, predicted):
    return np.sqrt(mean_squared_error(observed, predicted))

# Simple Bias Function
def bias(observed, predicted):
    return np.mean(predicted-observed)

