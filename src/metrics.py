# Set of functions to use for loss functions and model evaluation

import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error
import pandas as pd

# Construct Idealized ROS curve from eyeballing plot and connecting points with cubic splines
x = np.array([0, 5, 10, 15, 20, 25, 30, 35])
y = np.array([7.5, 4.3, 3.1, 2.6, 2.1, 1.4, 0, 0])*10**-3
xvals = np.linspace(start=0, stop=35, num=100)
ros_f = CubicSpline(x, y)

def ros_0wind(fm):
    r = ros_f(fm)
    r[fm>30]=0
    return r


x = np.array(
    [
    0.0000, 0.0051, 0.0102, 0.0153, 0.0203, 0.0254, 0.0305, 0.0356, 0.0407, 0.0458, 0.0508, 0.0559, 0.0610, 0.0661, 0.0712, 0.0763, 0.0814,
    0.0864, 0.0915, 0.0966, 0.1017, 0.1068, 0.1119, 0.1169, 0.1220, 0.1271, 0.1322, 0.1373, 0.1424, 0.1475, 0.1525, 0.1576, 0.1627, 0.1678,
    0.1729, 0.1780, 0.1830, 0.1881, 0.1932, 0.1983, 0.2034, 0.2085, 0.2136, 0.2186, 0.2237, 0.2288, 0.2339, 0.2390, 0.2441, 0.2492, 0.2542,
    0.2593, 0.2644, 0.2695, 0.2746, 0.2797, 0.2848, 0.2898, 0.2949, 0.3000, 0.3051
]
)* 100
y = np.array([
    0.0955, 0.0898, 0.0845, 0.0796, 0.0752, 0.0711, 0.0674, 0.0641, 0.0610, 0.0582, 0.0556, 0.0533, 0.0512, 0.0493, 0.0475, 0.0459, 0.0445,
    0.0432, 0.0421, 0.0410, 0.0401, 0.0392, 0.0384, 0.0377, 0.0370, 0.0364, 0.0358, 0.0352, 0.0347, 0.0342, 0.0337, 0.0332, 0.0327, 0.0322,
    0.0317, 0.0311, 0.0306, 0.0300, 0.0293, 0.0286, 0.0279, 0.0271, 0.0263, 0.0254, 0.0244, 0.0234, 0.0223, 0.0211, 0.0199, 0.0185, 0.0171,
    0.0156, 0.0140, 0.0123, 0.0105, 0.0086, 0.0066, 0.0045, 0.0023, 0.0000, 0.0
])

xvals = np.linspace(start=0, stop=35, num=100)

ros_f = CubicSpline(x, y)
def ros_3wind(fm):
    r = ros_f(fm)
    r[fm>30]=0
    return r


# RMSE Function
def rmse(observed, predicted):
    return np.sqrt(mean_squared_error(observed, predicted))

# Simple Bias Function
def bias(observed, predicted):
    return np.mean(predicted-observed)

# Exponential weighting function, as w increases less weight is put on higher values of y_train
def exp_weight(y_train, w=0.1):
    """
    Function to return weight vector of length equal to vector input y. 
    Math Definition: e^(-w*y) Used for weighted loss func.
    Parameters:
    -----------
    y_train : numpy array
        observed data vector. Used on training observations, never on test for forecasts
    w : float, default=0.1
        Column of dataframe to be used for y_train, y_test
    Returns:
    -----------
    Array of length (len(y_train))
    """
    return tf.exp(tf.multiply(-w, y_train))

