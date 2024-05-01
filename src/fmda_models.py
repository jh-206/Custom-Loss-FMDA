import pandas as pd
import numpy as np
import xgboost as xg
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Machine Learning Model Classes
#
# The following are classes used to orgnaize "fit" and "predict" calls for various ML models,
# and additionally some helpful functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# XGBoost Model
class XGB:
    """
    Wrapper class for XGBoost regression model.
    Parameters:
    -----------
    params : dict
        Parameters to be passed to the XGBoost model.
    objective : str, default='reg:squarederror'
        Objective function to be optimized during training.
        See https://xgboost.readthedocs.io/en/latest/parameter.html for options.

    Attributes:
    -----------
    model : xgboost.XGBRegressor
        Underlying XGBoost regression model.
    params : dict
        Parameters passed to the XGBoost model.

    Methods:
    --------
    fit(X_train, y_train):
        Train the XGBoost model on the training data.

    predict(X_test):
        Make predictions using the trained model.

    get_feature_importance():
        Get feature importances from the trained model.
    """
    def __init__(self, params, objective='reg:squarederror'):
        """
        Initialize the XGB class.
        Parameters:
        -----------
        params : dict
            HyperParameters to be passed to the XGBoost model.
        objective : str or custom func, default='reg:squarederror'
            Objective function to be optimized during training.
            See https://xgboost.readthedocs.io/en/latest/parameter.html for options.
        """
        self.params = params
        self.params['objective'] = objective
        self.model = xg.XGBRegressor(**self.params)

    def fit(self, X_train, y_train):
        """
        Train the XGBoost model on the training data.

        Parameters:
        -----------
        X_train : array-like or sparse matrix of shape (n_samples, n_features)
            Training input samples.
        y_train : array-like of shape (n_samples,)
            Target values.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the trained model.
        Parameters:
        -----------
        X_test : array-like or sparse matrix of shape (n_samples, n_features)
            Test input samples.

        Returns:
        --------
        array-like of shape (n_samples,)
            Predicted target values.
        """
        return self.model.predict(X_test)

    def get_feature_importance(self):
        """
        Get feature importances from the trained model.
        Returns:
        --------
        array of shape (n_features,)
            Feature importances.
        """
        return self.model.feature_importances_


# Linear Regression Model
class LM:
    """
    Wrapper class for LinearRegression model.
    Parameters:
    -----------
    params : dict
        Parameters to be passed to the XGBoost model.
    objective : 

    Attributes:
    -----------
    model : sklearn.linear_model._base.LinearRegression
        Underlying linear regression model.

    Methods:
    --------
    fit(X_train, y_train):
        Train the XGBoost model on the training data.

    predict(X_test):
        Make predictions using the trained model.

    get_coefs():
        Get coefficient estimates and summary stats.
    """
    def __init__(self, objective=None):
        """
        Initialize the LM class.
        Parameters:
        -----------
        objective : str or custom func
        """
        # self.params['objective'] = objective
        self.model = LinearRegression()

    def fit(self, X_train, y_train):
        """
        Train the LM model on the training data.

        Parameters:
        -----------
        X_train : array-like or sparse matrix of shape (n_samples, n_features)
            Training input samples.
        y_train : array-like of shape (n_samples,)
            Target values.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the trained model.
        Parameters:
        -----------
        X_test : array-like or sparse matrix of shape (n_samples, n_features)
            Test input samples.

        Returns:
        --------
        array-like of shape (n_samples,)
            Predicted target values.
        """
        return self.model.predict(X_test)
    def get_coefs(self):
        return lm.coef_

