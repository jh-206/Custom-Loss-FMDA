import pandas as pd
import numpy as np
import xgboost as xg
import tensorflow as tf
import tensorflow.keras.backend as K
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
    loss : str, default='reg:squarederror'
        loss function to be optimized during training.
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
    def __init__(self, params, loss='reg:squarederror'):
        """
        Initialize the XGB class.
        Parameters:
        -----------
        params : dict
            HyperParameters to be passed to the XGBoost model.
        loss : str or custom func, default='reg:squarederror'
            loss function to be optimized during training.
            See https://xgboost.readthedocs.io/en/latest/parameter.html for options.
        """
        self.params = params
        self.params['objective'] = loss # XGBoost package uses term objective function, translate here
        self.model = xg.XGBRegressor(**self.params)

    def fit(self, X_train, y_train, w=None):
        """
        Train the XGBoost model on the training data.

        Parameters:
        -----------
        X_train : array-like or sparse matrix of shape (n_samples, n_features)
            Training input samples.
        y_train : array-like of shape (n_samples,)
            Target values.
        w: array-like of shape (n_ssamples,)
            Weights to use for custom loss
        """
        if w is None:
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, sample_weight=w)

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
    loss : 

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
    def __init__(self, loss=None):
        """
        Initialize the LM class.
        Parameters:
        -----------
        loss : str or custom func
        """
        # self.params['loss'] = loss
        self.model = LinearRegression()

    def fit(self, X_train, y_train, w=None):
        """
        Train the LM model on the training data.

        Parameters:
        -----------
        X_train : array-like or sparse matrix of shape (n_samples, n_features)
            Training input samples.
        y_train : array-like of shape (n_samples,)
            Target values.
        w : array-like of shape (n_samples,)
            sample weights for weighted loss func
        """
        if w is None:
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, w)

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

# Multilayer Perceptron
class MLP:
    """
    Wrapper class for multilayer perceptron neural network model.
    Parameters:
    -----------
    params : dict
        Parameters to be passed to the XGBoost model.
    loss : 
    Attributes:
    -----------
    model : keras.engine.sequential.Sequential
        Underlying neural network.
    params : dict
        Parameters passed to the NN model.

    Methods:
    --------
    fit(X_train, y_train):
        Train the model on the training data.
    predict(X_test):
        Make predictions using the trained model.
    """
    def __init__(self, params, loss='mean_squared_error'):
        """
        Initialize the MLP class.
        Parameters:
        -----------
        params : dict
            HyperParameters to be passed to the XGBoost model.
        objective : str or custom func.
        """
        self.params = params
        self.params['loss'] = loss
        self.model = self._build_model()
        self.compile_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.params['hidden_units'], activation=self.params['activation'], input_shape=(self.params['input_dim'],)),
            tf.keras.layers.Dropout(self.params['dropout']),  # Dropout layer
            tf.keras.layers.Dense(1)  # Output layer with a single neuron for regression
        ])
        return model
    def compile_model(self):
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        self.model.compile(optimizer=optimizer,
                           loss=self.params['loss'],
                           metrics=self.params.get('metrics', ['accuracy']))
    def fit(self, X_train, y_train):
        """
        Train the model model on the training data.

        Parameters:
        -----------
        X_train : array-like or sparse matrix of shape (n_samples, n_features)
            Training input samples.
        y_train : array-like of shape (n_samples,)
            Target values.
        """
        self.model.fit(X_train, y_train, epochs=self.params['epochs'], batch_size=self.params['batch_size'], validation_split=self.params['validation_split'])

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
    def summary(self):
        return self.model.summary()