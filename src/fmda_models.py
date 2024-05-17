import pandas as pd
import numpy as np
import xgboost as xg
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from abc import ABC, abstractmethod
from metrics import ros


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Machine Learning Model Classes
#
# The following are classes used to orgnaize "fit" and "predict" calls for various ML models,
# and additionally some helpful functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



class MLModel(ABC):
    def __init__(self, params: dict):
        self.params = params
        if type(self) is MLModel:
            raise TypeError("MLModel is an abstract class and cannot be instantiated directly")
        super().__init__()

    @abstractmethod
    def fit(self, X_train, y_train, weights=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def eval(self, X_test, y_test):
        preds = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        rmse_ros = np.sqrt(mean_squared_error(ros(y_test), ros(preds)))
        print(f"Test RMSE: {rmse}")
        print(f"Test RMSE (ROS): {rmse_ros}")
        return rmse, rmse_ros

class XGB(MLModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.model = XGBRegressor(**self.params)

    def fit(self, X_train, y_train, weights=None):
        self.model.fit(X_train, y_train, sample_weight=weights)
        print(f"Training XGB with params: {self.params}")

    def predict(self, X):
        preds = self.model.predict(X)
        print("Predicting with XGB")
        return preds

class LM(MLModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.model = LinearRegression(**self.params)

    def fit(self, X_train, y_train, weights=None):
        self.model.fit(X_train, y_train, sample_weight=weights)
        print(f"Training LM with params: {self.params}")

    def predict(self, X):
        preds = self.model.predict(X)
        print("Predicting with LM")
        return preds





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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Baseline Models
# Established methods to compare ML models, used to ensure reasonable results so
# conclusions can be drawn about loss functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Define model function with drying, wetting, and rain equilibria

# Parameters
r0 = 0.05                                   # threshold rainfall [mm/h]
rs = 8.0                                    # saturation rain intensity [mm/h]
Tr = 14.0                                   # time constant for rain wetting model [h]
S = 250                                     # saturation intensity [dimensionless]
T = 10.0                                    # time constant for wetting/drying

def model_decay(m0,E,partials=0,T1=0.1,tlen=1):  
    # Arguments: 
    #   m0          fuel moisture content at start dimensionless, unit (1)
    #   E           fuel moisture eqilibrium (1)
    #   partials=0: return m1 = fuel moisture contents after time tlen (1)
    #           =1: return m1, dm0/dm0 
    #           =2: return m1, dm1/dm0, dm1/dE
    #           =3: return m1, dm1/dm0, dm1/dE dm1/dT1   
    #   T1          1/T, where T is the time constant approaching the equilibrium
    #               default 0.1/hour
    #   tlen        the time interval length, default 1 hour

    exp_t = np.exp(-tlen*T1)                  # compute this subexpression only once
    m1 = E + (m0 - E)*exp_t                   # the solution at end
    if partials==0:
        return m1
    dm1_dm0 = exp_t
    if partials==1:
        return m1, dm1_dm0          # return value and Jacobian
    dm1_dE = 1 - exp_t      
    if partials==2:
        return m1, dm1_dm0, dm1_dE 
    dm1_dT1 = -(m0 - E)*tlen*exp_t            # partial derivative dm1 / dT1
    if partials==3:
        return m1, dm1_dm0, dm1_dE, dm1_dT1       # return value and all partial derivatives wrt m1 and parameters
    raise('Bad arg partials')


def ext_kf(u,P,F,Q=0,d=None,H=None,R=None):
    """
    One step of the extended Kalman filter. 
    If there is no data, only advance in time.
    :param u:   the state vector, shape n
    :param P:   the state covariance, shape (n,n)
    :param F:   the model function, args vector u, returns F(u) and Jacobian J(u)
    :param Q:   the process model noise covariance, shape (n,n)
    :param d:   data vector, shape (m). If none, only advance in time
    :param H:   observation matrix, shape (m,n)
    :param R:   data error covariance, shape (n,n)
    :return ua: the analysis state vector, shape (n)
    :return Pa: the analysis covariance matrix, shape (n,n)
    """
    def d2(a):
        return np.atleast_2d(a) # convert to at least 2d array

    def d1(a):
        return np.atleast_1d(a) # convert to at least 1d array

    # forecast
    uf, J  = F(u)          # advance the model state in time and get the Jacobian
    uf = d1(uf)            # if scalar, make state a 1D array
    J = d2(J)              # if scalar, make jacobian a 2D array
    P = d2(P)              # if scalar, make Jacobian as 2D array
    Pf  = d2(J.T @ P) @ J + Q  # advance the state covariance Pf = J' * P * J + Q
    # analysis
    if d is None or not d.size :  # no data, no analysis
        return uf, Pf
    # K = P H' * inverse(H * P * H' + R) = (inverse(H * P * H' + R)*(H P))'
    H = d2(H)
    HP  = d2(H @ P)            # precompute a part used twice  
    K   = d2(np.linalg.solve( d2(HP @ H.T) + R, HP)).T  # Kalman gain
    # print('H',H)
    # print('K',K)
    res = d1(H @ d1(uf) - d)          # res = H*uf - d
    ua = uf - K @ res # analysis mean uf - K*res
    Pa = Pf - K @ d2(H @ P)        # analysis covariance
    return ua, d2(Pa)

def model_moisture(m0,Eqd,Eqw,r,t=None,partials=0,T=10.0,tlen=1.0):
    # arguments:
    # m0         starting fuel moistureb (%s
    # Eqd        drying equilibrium      (%) 
    # Eqw        wetting equilibrium     (%)
    # r          rain intensity          (mm/h)
    # t          time
    # partials = 0, 1, 2
    # returns: same as model_decay
    #   if partials==0: m1 = fuel moisture contents after time 1 hour
    #              ==1: m1, dm1/dm0 
    #              ==2: m1, dm1/dm0, dm1/dE  
    
    if r > r0:
        # print('raining')
        E = S
        T1 =  (1.0 - np.exp(- (r - r0) / rs)) / Tr
    elif m0 <= Eqw: 
        # print('wetting')
        E=Eqw
        T1 = 1.0/T
    elif m0 >= Eqd:
        # print('drying')
        E=Eqd
        T1 = 1.0/T
    else: # no change'
        E = m0
        T1=0.0
    exp_t = np.exp(-tlen*T1)
    m1 = E + (m0 - E)*exp_t  
    dm1_dm0 = exp_t
    dm1_dE = 1 - exp_t
    #if t>=933 and t < 940:
    #  print('t,Eqw,Eqd,r,T1,E,m0,m1,dm1_dm0,dm1_dE',
    #        t,Eqw,Eqd,r,T1,E,m0,m1,dm1_dm0,dm1_dE)   
    if partials==0: 
        return m1
    if partials==1:
        return m1, dm1_dm0
    if partials==2:
        return m1, dm1_dm0, dm1_dE
    raise('bad partials')

def model_augmented(u0,Ed,Ew,r,t):
    # state u is the vector [m,dE] with dE correction to equilibria Ed and Ew at t
    # 
    m0, Ec = u0  # decompose state u0
    # reuse model_moisture(m0,Eqd,Eqw,r,partials=0):
    # arguments:
    # m0         starting fuel moistureb (1)
    # Ed         drying equilibrium      (1) 
    # Ew         wetting equilibrium     (1)
    # r          rain intensity          (mm/h)
    # partials = 0, 1, 2
    # returns: same as model_decay
    #   if partials==0: m1 = fuel moisture contents after time 1 hour
    #              ==1: m1, dm0/dm0 
    #              ==2: m1, dm1/dm0, dm1/dE 
    m1, dm1_dm0, dm1_dE  = model_moisture(m0,Ed + Ec, Ew + Ec, r, t, partials=2)
    u1 = np.array([m1,Ec])   # dE is just copied
    J =  np.array([[dm1_dm0, dm1_dE],
                   [0.     ,     1.]])
    return u1, J

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Default Uncertainty Matrices
Q = np.array([[1e-3, 0.],
            [0,  1e-3]]) # process noise covariance
H = np.array([[1., 0.]])  # first component observed
R = np.array([1e-3]) # data variance

def run_augmented_kf(dat,h2=None,hours=None, H=H, Q=Q, R=R):
    if h2 is None:
        h2 = int(dat['h2'])
    if hours is None:
        hours = int(dat['hours'])
    
    d = dat['fm']
    Ed = dat['Ed']
    Ew = dat['Ew']
    rain = dat['rain']
    
    u = np.zeros((2,hours))
    u[:,0]=[0.1,0.0]       # initialize,background state  
    P = np.zeros((2,2,hours))
    P[:,:,0] = np.array([[1e-3, 0.],
                      [0.,  1e-3]]) # background state covariance
    # Q = np.array([[1e-3, 0.],
    #             [0,  1e-3]]) # process noise covariance
    # H = np.array([[1., 0.]])  # first component observed
    # R = np.array([1e-3]) # data variance

    for t in range(1,h2):
      # use lambda construction to pass additional arguments to the model 
        u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
                                  lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t),
                                  Q,d[t],H=H,R=R)
      # print('time',t,'data',d[t],'filtered',u[0,t],'Ec',u[1,t])
    for t in range(h2,hours):
        u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
                                  lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t),
                                  Q*0.0)
      # print('time',t,'data',d[t],'forecast',u[0,t],'Ec',u[1,t])
    return u














