import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import plotly.express as px
import plotly.graph_objects as go
from functools import singledispatch
from fmda_models import XGB, LM, RF
from metrics import ros_3wind, ros_0wind
import tensorflow as tf

def initialize_models(params):
    models = {
        'XGB' : XGB(params['xgb']),
        'LM' : LM(params['lm']),
        'RF' : RF(params['rf'])
    }

    return models

def create_exp_function(w):
    def exp_function(y_train):
        return tf.exp(tf.multiply(-w, y_train))
    return exp_function

def loss_setup(params, ws = None):
    loss_fucs = ["MSE", "exp", "ROS"]
    # set up return dictionary
    loss = {
        'MSE' : {
            'w_func' : None
        }
    } 
    # Using input omega parameter list, add dictionary key for exponential weighting for each omega in list 
    if ws is not None:
        for w in ws:
            assert isinstance(w, float) # Check that given list of floats
            dname = f"exp_{w}" # create name of dictionary key
            loss[dname] = {
                'w_func' : create_exp_function(w)
            }
    loss["ROS"] = {'w_func': ros_3wind}

    models = initialize_models(params)
    for l in loss:
        loss[l][f"errs"]={}
        for mod in models:
            loss[l][f"errs"][mod] = {
                "rmse_test" : [],
                "rmse_test_ROS" : [],
                "t": [] # time of earliest train period
            }
    
    return models, loss


# Map stations, credit 
    # https://stackoverflow.com/questions/53233228/plot-latitude-longitude-from-csv-in-python-3-6
    # https://gist.github.com/blaylockbk/79658bdde8c1334ab88d3a67c6e57477

## NOTE: doesn't work on conda environment ml, but works on base. TODO: investigate installs
def make_st_map(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    buf = 4
    lons = (np.amin(df.lon), np.amax(df.lon))
    lats = (np.amin(df.lat), np.amax(df.lat))
    map = Basemap(projection='cyl', llcrnrlat=lats[0]-buf, urcrnrlat=lats[1]+buf,
                    llcrnrlon=lons[0]-buf,urcrnrlon=lons[1]+buf, resolution="l")
    map.drawcoastlines()
    map.drawcountries()
    map.drawstates()

    map.arcgisimage(service="World_Street_Map", xpixels=1000, verbose=False)
    
    x, y = map(df['lon'].values, df['lat'].values)
    map.scatter(x, y, marker='o')
    
    # Add a rectangle representing the bounding box
    vertices = [map(lons[0], lats[1]), map(lons[1], lats[1]), 
                map(lons[1], lats[0]), map(lons[0], lats[0])]
    bounding_box = Polygon(vertices, edgecolor='r', linewidth=1.5, facecolor='none', zorder=5)

    ax.add_patch(bounding_box)
    
    return 

# Map stations, credit https://stackoverflow.com/questions/53233228/plot-latitude-longitude-from-csv-in-python-3-6

def make_st_map_interactive(df):
    fig = go.Figure(go.Scattermapbox(
        lat=df['lat'],
        lon=df['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10,
            opacity=0.7,
        ),
        text=df['stid'],
        showlegend=False  # Turn off legend
    ))

    # Add Points
    center_lon=df['lon'].median()
    center_lat=df['lat'].median()
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center=dict(lat=center_lat, lon=center_lon)
    )
    # Add Lines for Bounding Box
    
    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=[df['lon'].min(), df['lon'].min(), df['lon'].max(), df['lon'].max(), df['lon'].min()],
        lat=[df['lat'].min(), df['lat'].max(), df['lat'].max(), df['lat'].min(), df['lat'].min()],
        marker=dict(size=5, color="black"),
        line=dict(width=1.5, color="black"),
        showlegend=False
    ))
    
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        mapbox_zoom =5,
        mapbox_center={"lat": np.median(df.lat), "lon": np.median(df.lon)},  # Center the map on desired location
    )
    return fig




# Function to generate hash value for various data structures not readily hashed in base python
# used to track reproducibility

@singledispatch
## Top level hash function with built-in hash function for str, float, int, etc
def hash2(x):
    return hash(x)

@hash2.register(np.ndarray)
## Hash numpy array, hash array with pandas and return integer sum
def _(x):
    # return hash(x.tobytes())
    return np.sum(pd.util.hash_array(x))

@hash2.register(list)
## Hash list, convert to tuple
def _(x):
    return hash2(tuple(x))

@hash2.register(tuple)
def _(x):
    r = 0
    for i in range(len(x)):
        r+=hash2(x[i])
    return r

@hash2.register(dict)
## Hash dict, loop through keys and hash each element via dispatch. Return hashed integer sum of hashes
def _(x, keys = None, verbose = False):
    r = 0 # return value integer
    if keys is None: # allow user input of keys to hash, otherwise hash them all
        keys = [*x.keys()]
    keys.sort()
    for key in keys:
        if (verbose): print('Hashing', key)
        r += hash2(x[key])
    return hash(r)
