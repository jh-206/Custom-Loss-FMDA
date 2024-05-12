import json
import subprocess
import os.path as osp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import logging

# Function for spatiotemporal crossvalidation
def train_test_split_spacetime(df, yid = "fm", spid = "stid", tid = "date", 
                               temporal_test_frac = 0.1, spatial_test_frac = 0.2,
                               verbose=True):
    """
    Method to split a dataframe into train/test, accounting for spatiotemporal relationships.
    Resulting test set will consist of locations not included in training and at future times.
    NOTE: set seed externally
    Parameters:
    -----------
    df : DataFrame
        Dataframe of data to be split
    yid : str, default='fm'
        Column of dataframe to be used for y_train, y_test
    spid : str, default='stid'
        Spatial ID, column of dataframe used to identify unique locations. 
        If lon/lat specify unique locations, use these to construct a single IDs column
    tid : str, default = "date"
        Temporal ID, column of dataframe consisting of datetimes of observations
    temporal_test_frac : float, default = 0.1
        Percent of timesteps to hold out for test set. Test set taken from most recent times
    spatial_test_frac : float, default = 0.2
        Percent of unique locations to hold out for test set.
    vergose : boolean, default = True
        Indicator on whether to print info to console
    
    Returns:
    -----------
    X_train, X_test, y_train, y_test : 
        X_train & X_test : dataframes
        y_train, y_test : pandas Series
    """

    # Get array of unique spatial locations
    locs = df[spid].unique()
    train_locs, test_locs = train_test_split(locs, test_size=spatial_test_frac)

    # Subset based on time
    times = df[tid].unique()
    hour_diff = (times.max() - times.min()).total_seconds() / 3600 # Difference in hours between start and stop times
    h2 = times.min() + timedelta(hours=hour_diff*temporal_test_frac) # time marking train/test split

    # Split based on space and time
    df_train = df[(df[tid] < h2) & (df[spid].isin(train_locs))]
    df_test = df[(df[tid] >= h2) & (~df[spid].isin(train_locs))]

    # Return arrays in the standard format
    X_train = df_train.loc[:, ~df.columns.str.contains(yid)]
    X_test = df_test.loc[:, ~df.columns.str.contains(yid)]
    y_train = df_train[yid]
    y_test = df_test[yid]

    # Print info if verbose
    if verbose:
        print(f"Number of Training Observations: {X_train.shape[0]}")
        print(f"Number of Training Locations: {len(X_train.stid.unique())}")
        print(f"Time range Train: {X_train.date.min().strftime('%Y-%m-%d %H:%M:%S'), X_train.date.max().strftime('%Y-%m-%d %H:%M:%S')}")
        print("~"*50)
        print(f"Number of Test Observations: {X_test.shape[0]}")
        print(f"Number of Test Locations: {len(X_test.stid.unique())}")
        print(f"Time range Train: {X_test.date.min().strftime('%Y-%m-%d %H:%M:%S'), X_test.date.max().strftime('%Y-%m-%d %H:%M:%S')}")
    return X_train, X_test, y_train, y_test














def get_stids(start, end, bbox, meso_token, raws_vars = ["fuel_moisture"], save_path = "data"):
    # Given bounding box and time frame, and desired data varaibles, 
    # return dictionary of RAWS station IDs with available data
    # Inputs:
        # bbox : (list) lon/lat bounding box. Input format expected to be that of wrfxpy, note different from Synoptic
        # start: (str) start time string, expected format "%Y%m%d%H%m"
        # end : (str) end time string
        # meso_token: (str) data access public token from Synoptic / MesoWest
        # vars: 
        # save_path: (str) local path to save json file
    # Returns: (df) Dataframe of RAWS station IDs. Also saves json file locally

    # Helper function used internally to call curl as a subprocess. NOTE: this is needed since SynopticPy returns too many STIDs for this desired implementation
    def call_curl(url):
        try:
            # Run the curl command and capture its output
            result = subprocess.run(['curl', url], capture_output=True, text=True, check=True)
            # Decode the JSON output
            json_data = json.loads(result.stdout)
            return json_data
        except subprocess.CalledProcessError as e:
            print("Error executing curl command:", e)
            return None
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            return None

    # Convert bbox from wrfxpy format to synoptic
    bbox2 = [bbox[1], bbox[0], bbox[3], bbox[2]]

    url = f"https://api.synopticdata.com/v2/stations/metadata?bbox={bbox2[0]},{bbox2[1]},{bbox2[2]},{bbox2[3]}&vars={','.join(raws_vars)}&obrange={start},{end}&token={meso_token}"
    print(f"Attempting Synoptic retrieval from URL: https://api.synopticdata.com/v2/stations/metadata?bbox={bbox2[0]},{bbox2[1]},{bbox2[2]},{bbox2[3]}&vars={','.join(raws_vars)}&obrange={start},{end}&token=HIDDEN")
    command = f"curl -X GET '{url}'"
    sts_json = call_curl(url)

    # Save Output as json
    print(f"Saving STIDs to {osp.join(save_path, 'raws_stations.json')}")
    with open(osp.join(save_path, "raws_stations.json"), 'w') as json_file:
        json.dump(sts_json, json_file)

    # Convert to DataFrame with a format friendly to SynopticPy
    sts = pd.DataFrame(sts_json["STATION"], index=[i["STID"] for i in sts_json["STATION"]])
    sts = sts.transpose()
    
    return sts


def filter_nan_values(t1, v1):
    # Filter out NaN values from v1 and corresponding times in t1
    valid_indices = ~np.isnan(v1)  # Indices where v1 is not NaN
    t1_filtered = np.array(t1)[valid_indices]
    v1_filtered = np.array(v1)[valid_indices]
    return t1_filtered, v1_filtered


# Interpolation Function to use for missing data
def time_intp(t1, v1, t2):
    # Check if t1 v1 t2 are 1D arrays
    if t1.ndim != 1:
        logging.error("Error: t1 is not a 1D array. Dimension: %s", t1.ndim)
        return None
    if v1.ndim != 1:
        logging.error("Error: v1 is not a 1D array. Dimension %s:", v1.ndim)
        return None
    if t2.ndim != 1:
        logging.errorr("Error: t2 is not a 1D array. Dimension: %s", t2.ndim)
        return None
    # Check if t1 and v1 have the same length
    if len(t1) != len(v1):
        logging.error("Error: t1 and v1 have different lengths: %s %s",len(t1),len(v1))
        return None
    t1_no_nan, v1_no_nan = filter_nan_values(t1, v1)
    # print('t1_no_nan.dtype=',t1_no_nan.dtype)
    # Convert datetime objects to timestamps
    t1_stamps = np.array([t.timestamp() for t in t1_no_nan])
    t2_stamps = np.array([t.timestamp() for t in t2])
    
    # Interpolate using the filtered data
    v2_interpolated = np.interp(t2_stamps, t1_stamps, v1_no_nan)
    if np.isnan(v2_interpolated).any():
        logging.error('time_intp: interpolated output contains NaN')
    
    return v2_interpolated

def str2time(input):
    """
    Convert a single string timestamp or a list of string timestamps to corresponding datetime object(s).
    """
    if isinstance(input, str):
        return datetime.strptime(input.replace('Z', '+00:00'), '%Y-%m-%dT%H:%M:%S%z')
    elif isinstance(input, list):
        return [str2time(s) for s in input]
    else:
        raise ValueError("Input must be a string or a list of strings")