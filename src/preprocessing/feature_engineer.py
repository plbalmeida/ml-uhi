import numpy as np
import pandas as pd
from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import DropMissingData
from feature_engine.selection import DropFeatures
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures


def resample_data(df):
    '''Resample hourly and get mean of each feature.
    
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with all features.
        
    Returns
    -------
    df : pandas dataframe
        Dataframe with all features resampled.
    '''
    posto = df.station.unique()[0]
    posto_nome = df.station_name.unique()[0]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df = df.set_index('timestamp')
    df = df.resample(rule='60min').mean()
    df['station'] = posto
    df['station_name'] = posto_nome
    
    return df


def get_wind_components(df, wind_velocity, wind_direction, x_name, y_name):
    '''Get wind components from wind velocity and wind direction.
    
    Parameters
    ----------
    df : pandas dataframe
        Data frame with all features.
    wind_velocity : str
        Wind velocity feature name in dataframe.
    wind_direction : str
        Wind direction feature name in dataframe.
    x_name : str
        x component desired name.
    y_name : str
        y component desired name.
    
    Returns
    -------
    df : pandas dataframe
        Dataframe with wind components features.
    '''
  
    wv = df[wind_velocity]
    
    # convert to radians
    wd_rad = df[wind_direction] * np.pi / 180
    
    # calculate the wind x and y components
    df[x_name] = wv * np.cos(wd_rad)
    df[y_name] = wv * np.sin(wd_rad)
    
    return df