import numpy as np
import pandas as pd
from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures
from sklearn.pipeline import Pipeline


def resample_data(df : pd.DataFrame) -> pd.DataFrame:
    """
    Resample hourly and get mean of each feature.
    
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with all features.
        
    Returns
    -------
    df : pandas dataframe
        Dataframe with all features resampled.
    """

    posto = df.station.unique()[0]
    posto_nome = df.station_name.unique()[0]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    df = df.resample(rule="60min").mean()
    df["station"] = posto
    df["station_name"] = posto_nome
    return df


def get_wind_components(
    df : pd.DataFrame, 
    wind_velocity : str, 
    wind_direction : str, 
    x_name : str, 
    y_name : str
    ) -> pd.DataFrame:
    """
    Get wind components from wind velocity and wind direction.
    
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
    """
  
    wv = df[wind_velocity]
    # convert to radians
    wd_rad = df[wind_direction] * np.pi / 180
    # calculate the wind x and y components
    df[x_name] = wv * np.cos(wd_rad)
    df[y_name] = wv * np.sin(wd_rad)
    return df


def feature_engineer(
    df : pd.DataFrame,
    features : list,
    lags : int,
    window : int
    ):
    """
    Get wind components from wind velocity and wind direction.
    
    Parameters
    ----------
    df : pandas dataframe
        Data frame with all features.

    features : list
        Features to get new features.

    lags : int 
        Number of lags.
        
    window : int
        Window size to get statistics. 
    
    Returns
    -------
    df : pandas dataframe
        Data frame with new features.
    """

    # continuous features
    dtf = DatetimeFeatures(
        variables="index",
        features_to_extract=[
            "month",
            "hour",
        ],
    )

    cyclicf = CyclicalFeatures(
        variables=["month", "hour"],
        drop_original=True, 
    )

    lagf = LagFeatures(
        variables=features, 
        freq=[f"{i}H" for i in range(1, lags+1)],
        missing_values="ignore"
    )

    winf = WindowFeatures(
        variables=features,
        window=[f"{i}H" for i in range(1, window+1)],
        freq="1H",
        functions=["mean", "std", "min", "max"],
        missing_values="ignore"
    )

    pipeline = Pipeline(
        [
            ("dtf", dtf),
            ("cyclicf", cyclicf),
            ("lagf", lagf),
            ("winf", winf)
        ]
    )

    df = pipeline.fit_transform(df)
    return df