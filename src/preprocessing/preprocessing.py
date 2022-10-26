import numpy as np
import pandas as pd


def get_station_name(df):
    '''Create station name feature.
    
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with all features.
        
    Returns
    -------
    df : pandas dataframe
        Dataframe with station name.
    '''
    
    stations_names = {
        1000866: 'Anhembi',
        1000850: "M'Boi Mirim",
        1000857: 'Capela do Socorro',
        1000860: 'Mooca',
        1000859: 'Vila Formosa',
        1000862: 'São Miguel Paulista',
        1000842: 'Butantã',
        1000880: 'Santana do Parnaíba',
        1000876: 'Mauá',
        503: 'Sé',
        1000848: 'Lapa',
        1000854: 'Campo Limpo',
        1000844: 'São Mateus',
        495: 'Vila Mariana',
        1000840: 'Ipiranga',
        1000852: 'Santo Amaro',
        507: 'Parelheiros',
        504: 'Perus',
        509: 'Freguesia do Ó',
        510: 'Tucuruvi',
        400: 'Riacho Grande',
        1000887: 'Penha',
        524: 'Vila Prudente',
        1000864: 'Itaquera',
        515: 'Pirituba',
        540: 'Vila Maria',
        592: 'Cidade Ademar',
        1000882: 'Itaim Paulista',
        634: 'Jabaquara',
        635: 'Pinheiros',
        1000944: 'Tremembé',
        1000300: 'NaN'
        }

    df['Posto Nome'] = df['Posto']
    df['Posto Nome'] = df['Posto Nome'].map(stations_names)
    
    return df
    
    
def change_features_names(df):
    '''Rename features.
    
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with all features.
        
    Returns
    -------
    df : pandas dataframe
        Dataframe renamed features.
    '''
    
    df.columns = [
        'station',
        'station_name',
        'timestamp', 
        'precipitation', 
        'wind_velocity', 
        'wind_direction', 
        'temperature',
        'relative_humidity', 
        'pressure', 
        'wind_blow', 
        'thermal_sensation' 
        ]

    return df


def resample_data(df):
    '''Resample hourly and get mean.
    
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


def get_time_periodicity(df, timestamp):
    '''Get time periodicity features.
    
    Parameters
    ----------
    df : pandas dataframe
        Data frame with all features.
    timestamp : pandas series
        Timestamp from data.
    
    Returns
    -------
    df : pandas dataframe
        Dataframe with periodicity features.
    '''

    day = 24 * 60 * 60
    year = (365.2425) * day
    
    df['day_sin'] = np.sin(timestamp * (2 * np.pi / day))
    df['day_cos'] = np.cos(timestamp * (2 * np.pi / day))
    df['year_sin'] = np.sin(timestamp * (2 * np.pi / year))
    df['year_cos'] = np.cos(timestamp * (2 * np.pi / year))
    
    return df


def lags(df, features, n_lags):
  for feature in features:
    for i in range(1, n_lags+1):
      df['{}__lag{}'.format(feature, i)] = df[feature].shift(i)
  return df


def difference(df, features):
  for feature in features:
    df['{}__diff'.format(feature)] = df[feature].diff()
  return df


def feature_engineer_pipeline(df):
    '''Apply all feature engineer functions on dataframe.
    
    Parameters
    ----------
    df : pandas dataframe
        Data frame with all features.
    
    Returns
    -------
    df : pandas dataframe
        Dataframe with all new features.
    '''

    l = []
    for i in list(df.station_name.unique()):
        df_station = df[df.station_name == i]
        try:
            df_station = get_wind_components(
                df_station, 
                wind_velocity='wind_velocity', 
                wind_direction='wind_direction', 
                x_name='wind_velocity_x', 
                y_name='wind_velocity_y'
                )
        
            df_station = df_station.drop(['wind_velocity'], axis=1)
        except:
            pass

        try:
            df_station = get_wind_components(
                df_station, 
                wind_velocity='wind_blow', 
                wind_direction='wind_direction', 
                x_name='wind_blow_x', 
                y_name='wind_blow_y'
                )
        
            df_station = df_station.drop(['wind_blow'], axis=1)
        except:
            pass

        try:
            df_station = df_station.drop(['wind_direction'], axis=1)
        except:
            pass

        df_station = resample_data(df_station)
        df_station = df_station.interpolate()
        df_station = df_station.reset_index()
        date_time = pd.to_datetime(df_station['timestamp'])
        timestamp_seconds = date_time.map(pd.Timestamp.timestamp) # converting it to seconds
        df_station = get_time_periodicity(df_station, timestamp=timestamp_seconds)
        l.append(df_station)

    df = pd.concat(l)

    df = df[[
            'station',
            'station_name',
            'timestamp',
            'temperature',
            'precipitation',
            'relative_humidity',
            'pressure',
            'wind_velocity_x',
            'wind_velocity_y',
            'wind_blow_x',
            'wind_blow_y',
            'day_sin',
            'day_cos',
            'year_sin',
            'year_cos'
            ]]

    return df