import numpy as np
import pandas as pd


def get_station_name(df : pd.DataFrame) -> pd.DataFrame:
    """
    Creates station name feature.
    
    Parameters
    ----------
    df : pandas data frame
        Data set with all features.
        
    Returns
    -------
    df : pandas data frame
        Data set with station name.
    """
    
    station_names = {
        1000866: "Anhembi",
        1000850: "M'Boi Mirim",
        1000857: "Capela do Socorro",
        1000860: "Mooca",
        1000859: "Vila Formosa",
        1000862: "São Miguel Paulista",
        1000842: "Butantã",
        1000880: "Santana do Parnaíba",
        1000876: "Mauá",
        503: "Sé",
        1000848: "Lapa",
        1000854: "Campo Limpo",
        1000844: "São Mateus",
        495: "Vila Mariana",
        1000840: "Ipiranga",
        1000852: "Santo Amaro",
        507: "Parelheiros",
        504: "Perus",
        509: "Freguesia do Ó",
        510: "Tucuruvi",
        400: "Riacho Grande",
        1000887: "Penha",
        524: "Vila Prudente",
        1000864: "Itaquera",
        515: "Pirituba",
        540: "Vila Maria",
        592: "Cidade Ademar",
        1000882: "Itaim Paulista",
        634: "Jabaquara",
        635: "Pinheiros",
        1000944: "Tremembé",
        1000300: "NaN"
        }

    df["Posto Nome"] = df["Posto"]
    df["Posto Nome"] = df["Posto Nome"].map(station_names)
    return df