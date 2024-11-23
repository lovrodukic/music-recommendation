import json
import pandas as pd
from typing import List, Tuple


data = pd.read_csv('data/dataset.csv')
data['artists'] = data['artists'].apply(
    lambda x: x.split('; ') if isinstance(x, str) else []
)


def get_song_id_by_name_artist(name: str, artist: str) -> str:
    """
    Retrieve song ID name and artist from tracks_features.csv
    """
    result = data[
        (data['name'].str.lower() == name.lower()) &
        (data['artists'].apply(
            lambda artists: artist.lower() in [a.lower() for a in artists]
            )
        )
    ]
    
    if result.empty:
        raise ValueError(
            f"No song found with the name '{name}' and artist '{artist}'"
        )
    
    return result['id'].iloc[0]


def get_songs_by_artist(artist: str) -> List[Tuple[str, str]]:
    """
    Retrieve all songs and their IDs for a given artist from dataset
    """
    result = data[
        data['artists'].apply(
            lambda artists: artist.lower() in [a.lower() for a in artists]
        )
    ]
    
    if result.empty:
        raise ValueError(f"No songs found for the artist '{artist}'")

    result = result.drop_duplicates(subset='name', keep='first')
    
    return list(result[['id', 'name']].itertuples(index=False, name=None))


def map_features(feature: str, value: float, mappings: dict) -> str:
    """
    Map numerical features from the dataset to descriptive phrases
    """
    mapping = mappings.get(feature, {"unknown": "unknown characteristic"})

    if feature == "mode":
        return (
            mapping.get("major", "unknown characteristic")
            if value == 1 else mapping.get("minor", "unknown characteristic")
        )
    
    if value > 0.8:
        return mapping.get("high", "unknown characteristic")
    elif value > 0.6:
        return mapping.get("moderate", "unknown characteristic")
    elif feature == "valence" and value > 0.4:
        return mapping.get("neutral", "unknown characteristic")
    elif feature == "valence" and value > 0.2:
        return mapping.get("somewhat_negative", "unknown characteristic")
    elif feature == "valence":
        return mapping.get("highly_negative", "unknown characteristic")
    else:
        return mapping.get("low", "unknown characteristic")
