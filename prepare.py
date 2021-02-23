import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import acquire

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')


def prep_iris(df):
    # drop and rename columns
    df = df.drop(columns=['species_id', 'measurement_id']).rename(columns={'species_name': 'species'})
    
    # create dummy columns for species
    df_dummies = pd.get_dummies(df.species)
    
    # add dummy columns to df
    df = pd.concat([df, df_dummies], axis=1)
    
    return df