import pandas as pd
from sklearn.model_selection import train_test_split




def clean_iris(df):
    '''
    clean_iris will take one argument df, a pandas dataframe, anticipated to be the iris dataset
    and will remove species_id and measurement_id columns,
    rename species_name to species
    encode species into two new columns
    
    return: a single pandas dataframe with the above operations performed
    '''
        # drop and rename columns
    df = df.drop(['species_id', 'measurement_id'], axis=1)
    
        # create dummy columns for species
    df.rename(columns={'species_name': 'species'}, inplace=True)
    
        # add dummy columns to df
    dummies = pd.get_dummies(df[['species']], drop_first=True)
    
    return pd.concat([df, dummies], axis=1)




def prep_iris(df):
    '''
    prep_iris will take one argument df, a pandas dataframe, anticipated to be the iris dataset
    and will remove species_id & measurement_id columns,
    rename species_name to species,
    encode species into two new columns
    
    perform a train, validate, test split
    
    return: three pandas dataframes: train, validate, test
    '''
    df = clean_iris(df)
    
    # 20% test, 80% train_validate
    #splitting into two groups, (train+validate) and test group
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123, stratify=df.species)
    
    # then of the 80% train_validate: 30% validate, 70% train. 
    #next splitting the (train+validate) into respective groups
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=123, stratify=train_validate.species)
    
    return train, validate, test   