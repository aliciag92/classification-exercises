import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


###################### Prepare Iris Data ######################


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




###################### Prepare Titanic Data ######################


def clean_titanic(df):
    '''
    This function will:
    drop any duplicate observations, 
    drop columns not needed, 
    fill missing embarktown with 'Southampton',
    create dummy vars of sex and embark_town(encoding)
    drop sex and embark_town since it is encoded
    
    and return a single cleaned dataframe
    '''
        # drop duplicate rows
    df.drop_duplicates(inplace=True)

        # removing duplicates and columns w/ too many nulls
    df.drop(columns=['deck', 'embarked', 'class', 'age'], inplace=True)

        # fill in missing values on embark_town w/ most common value "Southampton"
    df.embark_town.fillna(value='Southampton', inplace=True)

        #encode categorical variables that remain as strings/objects
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    df = df.drop(columns=['sex', 'embark_town'])

    return pd.concat([df, dummy_df], axis=1)





def prep_titanic(df):
    df = clean_titanic(df)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.survived)
    return train, validate, test




############## Titanic Data for Exploration Lesson ############

def handle_missing_values(df):
    return df.assign(
        embark_town=df.embark_town.fillna('Other'),
        embarked=df.embarked.fillna('O'),
    )

def remove_columns(df):
    return df.drop(columns=['deck'])

def encode_embarked(df):
    encoder = LabelEncoder()
    encoder.fit(df.embarked)
    return df.assign(embarked_encode = encoder.transform(df.embarked))

def prep_titanic_data(df):
    df = df\
        .pipe(handle_missing_values)\
        .pipe(remove_columns)\
        .pipe(encode_embarked)
    return df

def train_validate_test_split(df, seed=123):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df.survived
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify=train_and_validate.survived,
    )
    return train, validate, test