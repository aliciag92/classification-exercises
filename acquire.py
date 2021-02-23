import os
import pandas as pd

#defines function to create a sql url using personal credentials
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'



#returns titanic data from db as a pandas df
def get_titanic_data():
    '''
    This function reads in the titanic data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = 'SELECT * FROM passengers'
    titanic_df = pd.read_sql(sql_query, get_connection('titanic_db'))
    return titanic_df


#adds caching to get_titanic_data and checks for local filename (titanic.csv)
#if file exists, uses the .csv file
#if file doesn't exist, then produces SQL & pandas necessary to create a df, then write the df to a .csv file
def caching_titanic_data(cached=False):
    '''
    This function reads in titanic data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in titanic df from
    a csv file, returns df.
    ''' 
    if cached == False or os.path.isfile('titanic_df.csv') == False:
        # Read fresh data from db into a DataFrame.
        df = get_titanic_data()
        # Write DataFrame to a csv file.
        df.to_csv('titanic_df.csv')
    else:
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('titanic_df.csv', index_col=0)
        
    return df



#returns iris data from db as a pandas df
def get_iris_data():
    '''
    This function reads in the iris data from the Codeup db
    and returns a pandas DataFrame.
    '''
    sql_query = 'SELECT * FROM measurements JOIN species USING (species_id)'
    iris_df = pd.read_sql(sql_query, get_connection('iris_db'))
    return iris_df


#adds caching to get_iris_data and checks for local filename (iris.csv)
#if file exists, uses the .csv file
#if file doesn't exist, then produces SQL & pandas necessary to create a df, then write the df to a .csv file
def caching_iris_data(cached=False):
    '''
    This function reads in iris data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in iris df from
    a csv file, returns df.
    ''' 
    if cached == False or os.path.isfile('iris_df.csv') == False:
        # Read fresh data from db into a DataFrame.
        df = get_iris_data()
        # Write DataFrame to a csv file.
        df.to_csv('iris_df.csv')
    else:
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('iris_df.csv', index_col=0)
    return df

