import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import sys

np.set_printoptions(threshold=sys.maxsize, precision=2)

def read_data():
     # read in data
    raw_df = pd.read_csv('cc-data.csv')
    raw_df = raw_df.drop('CUST_ID', axis = 1) 
    raw_df.fillna(method ='ffill', inplace = True) 
    raw_df.head(2)

    return raw_df

def preprocess():
    # reduce dimensions of data
    raw_df = read_data()

    # Standardize data
    scaler = StandardScaler() 
    scaled_df = scaler.fit_transform(raw_df) 
    
    # Normalizing the Data 
    normalized_df = normalize(scaled_df) 
    
    # Converting the numpy array into a pandas DataFrame 
    normalized_df = pd.DataFrame(normalized_df) 
    
    # Reducing the dimensions of the data 
    pca = PCA(n_components = 2) 
    X_principal = pca.fit_transform(normalized_df) 
    X_principal = pd.DataFrame(X_principal) 
    X_principal.columns = ['P1', 'P2'] 
    
    X_principal.head(2)

    # convert dataFrame to list
    return X_principal.values.tolist()