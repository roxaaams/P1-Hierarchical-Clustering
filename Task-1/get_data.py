# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:21:34 2021

@author: pault
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

def read_and_normalize(path):
    raw_df = pd.read_csv(path)
    raw_df = raw_df.drop('CUST_ID', axis = 1) 
    raw_df.fillna(method ='ffill', inplace = True) 
    
    # Standardize data
    scaler = StandardScaler() 
    scaled_df = scaler.fit_transform(raw_df) 
      
    # Normalizing the Data 
    normalized_df = normalize(scaled_df) 
      
    # Converting the numpy array into a pandas DataFrame 
    normalized_df = pd.DataFrame(normalized_df) 
      
    # Reducing the dimensions of the data 
    pca = PCA(n_components = 2) 
    data = pca.fit_transform(normalized_df)
    
    return data