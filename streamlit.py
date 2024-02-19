
'''
Created on 19 Feb 2023

@author: Sanjjushri

source:

How to run?
stremlit run streamlit.py

'''

import streamlit as st
import pandas as pd 
from ctgan import CTGAN 

number = st.number_input('Number of rows', min_value=0, step=1000)

def data_gen():

    df = pd.read_csv("dataset/rising-ml-stars.csv")

    string_data = df.select_dtypes(exclude="number").columns.tolist()

    ctgan = CTGAN(epochs=5)
    ctgan.fit(df, string_data)

    synthetic_data = ctgan.sample(number)

    return synthetic_data

st.dataframe(data_gen())

# synthetic_data.to_csv("dataset/fake-data.csv")