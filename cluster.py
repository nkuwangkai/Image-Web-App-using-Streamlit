import pandas as pd
import numpy as np
import xgboost
from sklearn_extra.cluster import KMedoids
import streamlit as st

data = pd.read_csv("https://raw.githubusercontent.com/nkuwangkai/app-for-mortality-prediction/main/data2.csv",thousands=',')
features = data.drop(columns=[])
features
kmedoids = KMedoids(n_clusters=2,metric='euclidean',method='pam',random_state=123).fit(features)

# Title
st.header("Clustering for 2D-UCG")	

LA = st.number_input("LA (mm)",step=1)
LV = st.number_input("LV (mm)",step=1)
RA = st.number_input("RA (mm)",step=1)
RV = st.number_input("RV (mm)",step=1)
IVS = st.number_input("IVS (mm)",step=1)
LVPW = st.number_input("LVPW (mm)",step=1)
AO = st.number_input("AO (mm)",step=1)
RVTO = st.number_input("RVTO (mm)",step=1)
ESD = st.number_input("ESD (mm)",step=1)
MPA = st.number_input("MPA (mm)",step=1)

# If button is pressed
if st.button("Cluster"):
    
    data_new = pd.DataFrame([[LA, LV, RA, RV, IVS, LVPW, AO, RVTO,MPA,ESD]],
                     columns=["LA", "LV", "RA", "RV", "IVS", "LVPW", "AO", "RVTO", "ESD", "MPA"])
    # Store inputs into dataframe
    features_new = data_new.drop(columns=[])

    label=kmedoids.predict(features_new)

    # Output prediction
    st.text(f"The patient is clustered as cluster {label}")
