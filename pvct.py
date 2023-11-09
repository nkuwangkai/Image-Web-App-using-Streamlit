import pandas as pd
import numpy as np
import xgboost
from sklearn_extra.cluster import KMedoids
import streamlit as st

data = pd.read_csv("https://raw.githubusercontent.com/nkuwangkai/app-for-mortality-prediction/main/data4.csv",thousands=',')
features = data.drop(columns=[])
features
kmedoids = KMedoids(n_clusters=2,metric='euclidean',method='pam',random_state=123).fit(features)

# Title
st.title("AF recurrence cluster form PVCT")
# st.header("Clustering for PVCT")
st.subheader("Yuehui Yin, Department of Cardiology, the Second Affiliated Hospital of Chongqing Medical University")
# st.caption("Yuehui Yin, Department of Cardiology, the Second Affiliated Hospital of Chongqing Medical University")
							

LSPV_APD = st.number_input("LSPV_APD (mm)")
LSPV_SID = st.number_input("LSPV_SID (mm)")
LIPV_APD = st.number_input("LIPV_APD (mm)")
LIPV_SID = st.number_input("LIPV_SID (mm)")
RSPV_APD = st.number_input("RSPV_APD (mm)")
RSPV_SID = st.number_input("RSPV_SID (mm)")
RIPV_APD = st.number_input("RIPV_APD (mm)")
RIPV_SID = st.number_input("RIPV_SID (mm)")

# If button is pressed
if st.button("Cluster"):
    
    data_new = pd.DataFrame([[LSPV_APD, LSPV_SID, LIPV_APD, LIPV_SID, RSPV_APD, RSPV_SID, RIPV_APD, RIPV_SID]],
                     columns=["LSPV_APD", "LSPV_SID", "LIPV_APD", "LIPV_SID", "RSPV_APD", "RSPV_SID", "RIPV_APD", "RIPV_SID"])
    # Store inputs into dataframe
    features_new = data_new.drop(columns=[])

    label=kmedoids.predict(features_new)

    # Output prediction
    st.text(f"The patient is clustered as cluster {label}")
