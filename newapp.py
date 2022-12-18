import pandas as pd
import numpy as np
import xgboost
from xgboost.sklearn import XGBClassifier
import streamlit as st

# Title
st.header("Machine learning app for in-hospital mortality prediction")
st.sidebar.slider(‘Age’, 18, 100, 50)
st.sidebar.slider(‘Temperature’, 31, 40, 37)
st.sidebar.slider(‘RespiratoryRate’, 10, 48, 20)
st.sidebar.slider(‘HeartRate’, 29, 175, 100)
st.sidebar.slider(‘SBP’, 40, 203, 160)
st.sidebar.slider(‘AG’, 5, 49, 30)
st.sidebar.slider(‘BUN’, 2, 270, 10)
st.sidebar.slider(‘MCHC’, 24, 39, 20)
st.sidebar.slider(‘MCV’, 24, 39, 20)
st.sidebar.slider(‘RDW’, 10, 36, 20)
st.sidebar.slider(‘WBC’, 0.1, 250, 10)

Race = st.selectbox("Race (white=1,black=2,others=3)", ("1", "2","3"))
Norepinephrine = st.selectbox("Norepinephrine (No=0,Yes=1)", ("0","1"))
Dopamine = st.selectbox("Dopamine (No=0,Yes=1)", ("0","1"))
Phenylephrine = st.selectbox("Phenylephrine (No=0,Yes=1)", ("0","1"))
Vasopressin = st.selectbox("Vasopressin (No=0,Yes=1)", ("0","1"))
Vent = st.selectbox("Vent (No=0,Yes=1)", ("0","1"))
Intubated = st.selectbox("Intubated (No=0,Yes=1)", ("0","1"))
MC = st.selectbox("MC (No=0,Yes=1)", ("0","1"))
HepF = st.selectbox("HepF (No=0,Yes=1)", ("0","1"))


# If button is pressed
if st.button("Predict"):
    
    df = pd.read_csv("https://raw.githubusercontent.com/nkuwangkai/app-for-mortality-prediction/main/data3.csv",thousands=',')
    # Store inputs into dataframe
    
    X = pd.DataFrame([[Age, Temperature, RespiratoryRate, HeartRate, SBP, AG, BUN, MCHC, MCV, RDW, WBC, Race,
                       Norepinephrine, Dopamine, Phenylephrine, Vasopressin, Vent, Intubated, MC, HepF]],
                     columns=["Age", "Temperature", "RespiratoryRate", "HeartRate", "SBP", "AG", "BUN", "MCHC", "MCV",
                              "RDW", "WBC", "Race", "Norepinephrine", "Dopamine", "Phenylephrine", "Vasopressin",
                              "Vent", "Intubated", "MC", "HepF"])
    X[['Age','RespiratoryRate','HeartRate','SBP','AG','BUN','MCV']] = X[['Age','RespiratoryRate','HeartRate','SBP','AG','BUN','MCV']].astype(int)
    X[['Race','Norepinephrine','Dopamine','Phenylephrine','Vasopressin','Vent','Intubated','MC','HepF']] = X[['Race','Norepinephrine','Dopamine','Phenylephrine','Vasopressin','Vent','Intubated','MC','HepF']].astype(category)
    
    Y = df[["label"]]
    
    
    clf = XGBClassifier(objective='binary:logistic',
              booster='gbtree',
              colsample_bytree=0.558759,
              gamma=0.1477409,
              learning_rate=0.08694605,
              max_delta_step=8,
              max_depth=3,
              min_child_weight=37,
              n_estimators=92,
              subsample=0.6428299)
    
    clf.fit(X,Y)
    
    # Get prediction
    prediction = clf.predict(X)[0]
    prectionProbability = clf.predict_proba(X)

    # Output prediction
    st.text(f"survive/mortality probability {prectionProbability}")
    st.text(f"mortality prediction {prediction}")
