import pandas as pd
import numpy as np
import xgboost
from xgboost.sklearn import XGBClassifier
import streamlit as st

data3 = pd.read_csv("https://raw.githubusercontent.com/nkuwangkai/app-for-mortality-prediction/main/data3.csv",thousands=',')
Xtrain = (data3.iloc[:,1:22]) 
Ytrain = (data3.iloc[:,0])

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

clf.fit(Xtrain,Ytrain)

# Title
st.header("Machine learning app for in-hospital mortality prediction")

Age = st.number_input("Age (years)",step=1,min_value=0)
Temperature = st.number_input("Temperature (℃)",step=0.01,min_value=0)
RespiratoryRate = st.number_input("RespiratoryRate (breaths per minute)",step=1,min_value=0)
HeartRate = st.number_input("HeartRate (beats per minute)",step=1,min_value=0)
SBP = st.number_input("SBP (mmHg)",step=1,min_value=0)
AG = st.number_input("AG",step=1,min_value=0)
BUN = st.number_input("BUN (mg/dL)",step=1,min_value=0)
MCHC = st.number_input("MCHC (g/L)",step=0.01,min_value=0)
MCV = st.number_input("MCV (fL)",step=1,min_value=0)
RDW = st.number_input("RDW",step=0.01,min_value=0)
WBC = st.number_input("WBC (×109/L)",step=0.01,min_value=0)
Race = st.number_input("Race (white=1,black=2,others=3)",step=1,min_value=1,max_value=3)
Norepinephrine = st.number_input("Norepinephrine (No=0,Yes=1)",step=1,min_value=0,max_value=1)
Dopamine = st.number_input("Dopamine (No=0,Yes=1)",step=1,min_value=0,max_value=1)
Phenylephrine = st.number_input("Phenylephrine (No=0,Yes=1)",step=1,min_value=0,max_value=1)
Vasopressin = st.number_input("Vasopressin (No=0,Yes=1)",step=1,min_value=0,max_value=1)
Vent = st.number_input("Vent (No=0,Yes=1)",step=1,min_value=0,max_value=1)
Intubated = st.number_input("Intubated (No=0,Yes=1)",step=1,min_value=0,max_value=1)
MC = st.number_input("MC (No=0,Yes=1)",step=1,min_value=0,max_value=1)
HepF = st.number_input("HepF (No=0,Yes=1)",step=1,min_value=0,max_value=1)

# If button is pressed
if st.button("Predict"):

    # Store inputs into dataframe
    X = pd.DataFrame([[Age, Temperature, RespiratoryRate, HeartRate, SBP, AG, BUN, MCHC, MCV, RDW, WBC, Race,
                       Norepinephrine, Dopamine, Phenylephrine, Vasopressin, Vent, Intubated, MC, HepF]],
                     columns=["Age", "Temperature", "RespiratoryRate", "HeartRate", "SBP", "AG", "BUN", "MCHC", "MCV",
                              "RDW", "WBC", "Race", "Norepinephrine", "Dopamine", "Phenylephrine", "Vasopressin",
                              "Vent", "Intubated", "MC", "HepF"])

    # Get prediction
    prediction = clf.predict(X)[0]
    prectionProbability = clf.predict_proba(X)

    # Output prediction
    st.text(f"in-hospital survive/mortality probability [{prectionProbability}]")
    st.text(f"in-hospital mortality prediction [{prediction}]")
