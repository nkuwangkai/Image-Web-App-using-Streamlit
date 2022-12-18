import pandas as pd
import numpy as np
import xgboost
from xgboost.sklearn import XGBClassifier
import streamlit as st

# Title
st.header("Machine learning app for in-hospital mortality prediction")

Age = st.int_input("Age (years)") 
Temperature = st.number_input("Temperature (℃)")
RespiratoryRate = st.int_input("RespiratoryRate (breaths per minute)")
HeartRate = st.int_input("HeartRate (beats per minute)")
SBP = st.int_input("SBP (mmHg)")
AG = st.int_input("AG")
BUN = st.int_input("BUN (mg/dL)")
MCHC = st.number_input("MCHC (g/L)")
MCV = st.int_input("MCV (fL)")
RDW = st.number_input("RDW")
WBC = st.number_input("WBC (×109/L)")

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
