import pandas as pd
import numpy as np
import xgboost
from xgboost.sklearn import XGBClassifier
import streamlit as st

data3 = pd.read_csv("https://raw.githubusercontent.com/nkuwangkai/app-for-mortality-prediction/main/data3.csv",thousands=',',encoding='GBK')
Xtrain = (data3.iloc[:,1:18]) 
Ytrain = (data3.iloc[:,0])

clf = XGBClassifier(objective='binary:logistic',
              booster='gbtree',
              colsample_bytree=0.6447264,
              gamma=0.1934144,
              learning_rate=0.1527683,
              max_delta_step=10,
              max_depth=3,
              min_child_weight=10,
              n_estimators=95,
              subsample=0.6293208)

clf.fit(Xtrain,Ytrain)

# Title
st.header("Machine learning app for in-hospital mortality prediction")								

age = st.number_input("age (years)",step=1,min_value=0)
diabetes = st.number_input("diabetes (No=0,Yes=1)",step=1,min_value=0,max_value=1)
SevereLiverDisease = st.number_input("severe liver disease (No=0,Yes=1)",step=1,min_value=0,max_value=1)
myocardial_infarct = st.number_input("myocardial infarction (No=0,Yes=1)",step=1,min_value=0,max_value=1)
SOFA = st.number_input("SOFA (scores)",step=1,min_value=0)
APSⅢ = st.number_input("APS Ⅲ (scores)",step=1,min_value=0)
RespiratoryRate = st.number_input("respiratory rate (breaths per minute)",step=1,min_value=0)
mchc = st.number_input("mchc (%)",step=0.1,min_value=0)
rdw = st.number_input("rdw (%)",step=0.1,min_value=0)
TotalBilirubin = st.number_input("total bilirubin (mg/dL)",step=0.1,min_value=0)
dopamine = st.number_input("dopamine (No=0,Yes=1)",step=1,min_value=0,max_value=1)
dobutamine = st.number_input("dobutamine (No=0,Yes=1)",step=1,min_value=0,max_value=1)
phenylephrine = st.number_input("phenylephrine (No=0,Yes=1)",step=1,min_value=0,max_value=1)
epinephrine = st.number_input("epinephrine (No=0,Yes=1)",step=1,min_value=0,max_value=1)
vasopressin = st.number_input("vasopressin (No=0,Yes=1)",step=1,min_value=0,max_value=1)
blocker = st.number_input("beta blocker (No=0,Yes=1)",step=1,min_value=0,max_value=1)
ventilation = st.number_input("ventilation (No=0,Yes=1)",step=1,min_value=0,max_value=1)

# If button is pressed
if st.button("Predict"):

    # Store inputs into dataframe
    X = pd.DataFrame([[age, diabetes, SevereLiverDisease, myocardial_infarct, SOFA, APSⅢ, RespiratoryRate, mchc, rdw, TotalBilirubin,
                       dopamine, dobutamine, phenylephrine, epinephrine, vasopressin, blocker, ventilation]],
                     columns=["age", "diabetes", "SevereLiverDisease", "myocardial_infarct", "SOFA", "APSⅢ", "RespiratoryRate", "mchc", "rdw", "TotalBilirubin",
                              "dopamine", "dobutamine", "phenylephrine", "epinephrine", "vasopressin", "blocker", "ventilation"])

    # Get prediction
    prediction = clf.predict(X)[0]
    prectionProbability = clf.predict_proba(X)

    # Output prediction
    st.text(f"in-hospital survive/mortality probability [{prectionProbability}]")
    st.text(f"in-hospital mortality prediction [{prediction}] (0=survive, 1=mortality")
