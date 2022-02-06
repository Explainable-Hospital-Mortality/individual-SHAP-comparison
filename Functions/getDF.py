# IMPORT
from Functions.SQLfunc import getPatientsWithMoreThanOneICUStay, getExpiredPatients, getAlivePatients, getAPS, getAPACHE, getAPACHEPredVar, getAdx
from Functions.util import calculateBMI, calculateGCS, calculatePFRatio

import pandas as pd
import numpy as np


# FUNCTIONS
def mergeAPACHE(query_schema, conn, df, apacheVersion, apsPredVar):
    patientunitstayids = df['patientunitstayid'].values
    apsVar = getAPS(query_schema, conn, patientunitstayids)
    apacheResults = getAPACHE(query_schema, conn, patientunitstayids, 'IVa')
    apachePredVar = getAPACHEPredVar(query_schema, conn, patientunitstayids)

    df = pd.merge(df, apsVar, on='patientunitstayid', how='inner')
    df = pd.merge(df, apacheResults, on='patientunitstayid', how='inner')
    df = pd.merge(df, apachePredVar[apsPredVar + ['patientunitstayid']], on='patientunitstayid', how='left')

    for column in df.columns:
        df.loc[df[column] == -1, column] = np.nan

    return df

def removeOutliers(df): # zScore 5
    df = df.loc[~(df['pfratio'] >= 8.925)] #17
    df = df.loc[~(df['urine'] >= 9777.7152)] #99
    df = df.loc[~(df['wbc'] >= 53.31)] #150
    df = df.loc[~(df['temperature'] >= 41.5)] #178
    df = df.loc[~(df['temperature'] <= 31.3)]
    df = df.loc[~(df['sodium'] >= 167)] #132
    df = df.loc[~(df['sodium'] <= 109)]
    df = df.loc[~(df['ph'] <= 6.88)] #5
    df = df.loc[~(df['creatinine'] >= 9.8)] #386
    df = df.loc[~(df['albumin'] >= 6.6)] #1
    df = df.loc[~(df['pco2'] >= 103.2)] #33
    df = df.loc[~(df['bun'] >= 135.0)] #180
    df = df.loc[~(df['glucose'] >= 662.0)] #215
    df = df.loc[~(df['bilirubin'] >= 13.5)] #161
    df = df.loc[~(df['bmi'] >= 69.8)] #134

    return df

def getPatientsDF(conn, query_schema, startAge, endAge, minLoS, apsPredVar):
    moreThanOneStay = getPatientsWithMoreThanOneICUStay(conn, query_schema)
    df_e = getExpiredPatients(conn, query_schema, moreThanOneStay, startAge, endAge, minLoS)
    df_e = df_e.loc[df_e['hospitaldischargestatus'] == 'Expired']
    df_a = getAlivePatients(conn, query_schema, moreThanOneStay, startAge, endAge, minLoS)

    df_e = df_e.assign(status=1)
    df_a = df_a.assign(status=0)
    df = df_e
    df = df.append(df_a)

    calculateBMI(df)
    df.loc[df['gender'] == 'Male', 'gender'] = 1
    df.loc[df['gender'] == 'Female', 'gender'] = 0

    df = mergeAPACHE(query_schema, conn, df, 'IVa', apsPredVar)

    calculateGCS(df)
    calculatePFRatio(df)

    # Admission diagnosis
    adx = getAdx(conn, query_schema)
    adx['opNonOp'] = ''
    adx.loc[adx['operative'].isnull(), 'opNonOp'] = 0
    adx.loc[~adx['operative'].isnull(), 'adx'] = adx['operative']
    adx.loc[adx['nonOperative'].isnull(), 'opNonOp'] = 1
    adx.loc[~adx['nonOperative'].isnull(), 'adx'] = adx['nonOperative']
    df = pd.merge(df, adx[['patientunitstayid', 'opNonOp', 'adx']], on='patientunitstayid', how='left')

    print()
    print('Total: {}, A: {}, E: {} ({:.1f}%)\n'.format(len(df), len(df.loc[df['status']==0]), len(df.loc[df['status']==1]), len(df.loc[df['status']==1])/len(df)*100))

    # Remove patients with more than one stay
    df = df[~df['uniquepid'].isin(moreThanOneStay)]
    print('More than one stay: {}, A: {}, E: {} ({:.1f}%)\n'.format(len(df), len(df.loc[df['status']==0]), len(df.loc[df['status']==1]), len(df.loc[df['status']==1])/len(df)*100))

    # Age between startAge and endAge
    df.loc[df['age'] == '> 89', 'age'] = 89
    df['age'] = pd.to_numeric(df['age'])
    df = df.loc[df['age'].between(startAge, endAge)]
    print('Age 18+: {}, A: {}, E: {} ({:.1f}%)\n'.format(len(df), len(df.loc[df['status']==0]), len(df.loc[df['status']==1]), len(df.loc[df['status']==1])/len(df)*100))

    # Stay longer than minLoS (minLoS in min)
    df = df.loc[df['unitdischargeoffset'] > minLoS]
    print('LoS > 24h: {}, A: {}, E: {} ({:.1f}%)\n'.format(len(df), len(df.loc[df['status']==0]), len(df.loc[df['status']==1]), len(df.loc[df['status']==1])/len(df)*100))

    # Dropna
    df.dropna(subset=['age', 'gender', 'patientunitstayid', 'hospitaldischargestatus', 'unitdischargestatus', 'bmi', 'predictedhospitalmortality', 'opNonOp', 'adx'], inplace=True)
    print('Dropna: {}, A: {}, E: {} ({:.1f}%)\n'.format(len(df), len(df.loc[df['status']==0]), len(df.loc[df['status']==1]), len(df.loc[df['status']==1])/len(df)*100))

    # Remove outliers
    df = df.loc[df['admissionheight'].between(110,250, inclusive=False)] # 119 + 7 patients
    df = df.loc[df['admissionweight'].between(30,300, inclusive=False)] # ? + 5 patients
    df = df.loc[df['gender'].isin([1, 0])] # 6 patients ('Other', 'Unknown', '')
    df = removeOutliers(df)

    print('Remove outliers: {}, A: {}, E: {} ({:.1f}%)\n'.format(len(df), len(df.loc[df['status']==0]), len(df.loc[df['status']==1]), len(df.loc[df['status']==1])/len(df)*100))

    # Shuffle
    df = df.sample(frac=1, random_state=42)

    return df
