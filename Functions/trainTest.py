# IMPORT
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# FUNCTIONS
def dropNa(df, predictors, n_nan=0):
    return df.dropna(subset=predictors, thresh=len(predictors)-n_nan)


def getTrainTest(df, predictors, trainTestSplit):
    expired = list(df.loc[df['status'] == 1]['patientunitstayid'].values)
    alive = list(df.loc[df['status'] == 0]['patientunitstayid'].values)

    shareExpired = len(expired)/(len(expired)+ len(alive))

    trainNumberExpired = round(len(expired)*trainTestSplit)
    trainNumberAlive = round(trainNumberExpired*((len(df)/len(expired))-1))

    train_e_pid = expired[0:trainNumberExpired]
    test_e_pid = expired[trainNumberExpired:]

    train_a_pid = alive[0:trainNumberAlive]

    test_a_pid = alive[trainNumberAlive:(trainNumberAlive + round(len(test_e_pid)/(len(expired)/len(alive))))]
    a_rest = alive[trainNumberAlive + round(len(test_e_pid)/(len(expired)/len(alive))):]

    train = df[df['patientunitstayid'].isin(train_e_pid + train_a_pid)]
    train = train.sample(frac=1, random_state=42)
    test = df[df['patientunitstayid'].isin(test_e_pid + test_a_pid)]
    test = test.sample(frac=1, random_state=42)
    rest = df[df['patientunitstayid'].isin(a_rest)]
    rest = rest.sample(frac=1, random_state=42)

    return train, test, rest


def fillNa(train, test, rest, predictors):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(train[predictors])
    train[predictors] = pd.DataFrame(imp_mean.transform(train[predictors]), index=train[predictors].index, columns=train[predictors].columns)
    test[predictors] = pd.DataFrame(imp_mean.transform(test[predictors]), index=test[predictors].index, columns=test[predictors].columns)
    rest[predictors] = pd.DataFrame(imp_mean.transform(rest[predictors]), index=rest[predictors].index, columns=rest[predictors].columns)

    return train, test, rest
