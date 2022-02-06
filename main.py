from Functions.connection import conn
from Functions.getDF import getPatientsDF
from Functions.trainTest import dropNa, getTrainTest, fillNa
from Functions.models import getModel, getPreprocessor
from Functions.printAndPlot import printResults, printConfusionMatrix, getCalibrationCurve, plotROC, plotROCs
import numpy as np
#import time
import shap
import pandas as pd


conn, query_schema = conn()

minAge = 18
maxAge = 100
minLoS = 1440

apsPredVar = ['thrombolytics', 'aids', 'hepaticfailure', 'lymphoma', 'metastaticcancer', 'leukemia', 'immunosuppression', 'cirrhosis', 'electivesurgery']
aps = ['intubated', 'vent', 'dialysis', 'meds'] + ['GCS', 'urine', 'wbc', 'temperature', 'respiratoryrate', 'sodium', 'heartrate', 'meanbp', 'ph', 'hematocrit', 'creatinine', 'albumin', 'pco2', 'bun', 'glucose', 'bilirubin']

numerical = ['age', 'bmi', 'gender', 'hospitaladmitoffset', 'pfratio', 'opNonOp'] + aps + apsPredVar
categorical = ['adx', 'unitadmitsource']

predictors = numerical + categorical

df = getPatientsDF(conn, query_schema, minAge, maxAge, minLoS, apsPredVar)

trainTestSplit = 0.75

scoring = 'roc_auc'

train, test, rest = getTrainTest(df, predictors, trainTestSplit)
print('Train: {}, Test: {}, Rest:{}'.format(len(train), len(test), len(rest)))
    
X_train = train[predictors].copy()
X_test = test[predictors].copy()
y_train = train['status'].copy()
y_test = test['status'].copy()

preprocessor = getPreprocessor(numerical, categorical, X_train)

cat_columns = preprocessor.named_transformers_['cat']['encoder'].get_feature_names(categorical)
X_train_transformed = pd.DataFrame(preprocessor.transform(X_train), columns=np.append(cat_columns, numerical))
X_test_transformed = pd.DataFrame(preprocessor.transform(X_test), columns=np.append(cat_columns, numerical))

modelNames = ['RF', 'ADA', 'LR', 'NB']
models = {}

for modelName in modelNames:
    model = getModel(modelName, X_train, y_train, preprocessor, scoring, randomized=True, n_iter=5)
    models[modelName] = model

# Get SHAP values
explainers = {}
shapValues = {}

testShap = 1000
explainers['NB'] = shap.KernelExplainer(models['NB']['model'].predict_proba, shap.sample(X_train_transformed, testShap))
shapValues['NB'] = explainers['NB'].shap_values(X_test_transformed)

treeModels = ['RF', 'ADA']
for treeModel in treeModels:
    explainers[treeModel] = shap.TreeExplainer(models[treeModel]['model'])
    shapValues[treeModel] = explainers[treeModel].shap_values(X_test_transformed)

explainers['LR'] = shap.LinearExplainer(models['LR']['model'], X_train_transformed, feature_perturbation="correlation_dependent")
shapValues['LR'] = explainers['LR'].shap_values(X_test_transformed)
