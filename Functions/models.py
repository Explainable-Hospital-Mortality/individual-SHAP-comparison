# IMPORT
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import numpy as np

# FUNCTIONS
def modelParamSwitch(modelName):
    switcher = {
        'LR': {'LR': {
                    'model': LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42),
                    'params': {
                        #'model__penalty': ['l1', 'l2'],
                        'model__tol': [1e-5, 1e-3, 1e-2],
                        'model__C': [0.1, 0.5, 1.0, 2.0],
                        'model__fit_intercept': [True, False],
                        'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                        'model__max_iter': [50, 100, 250, 500]
                    }}},
        'RF': {'RF': {
                    'model': RandomForestClassifier(class_weight="balanced", n_jobs=-1), #, n_estimators=100
                    'params': {
                        'model__bootstrap': [True, False], #, False
                        'model__max_depth': [None, 5, 10, 15], #5, 100,
                        'model__max_features': ['sqrt', 'auto', None], #'auto',
                        'model__min_samples_leaf': [i for i in range(1, 15)], #4
                        'model__min_samples_split': [5, 10, 15], #5
                        'model__n_estimators': [i for i in range(100, 5000)]  #200, 2000
                    }}},
        'ADA': {'ADA': {
                    'model': AdaBoostClassifier(random_state=42),
                    'params': {
                        'model__n_estimators': [100, 250, 500]
                    }}},
        'NB': {'NB': {
                    'model': GaussianNB(),
                    'params': {
                        'model__var_smoothing': np.logspace(0,-9, num=100)
                    }}}
    }
    return switcher.get(modelName, "Invalid model")


def getModelScores(model_params, X_train, y_train, scoring='accuracy', randomized=True, n_iter=5):
    scores = []

    for model_name, mp in model_params.items():
        if randomized:
            clf = RandomizedSearchCV(mp['model'], mp['params'], n_jobs=-1, cv=5, scoring=scoring, random_state=42, n_iter=n_iter)
        else:
            clf = GridSearchCV(mp['model'], mp['params'], n_jobs=-1, cv=5, scoring=scoring)

        clf.fit(X_train, y_train)
        scores.append({'model': model_name, 'best_score': clf.best_score_, 'best_params': clf.best_params_})

    return scores


def getBestParams(scores, modelName):
    for i in scores:
        if i['model'] == modelName:
            return i['best_params']

    print(modelName + ' does not exist')
    return None


def getPreprocessor(numerical, categorical, X_train):
    cat_pipe = Pipeline([('encoder', OneHotEncoder(sparse=False, handle_unknown="ignore"))])
    num_pipe = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean'))])
    preprocessor = ColumnTransformer(transformers=[('cat', cat_pipe, categorical), ('num', num_pipe, numerical)])
    preprocessor.fit(X_train)
    return preprocessor


def getModel(modelName, X_train, y_train, preprocessor, scoring, randomized=True, n_iter=5):
    gridSearchScores, model_params = getGridSearchScores(X_train, y_train, preprocessor, scoring, modelName, randomized=randomized, n_iter=n_iter)
    bestModel = getBestModel(modelName, gridSearchScores, preprocessor, model_params, X_train, y_train)
    return bestModel


def getBestModel(modelName, gridSearchScores, preprocessor, model_params, X_train, y_train):
    bestModel = Pipeline([("preprocessor", preprocessor), ("model", model_params[modelName]['model'])])
    bestModel.set_params(**getBestParams(gridSearchScores, modelName))
    bestModel.fit(X_train, y_train)
    return bestModel


def gridSearch(model_params, preprocessor, X_train, y_train, scoring='accuracy', randomized=True, n_iter=5):
    scores = []

    for model_name, mp in model_params.items():
        pipe = Pipeline([("preprocessor", preprocessor),
                         ("model", mp['model'])])
        if randomized:
            clf = RandomizedSearchCV(pipe, mp['params'], n_jobs=-1, cv=5, scoring=scoring, random_state=42, n_iter=n_iter)
        else:
            clf = GridSearchCV(pipe, mp['params'], n_jobs=-1, cv=5, scoring=scoring)

        clf.fit(X_train, y_train)
        scores.append({'model': model_name, 'best_score': clf.best_score_, 'best_params': clf.best_params_})

    return scores


def getGridSearchScores(X_train, y_train, preprocessor, scoring, modelName, randomized=True, n_iter=5):
    model_params = modelParamSwitch(modelName)
    gridSearchScores = gridSearch(model_params, preprocessor, X_train, y_train, scoring=scoring, randomized=randomized, n_iter=n_iter)
    return gridSearchScores, model_params
