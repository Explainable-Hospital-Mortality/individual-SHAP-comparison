# IMPORT
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, roc_curve, auc, precision_recall_curve, roc_auc_score

# FUNCTIONS
def printResults(y_test, prob, y_pred):
    # Print report
    print(classification_report(y_test, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() # If binary

    precision, recall, _ = precision_recall_curve(y_test, prob)
    auprc = auc(recall, precision)*100
    TN,FP,FN,TP = confusion_matrix(y_test, y_pred).ravel()
    specificity = TN/(TN+FP)*100
    sensitivity = TP/(TP+FN)*100
    PPV = TP/(TP+FP)*100
    NPV = TN/(TN+FN)*100

    print('---')
    kappa = cohen_kappa_score(y_test, y_pred)
    print('Cohen\'s kappa: '+ str(kappa))
    print('---')

    print('True positive: ' + str(tp))
    print('True negative: ' + str(tn))
    print('False positive: ' + str(fp))
    print('False negative: ' + str(fn))

    fpr, tpr, threshold = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)*100
    print()
    print('AUROC: %.1f' % roc_auc)
    print('AUPRC: %.1f' % auprc)
    print('Specificity: %.1f' % specificity)
    print('Sensitivity: %.1f' % sensitivity)
    print('PPV: %.1f' % PPV)
    print('NPV: %.1f' % NPV)
    print()
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    print('Optimal threshold: ' + str(optimal_threshold))

    return roc_auc


def printConfusionMatrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    df_cm = pd.DataFrame(cm, index = [i for i in ['expired', 'alive']],
                  columns = [i for i in ['expired', 'alive']])
    plt.figure(figsize = (5,4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu").set_ylim([0,2])
    plt.show()


def getCalibrationCurve(X, y, model, n_bins=10):
    binned_true_p, binned_predict_p = calibration_curve(y, model.predict_proba(X)[:,1], n_bins=n_bins)
    plt.scatter(binned_predict_p, binned_true_p)
    plt.title('Calibration curves')
    plt.ylabel('True')
    plt.xlabel('Predicted')


def getCalibrationCurves(models, test, predictors, n_bins=10, saveName=False):
    X_test = test[predictors]
    y_test = test['status']

    plt.figure(0).clf()

    for modelName in models:
        binned_true_p, binned_predict_p = calibration_curve(y_test, models[modelName].predict_proba(X_test)[:,1], n_bins=n_bins)
        plt.plot(binned_predict_p, binned_true_p, label=modelName)

    binned_true_p, binned_predict_p = calibration_curve(y_test, test['predictedhospitalmortality'].values, n_bins=n_bins)
    plt.plot(binned_predict_p, binned_true_p, label='APACHE')

    plt.plot([0,1], [0,1], '--', label='Perfect calibrated model')

    plt.title('Calibration curve')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.legend(loc=0)

    if saveName:
        plt.savefig(saveName)
    else:
        plt.show()


def plotROC(actual, predicted):
    fpr, tpr, threshold = roc_curve(actual, predicted)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    return roc_auc


def plotROCs(models, test, predictors, saveName=False):
    X_test = test[predictors]
    y_test = test['status']

    plt.figure(0).clf()

    for modelName in models:
        prob = models[modelName].predict_proba(X_test)[:,1]
        label = y_test
        fpr, tpr, thresh = roc_curve(label, prob)
        auc = roc_auc_score(label, prob)
        plt.plot(fpr,tpr,label='%s, AUC=%.3f' % (modelName, (auc)))

    prob = test['predictedhospitalmortality'].values
    label = np.array(test['actualhospitalmortality'].values).astype(int)
    fpr, tpr, thresh = roc_curve(label, prob)
    auc = roc_auc_score(label, prob)
    plt.plot(fpr,tpr,label='APACHE, AUC=%.3f' % (auc))

    plt.plot([0, 0, 1], [0, 1, 1], label='Perfect discriminative model, AUC=1')

    plt.title('Receiver operating characteristic curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc=0)
    if saveName:
        plt.savefig(saveName)
    else:
        plt.show()


def plotHistogram(df, variableName, title, saveName=False):
    plt.hist(df.loc[df['status'] == 0][variableName], bins=20, density=True, color=['blue'], alpha=1, label=['Alive'])
    plt.hist(df.loc[df['status'] == 1][variableName], bins=20, density=True, color=['orange'], alpha = 0.9, label=['Expired'])
    plt.xlabel(variableName)
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    if saveName:
        plt.savefig(saveName)
    plt.show()
