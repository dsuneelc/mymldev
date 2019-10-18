import numpy as np
import pandas as pd
from sklearn import datasets

from ..metrics.classification import ConfusionMatrix
from ..model.classifier import XGBClassifier
from ..utils.data import XYData


def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    return X, y, feature_names


def test_xgbclassifier():
    X, y, feature_names = load_data()
    y_mod = np.where(y == 1, 1, 0)
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.Series(y_mod, name='Species')
    print(y_df.value_counts())
    xgbc = XGBClassifier()
    xgbc.fit(X_df, y_df)
    print(xgbc.feature_importance_)
    xgbc_metric = xgbc.classification_metric(X_df, y_df,
                                             labels={1: 'Virginica', 0: 'Non-Virginica'})
    print(f'AUC: {xgbc_metric.auc_}')
    print(f'Gains table: {xgbc_metric.gains_table_}')
    for threshold in [0.3, 0.5, 0.8]:
        print(f'Metrics At threshold: {threshold}')
        cfm_threshold = xgbc_metric.confusion_matrix(threshold)
        print(f'Fancy Confusion Matrix: \n{xgbc_metric.cfm}')
        print(f'Confusion Matrix table: \n{cfm_threshold.table_}')
        print(f'F1-score: {cfm_threshold.f1_score_}')
        print(f'Accuracy: {cfm_threshold.accuracy_}')
        print(f'Precision: {cfm_threshold.precision_}')
        print(f'Recall: {cfm_threshold.recall_}')
        print(f'False Omission Rate: {cfm_threshold.false_omission_rate_}')
        print(f'False Discovery Rate: {cfm_threshold.false_discovery_rate_}')
        print(f'False Negative Rate: {cfm_threshold.false_negative_rate_}')
        print(f'False Positive Rate: {cfm_threshold.false_positive_rate_}')
        print(f'Negative Predictive Value: {cfm_threshold.negative_predictive_value_}')
        print(f'Specificity: {cfm_threshold.specificity_}')
        print(f'True Positives: {cfm_threshold.tp_}')
        print(f'False Positives: {cfm_threshold.fp_}')
        print(f'True Negatives: {cfm_threshold.tn_}')
        print(f'False Negatives: {cfm_threshold.fn_}')
    print(f'Plot precision recall curve: {xgbc_metric.plot_precision_recall_curve_}')
    print(f'Plot ROC: {xgbc_metric.plot_roc_curve_}')


def test_confusionmatrix():
    y_true_binary = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    y_pred_binary = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1])
    y_true_multiclass = np.array([1, 0, 2, 0, 1, 2, 1, 0, 2, 0, 1, 2, 1])
    y_pred_multiclass = np.array([1, 1, 2, 0, 0, 1, 1, 0, 2, 1, 1, 0, 0])

    def cfm(y_true, y_pred):
        cfm = ConfusionMatrix(y_true, y_pred)
        print(f'Confusion Matrix: {cfm.table_}')
        print(f'False positive: {cfm.fp_}')
        print(f'False negative: {cfm.fn_}')
        print(f'True positive: {cfm.tp_}')
        print(f'True negative: {cfm.tn_}')
        print(f'Recall: {cfm.recall_}')
        print(f'Specificity: {cfm.specificity_}')
        print(f'Precision: {cfm.precision_}')
        print(f'Negative predictive value: {cfm.negative_predictive_value_}')
        print(f'False positive rate: {cfm.false_positive_rate_}')
        print(f'False Negative rate: {cfm.false_negative_rate_}')
        print(f'False Discovery rate: {cfm.false_discovery_rate_}')
        print(f'False Omission rate: {cfm.false_omission_rate_}')
        print(f'Accuracy: {cfm.accuracy_}')
    cfm(y_true_binary, y_pred_binary)
    cfm(y_true_multiclass, y_pred_multiclass)


def test_xydata():
    X, y, feature_names = load_data()
    data = pd.DataFrame(X, columns=feature_names)
    data['species_type'] = y
    data['tmp_col'] = 'tmp'
    test = XYData(data, x_cols=feature_names, y_col='species_type')
    assert set(feature_names) & set(test.X.columns) == set(feature_names)
    assert test.y.name == 'species_type'
    assert set(test.excluded_X.columns) == set(data.columns[~data.columns.isin(feature_names + ['species_type'])])


if __name__ == '__main__':
    # print(f'Testing Confusion Matrix:\n{test_confusionmatrix()}')
    print(f'Testing XGB Classifier:\n{test_xgbclassifier()}')
