"""Classification metrics.
"""


__all__ = ["ConfusionMatrix", "BinaryClassificationMetrics", "MultiClassClassificationMetrics"]


from typing import Optional
import numpy as np
import pandas as pd
import sklearn.metrics as mt
import matplotlib as mpl

from ..visualization.metrics import plot_precision_recall, plot_roc


class ConfusionMatrix:
    """Confusion matrix and it's derived metrics.

    Parameters
    ----------
    y : numpy.ndarray
        True values.
    y_hat : numpy.ndarray
        Predicted values.
    """

    def __init__(self, y: np.ndarray, y_hat: np.ndarray) -> None:
        self.y = y
        self.y_hat = y_hat
        self.__binary_y = self.__is_y_binary()
        self._cfm = np.array([])
        self._fp = np.array([])
        self._fn = np.array([])
        self._tp = np.array([])
        self._tn = np.array([])

    def __is_y_binary(self) -> bool:
        """bool: Check whether dependent variable is binary or not."""
        return len(np.unique(self.y_hat)) == 2

    def __cfm_metric(self, value: np.ndarray) -> np.ndarray:
        """Computes metric based on classification type.

        Parameters
        ----------
        value : numpy.ndarray
            Metric.

        Returns
        -------
        numpy.ndarray
            Returns only required value in case of
            binary classification, assuming 'target' is '1'.
            As metrics are calculated for both '0' and '1'.
        """
        if self.__binary_y:
            return value[1]
        return value

    @property
    def table_(self) -> np.ndarray:
        """numpy.ndarray: Confusion Matrix."""
        if not self._cfm.size:
            self._cfm = mt.confusion_matrix(self.y, self.y_hat)
        return self._cfm

    @property
    def fn_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        False Negatives.
        """
        if not self._fn.size:
            fn = self.table_.sum(axis=1) - np.diag(self.table_)
            self._fn = self.__cfm_metric(fn)
        return self._fn

    @property
    def fp_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        False Positives.
        """
        if not self._fp.size:
            fp = self.table_.sum(axis=0) - np.diag(self.table_)
            self._fp = self.__cfm_metric(fp)
        return self._fp

    @property
    def tn_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        True Negatives.
        """
        if not self._tn.size:
            self._tn = self.table_.sum() - (self.fp_ + self.fn_ + self.tp_)
        return self._tn

    @property
    def tp_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        True Positives.
        """
        if not self._tp.size:
            tp = np.diag(self.table_)
            self._tp = self.__cfm_metric(tp)
        return self._tp

    @property
    def accuracy_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        Accuracy.
        """
        return (self.tp_ + self.tn_) / (self.tp_ + self.tn_ + self.fp_ + self.fn_)

    @property
    def false_discovery_rate_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        False Discovery Rate.
        """
        return self.fp_ / (self.fp_ + self.tp_)

    @property
    def false_negative_rate_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        False Negative Rate or Miss Rate.
        """
        return self.fn_ / (self.fn_ + self.tp_)

    @property
    def false_omission_rate_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        False Omission Rate.
        """
        return self.fn_ / (self.fn_ + self.tn_)

    @property
    def false_positive_rate_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        False Positive Rate or Fall-out.
        """
        return self.fp_ / (self.fp_ + self.tn_)

    @property
    def f1_score_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        F1 Score.
        """
        return (2 * self.tp_) / (2 * self.tp_ + self.fp_ + self.fn_)

    @property
    def negative_predictive_value_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        Negative Predictive Value.
        """
        return self.tn_ / (self.tn_ + self.fn_)

    @property
    def precision_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        Precision or Positive Predictive Value.
        """
        return self.tp_ / (self.tp_ + self.fp_)

    @property
    def recall_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        Sensitivity, Recall, Hit rate, or True Positive Rate.
        """
        return self.tp_ / (self.tp_ + self.fn_)

    @property
    def specificity_(self) -> float:
        """float, for binary classification else numpy.ndarray:
        Specificity, Selectivity, or True Negative Rate.
        """
        return self.tn_ / (self.tn_ + self.fp_)

    def __getattribute__(self, item):
        if item in {
            "recall_",
            "specificity_",
            "precision_",
            "negative_predictive_value_",
            "false_negative_rate_",
            "false_positive_rate_",
            "false_discovery_rate_",
            "false_omission_rate_",
            "accuracy_",
            "f1_score_",
        }:
            value = super(ConfusionMatrix, self).__getattribute__(item)
            # modifies the value of the attribute when called
            return np.round(value * 100, 2)
        return super(ConfusionMatrix, self).__getattribute__(item)


class BinaryClassificationMetrics:
    """ Binary Classification Metrics.

    Parameters
    ----------
    y : pandas.Series
        DV.
    predicted_proba : Union[pandas.DataFrame, pd.Series]
        predicted probabilities from IDVs based on given model.
    labels : Optional[dict]
        labels for y, eg: {0: 'negative', 1: 'positive'}
        (the default is, {0: 0, 1: 1}).

    Attributes
    ----------
    target_proba : numpy.ndarray
        predicted probabilities from IDVs based on given model,
        assuming 'target' is '1'.
    cfm : pandas.DataFrame
        Confusion Matrix.
    confusion_matrix : :obj:`ConfusionMatrix`
        Confusion Matrix object.
    """

    def __init__(self, y: pd.Series, predicted_proba: Union[pd.DataFrame, pd.Series], labels: Optional[dict] = None) -> None:
        self.y = y
        self.labels = labels
        self.predicted_proba = predicted_proba
        self.target_proba = self.predicted_proba[:, 1] if isinstance(predicted_proba, pd.DataFrame) else predicted_proba
        self.cfm = pd.DataFrame()
        self.confusion_matrix = ConfusionMatrix(self.y.values, self.y_hat)
        self._gains_table = pd.DataFrame()
        self._threshold: float = 0.5
        self._y_hat = None
        self.__fpr, self.__tpr, _ = mt.roc_curve(self.y.values, self.target_proba)

    @property
    def auc_(self) -> float:
        """float: Area Under the Curve."""
        return mt.auc(self.__fpr, self.__tpr)

    @property
    def gains_table_(self) -> pd.DataFrame:
        """`pandas.DataFrame`: Gains table."""
        if self._gains_table.empty:
            df = pd.DataFrame()
            df["y"] = self.y.values
            df["pred_prob"] = self.target_proba
            try:
                df["Decile"] = pd.qcut(df["pred_prob"], 10, labels=False)
            except ValueError:
                df["Decile"] = pd.qcut(df["pred_prob"].rank(method="first"), 10, labels=False)
            df = df.rename_axis("unique_id").reset_index()
            lift_df = (
                df.groupby(["Decile", "y"])["unique_id"]
                .count()
                .unstack("y")
                .sort_index(ascending=False)
            )
            lift_df = lift_df.fillna(0).astype(int)
            lift_df.index = np.arange(10)
            gains_df = pd.DataFrame()
            kwargs = {
                "Decile": lift_df.index + 1,
                "No. of Observations": lift_df.sum(axis=1),
                "Number of Targets": lift_df[1],
                "Cumulative Targets": lift_df[1].cumsum(),
                "% of Targets": lift_df[1] / lift_df[1].sum() * 100,
                "Gain": lift_df[1].cumsum() / lift_df[1].sum() * 100,
                "Random Targets": lift_df[1].sum() / 10,
            }
            gains_df = gains_df.assign(**kwargs)
            gains_df["Lift"] = lift_df[1] / gains_df["Random Targets"]
            gains_df["Cumulative Lift"] = gains_df["Lift"].cumsum()
            # gains_df['Cumulative Lift'] = (gains_df['Cumulative Targets'] /
            #                                (lift_df[1].sum() * (lift_df.index + 1) / 10))
            gains_sum = pd.DataFrame({}, columns=gains_df.columns, index=[0])
            gains_sum["Decile"] = "Total"
            gains_sum["No. of Observations"] = gains_df["No. of Observations"].sum()
            gains_sum["Number of Targets"] = gains_df["Number of Targets"].sum()
            gains_table = pd.concat([gains_df, gains_sum], ignore_index=True)
            gains_table.fillna("", inplace=True)
            self._gains_table = gains_table
        return self._gains_table

    @property
    def gini_coefficient_(self) -> float:
        """float: Gini Coefficient"""
        return (self.auc_ - 0.5) / 0.5

    @property
    def lift_score_(self) -> float:
        """float: Lift score."""
        return self.gains_table_["Lift"].iloc[0]

    @property
    def plot_cumulative_gain_(self) -> mpl.axes.Axes:
        pass

    @property
    def plot_ks_statistic_(self) -> mpl.axes.Axes:
        pass

    @property
    def plot_lift_curve_(self) -> mpl.axes.Axes:
        pass

    @property
    def plot_precision_recall_(self) -> mpl.axes.Axes:
        """`matplotlib.axes.Axes`: Plots Precision-Recall curve."""
        ax = plot_precision_recall(
            self.y, self.predicted_proba, labels=self.labels, classes_to_plot=[1]
        )
        return ax

    @property
    def plot_roc_(self) -> mpl.axes.Axes:
        """`matplotlib.axes.Axes`: Plots ROC curve."""
        ax = plot_roc(self.y, self.predicted_proba, labels=self.labels, classes_to_plot=[1])
        return ax

    @property
    def threshold(self) -> float:
        """float: Threshold chosen.

        Set the threshold, else it defaults to 0.5.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        if not (0 < threshold < 1):
            raise ValueError("Invalid 'threshold'. It should be in range of (0, 1)")
        self._threshold = threshold
        self.confusion_matrix = ConfusionMatrix(self.y.values, self.y_hat)
        if not self.labels:
            self.labels = {}
        label_0 = self.labels.get(0, 0)
        label_1 = self.labels.get(1, 1)
        self.cfm = (
            pd.DataFrame(
                self.confusion_matrix.table_, columns=[label_0, label_1], index=[label_0, label_1]
            )
            .rename_axis(f"threshold ({self.threshold})")
            .reset_index()
        )

    @property
    def y_hat(self) -> np.ndarray:
        """`numpy.ndarray`: Predicted values."""
        self._y_hat = (self.target_proba > self.threshold).astype(bool)
        return self._y_hat


class MultiClassClassificationMetrics:
    pass
