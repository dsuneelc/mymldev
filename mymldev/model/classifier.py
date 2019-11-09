from typing import Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from ..metrics import BinaryClassificationMetrics, MultiClassClassificationMetrics

__all__ = ["MXGBClassifier"]


SEED = 1310


class MXGBClassifier(xgb.XGBClassifier):
    """XGBClassifier
    """

    def get_feature_importance(self, importance_type="weight", normalized=False):
        """XGBoost Classifier feature importance
        """
        imp_vals = self.get_booster().get_score(importance_type=importance_type)
        fimp = {
            feature: float(imp_vals.get(feature, 0.0))
            for i, feature in enumerate(self.get_booster().feature_names)
        }
        total = sum(fimp.values())
        if normalized:
            fimp = {feature: (imp / total) for feature, imp in fimp.items()}
        fimp_sorted = sorted(fimp.items(), key=lambda kv: kv[1], reverse=True)
        fimp_df = pd.DataFrame(fimp_sorted, columns=["IDV", "Importance"])
        return fimp_df

    def classification_metric(
        self, X: pd.DataFrame, y: pd.Series, labels: Optional[dict] = None
    ) -> Union[BinaryClassificationMetrics, MultiClassClassificationMetrics]:
        """Classification Metrics

        Parameters
        ----------
        X : pandas.DataFrame, (n_samples, n_classes)
            Independent variables.
        y: pandas.Series, (n_samples,)
            Dependent variable.
        labels: Optional[dict]
            Labels for y.

        Returns
        -------
        Union[BinaryClassificationMetrics, MultiClassClassificationMetrics] object
            Classificaton metrics.
        """
        X = X.copy()
        y = y.copy()
        try:
            self.predict(X.iloc[:1])
        except xgb.core.XGBoostError as e:
            print(f"{e!r}")
        if len(np.unique(y)) == 2:
            metrics = BinaryClassificationMetrics(self, X, y, labels=labels)
        else:
            # TODO: MultiClassClassificationMetrics
            raise NotImplementedError("Multi-class classification metric not implemented.")
        return metrics


class LogisticRegression:
    pass
