__all__ = ["cramers_corrected_stat"]


import numpy as np
import pandas as pd
import scipy.stats as ss


def cramers_corrected_stat(contingency_table: pd.DataFrame) -> float:
    """Calculate bias corrected cramer's V

    Parameters
    ----------
    contingency_table : pandas.DataFrame
        Contingency table of two nominal variables

    Returns
    -------
    float
        Cramer's V value for the input nominal variables

    Examples
    --------
    >>> import pandas as pd
    >>> from mymldev.eda import cramers_corrected_stat
    >>> df = pd.read_csv("titanic_train.csv")
    >>> cramers_corrected_stat(
    ...     pd.crosstab(df["Pclass"], df["Survived"])
    ... )
    0.33668387622245516

    Notes
    -----
    Cramer's V is the measure of association between two
    nominal variables, giving a value between 0 and +1
    '0' corresponds to no association
    '1' corresponds to complete association, value can reach 1
    only when two variables are equal to each other.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    """
    chi2 = ss.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1) / (n - 1)))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    cramers_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    return cramers_v
