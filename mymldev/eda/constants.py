from ..utils import ConstanstHome


edac = ConstanstHome()

edac.allowed_types.NUMERICAL = "numerical"
edac.allowed_types.CATEGORICAL = "categorical"
edac.problem.CLASSIFICATION = "classification"
edac.problem.REGRESSION = "regression"
edac.datasource.PANDAS_DATAFRAME = "pandas_dataframe"
edac.datasource.HIVE_TABLE = "hive_table"
edac.anova.CATEGORICAL = "anova_categorical"
edac.anova.COUNT_PER_CLASS = "anova_count_per_class"
edac.anova.MEAN_PER_CLASS = "anova_mean_per_class"
edac.anova.VARIANCE_PER_CLASS = "anova_variance_per_class"
edac.anova.DF_GROUP = "anova_df_group"
edac.anova.DF_ERROR = "anova_df_error"
edac.MISSING = "MISSING"
edac.VALUE_COUNTS = "VALUE_COUNTS"

