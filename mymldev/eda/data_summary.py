import abc
from .constants import edac


class DataSummarizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_anova_summary(self, categorical_column, numerical_column):
        raise NotImplementedError

    @abc.abstractmethod
    def get_categorical_aggregation(self, categorical_columns):
        raise NotImplementedError

    @abc.abstractmethod
    def get_correlation_data(self, numerical_columns):
        raise NotImplementedError

    @abc.abstractmethod
    def get_numerical_summary(self, numerical_columns):
        raise NotImplementedError

    @abc.abstractmethod
    def get_categorical_summary(self, categorical_columns):
        raise NotImplementedError

    @abc.abstractmethod
    def get_histogram_data(self, numerical_column, num_bins):
        raise NotImplementedError

    @abc.abstractmethod
    def get_numerical_categorical_data(self, categorical_column, numerical_column):
        raise NotImplementedError

    @abc.abstractmethod
    def get_value_counts_data(self, categorical_column, limit):
        raise NotImplementedError


class PandasSummarizer(DataSummarizer):
    pass


class HiveSummarizer(DataSummarizer):

    def get_anova_summary(self, categorical_column, numerical_column):
        anova_template = f"""
            SELECT
                {edac.anova.CATEGORICAL},
                {edac.anova.COUNT_PER_CLASS},
                {edac.anova.MEAN_PER_CLASS},
                {edac.anova.VARIANCE_PER_CLASS},
                COUNT(1) OVER() - 1 AS {edac.anova.DF_GROUP},
                SUM(anova_count_per_class) OVER() - COUNT(1) OVER() AS {edac.anova.DF_ERROR}
            FROM (
                SELECT
                    {categorical_column} AS {edac.anova.CATEGORICAL},
                    COUNT(1) AS {edac.anova.COUNT_PER_CLASS},
                    AVG(CAST({numerical_column} AS FLOAT64)) AS {edac.anova.MEAN_PER_CLASS},
                    VAR_POP(CAST({numerical_column} AS FLOAT64)) AS {edac.anova.VARIANCE_PER_CLASS}
                FROM
                    {table}
                GROUP BY
                    {categorical_column}
            )
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_categorical_aggregation(self, categorical_columns):
        raise NotImplementedError

    @abc.abstractmethod
    def get_correlation_data(self, numerical_columns):
        raise NotImplementedError

    @abc.abstractmethod
    def get_numerical_summary(self, numerical_columns):
        raise NotImplementedError

    @abc.abstractmethod
    def get_categorical_summary(self, categorical_columns):
        raise NotImplementedError

    @abc.abstractmethod
    def get_histogram_data(self, numerical_column, num_bins):
        raise NotImplementedError

    @abc.abstractmethod
    def get_numerical_categorical_data(self, categorical_column, numerical_column):
        raise NotImplementedError

    @abc.abstractmethod
    def get_value_counts_data(self, categorical_column, limit):
        raise NotImplementedError


class SummaryFactory:

    @staticmethod
    def new_summarizer(**config):
        available_sources = {edac.data.PANDAS_DATAFRAME, edac.datasource.HIVE_TABLE}

        if config["datasource"] not in available_sources:
            raise ValueError(f"Only {available_sources!r} are supported datasources.")
        if config["datasource"] == edac.datasource.PANDAS_DATAFRAME:
            return PandasSummarizer(**config)
        if config["datasource"] == edac.datasource.HIVE_TABLE:
            return HiveSummarizer(**config)
