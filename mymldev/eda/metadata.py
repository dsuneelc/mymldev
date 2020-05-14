from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from ..utils import DotDict
from .constants import c


PANDAS_DTYPE_MAP = {
    "i": c.allowed_types.NUMERICAL,
    "u": c.allowed_types.NUMERICAL,
    "f": c.allowed_types.NUMERICAL,
    "b": c.allowed_types.CATEGORICAL,
    "O": c.allowed_types.CATEGORICAL,
}

SPARK_DTYPE_MAP = {
    "ByteType": c.allowed_types.NUMERICAL,
    "ShortType": c.allowed_types.NUMERICAL,
    "IntegerType": c.allowed_types.NUMERICAL,
    "LongType": c.allowed_types.NUMERICAL,
    "FloatType": c.allowed_types.NUMERICAL,
    "DoubleType": c.allowed_types.NUMERICAL,
    "DecimalType": c.allowed_types.NUMERICAL,
    "StringType": c.allowed_types.CATEGORICAL,
    "BinaryType": c.allowed_types.CATEGORICAL,
    "BooleanType": c.allowed_types.CATEGORICAL,
}


@dataclass
class EDARun:
    contingency_table: bool = False
    descriptive_stats: bool = True
    correlation_table: bool = True
    information_gain: bool = True
    chi_square: bool = False
    anova: bool = False
    head: bool = False
    tail: bool = False
    distributions: bool = True


@dataclass
class EDAParams:
    histogram_bins: int = 10
    value_counts_limit: int = 15
    cardinality_limit: int = 15


@dataclass
class EDAConfig:
    numerical_columns: List[str]
    categorical_columns: List[str]
    eda_run: EDARun = EDARun()
    eda_params: EDAParams = EDAParams()
    target_column: Optional[str] = None


def _generate_metadata(columns_info, metadata_file) -> None:
    numerical_cols = columns_info[columns_info["type_class"] == c.allowed_types.NUMERICAL]["name"].tolist()
    categorical_cols = columns_info[columns_info["type_class"] == c.allowed_types.CATEGORICAL][
        "name"
    ].tolist()
    config = EDAConfig(numerical_columns=numerical_cols, categorical_columns=categorical_cols)
    with open(metadata_file, "w") as stream:
        stream.write(
            """# If eda_run with val 'false', then analysis is not performed.
# If eda_run with val 'true' and columns 'null', then all the columns are considered for analysis.
# If eda_run with val 'true' and list of columns specified, then only those columns are used for analysis.
# Specify target_column with dependent variable column name.
# EDA for only "classification" or "regression" is performed.\n\n"""
        )
        yaml.dump(asdict(config), stream, default_flow_style=False)


def generate_metadata_from_pandas_df(df, metadata_file) -> None:
    columns_info = (
        df.dtypes.apply(lambda x: x.kind)
        .map(PANDAS_DTYPE_MAP)
        .rename_axis("name")
        .rename("type_class")
        .reset_index()
    )
    _generate_metadata(columns_info, metadata_file)


def generate_metadata_from_hive_table(spark_context, schema, table_name, metadata_file):
    sdf = spark_context.sql(f"SELECT * FROM {schema}.{table_name}")
    cols_meta = [(field.name, str(field.dataType)) for field in sdf.schema.fields]
    columns_info = pd.DataFrame(cols_meta, columns=["name", "type_class"]).assign(
        type_class=lambda df: df["type_class"].map(SPARK_DTYPE_MAP)
    )
    _generate_metadata(columns_info, metadata_file)


class Metadata:

    def __init__(self, eda_config: EDAConfig):
        self._eda_config = eda_config
        self._anova_run = self._eda_config.eda_run.anova.val
        self._cardinality_limit = self._eda_config.eda_params.cardinality_limit
        self._categorical_attributes = self._eda_config.categorical_columns
        self._chi_square_run = self._eda_config.eda_run.chi_square.val
        self._contingency_table_run = self._eda_config.eda_run.contingency_table.val
        self._correlation_table_run = self._eda_config.eda_run.correlation_table.val
        self._descriptive_stats_run = self._eda_config.eda_run.descriptive_stats.val
        self._distributions_run = self._eda_config.eda_run.distributions.val
        self._features = self._eda_config.numerical_columns + self._eda_config.categorical_columns
        self._head_run = self._eda_config.eda_run.head.val
        self._histogram_bins = self._eda_config.eda_params.histogram_bins
        self._information_gain_run = self._eda_config.eda_run.information_gain.val
        self._numerical_attributes = self._eda_config.numerical_columns
        self._tail_run = self._eda_config.eda_run.tail.val
        self._target_column = self._eda_config.target_column
        self._value_counts_limit = self._eda_config.eda_params.value_counts_limit
        if self.target_column in self.numerical_attributes:
            self._problem = c.problem.REGRESSION
        elif self.target_column in self.categorical_attributes:
            self._problem = c.problem.CLASSIFICATION
        else:
            self._problem = None

    @property
    def anova_run(self):
        return self._anova_run

    @property
    def cardinality_limit(self):
        return self._cardinality_limit

    @property
    def categorical_attributes(self):
        return self._categorical_attributes

    @property
    def chi_square_run(self):
        return self._chi_square_run

    @property
    def contingency_table_run(self):
        return self._contingency_table_run

    @property
    def correlation_table_run(self):
        return self._correlation_table_run

    @property
    def descriptive_stats_run(self):
        return self._descriptive_stats_run

    @property
    def distributions_run(self):
        return self._distributions_run

    @property
    def features(self):
        return self._features

    @property
    def head_run(self):
        return self._head_run

    @property
    def histogram_bins(self):
        return self._histogram_bins

    @property
    def information_gain_run(self):
        return self._information_gain_run

    @property
    def numerical_attributes(self):
        return self._numerical_attributes

    @property
    def tail_run(self):
        return self._tail_run

    @property
    def target_column(self):
        return self._target_column

    @property
    def value_counts_limit(self):
        return self._value_counts_limit

    @property
    def problem(self):
        return self._problem


def load_metadata_from_yaml(metadata_file) -> Metadata:
    with open(metadata_file, "r") as stream:
        config = DotDict(yaml.load(stream))
    eda_run = EDARun(
        contingency_table=config.eda_run.contingency_table,
        descriptive_stats=config.eda_run.descriptive_stats,
        correlation_table=config.eda_run.correlation_table,
        information_gain=config.eda_run.information_gain,
        chi_square=config.eda_run.chi_square,
        anova=config.eda_run.anova,
        head=config.eda_run.head,
        tail=config.eda_run.tail,
        distributions=config.eda_run.distributions,
    )
    eda_params = EDAParams(**config.eda_params)
    return Metadata(
        EDAConfig(
            numerical_columns=config.numerical_columns,
            categorical_columns=config.categorical_columns,
            eda_run=eda_run,
            eda_params=eda_params,
            target_column=config.target_column,
        )
    )


if __name__ == "__main__":
    data_path = Path().cwd().parent / "datasets" / "data" / "titanic_train.csv"
    df = pd.read_csv(data_path)
    yaml_path = Path().cwd() / "metadata.yml"
    generate_metadata_from_pandas_df(df, yaml_path)
