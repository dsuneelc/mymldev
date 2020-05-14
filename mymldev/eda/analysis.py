from typing import List
from dataclasses import dataclass


@dataclass
class Attribute:
    name: str
    dtype: str

    @property.getter
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        allowed_vals = {"numerical", "categorical"}
        if value not in allowed_vals:
            raise ValueError(f"Allowed value are, {allowed_vals!r}")
        self._dtype = value


@dataclass
class Datasource:
    source: str
    location: str
    target: Attribute
    features: List[Attribute]

    @property.getter
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        allowed_vals = {"CSV", "BIGQUERY"}
        if value not in allowed_vals:
            raise ValueError(f"Allowed value are, {allowed_vals!r}")
        self._source = value


@dataclass
class ScalarMetric:
    name: str
    value: float

    @property.getter
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        allowed_vals = {
            "F_STATISTIC",
            "P_VALUE",
            "INFORMATION_GAIN",
            "CORRELATION_COEFFICIENT",
            "MEAN",
            "MEDIAN",
            "STD",
            "MIN",
            "MAX",
            "QUANTILE_25",
            "QUANTILE_75",
            "QUANTILE_95",
            "SKEWNESS",
            "CARDINALITY",
            "TOTAL_COUNT",
            "MISSING",
        }
        if value not in allowed_vals:
            raise ValueError(f"Allowed value are, {allowed_vals!r}")
        self._name = value


@dataclass
class TableMetricCell:
    row_index: str
    column_index: str
    value: float


@dataclass
class TableMetricRow:
    row_index: str
    cells: List[TableMetricCell]


@dataclass
class TableMetric:
    name: str
    column_indexes: List[str]
    rows: List[TableMetricRow]

    @property.getter
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        allowed_vals = {"HISTOGRAM", "VALUE_COUTS", "CONTINGENCY_TABLE", "TABLE_DESCRIPTIVE"}
        if value not in allowed_vals:
            raise ValueError(f"Allowed value are, {allowed_vals!r}")
        self._name = value


@dataclass
class Analysis:
    name: str
    features: List[Attribute]
    smetrics: List[ScalarMetric]
    tmetrics: List[TableMetric]

    @property.getter
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        allowed_vals = {
            "DESCRIPTIVE",
            "HISTOGRM",
            "VALUE_COUTS",
            "CONTINGENCY_TABLE",
            "TABLE_DESCRIPTIVE",
            "PEARSON_CORRELATION",
            "ANOVA",
            "CHI_SQUARE",
            "INFORMATION_GAIN",
        }
        if value not in allowed_vals:
            raise ValueError(f"Allowed value are, {allowed_vals!r}")
        self._name = value


@dataclass
class AnalysisRun:
    timestamp_sec: float
    datasource: Datasource
    analyses: List[Analysis]
