from __future__ import annotations
from ..base import EDATracker
import pandas as pd
from pathlib import Path
from typing import Optional
from .metadata import (
    generate_metadata_from_pandas_df,
    generate_metadata_from_hive_table,
    load_metadata_from_yaml,
    Metadata,
)
from .constants import edac


class EDA:
    def __init__(
        self,
        datasource: str,
        metadata_file: Path,
        metadata_generation: bool = False,
        df: pd.DataFrame = None,
        spark_context=None,
        schema: str = None,
        table_name: str = None,
    ):
        self.datasource = datasource
        self.metadata_file = metadata_file
        self.metadata_generation = metadata_generation
        self.df = df
        self.spark_context = spark_context
        self.schema = schema
        self.table_name = table_name
        self.tracker = EDATracker()
        self.metadata = None

    def load_metadata(self) -> Metadata:
        if self.metadata_generation and self.datasource == edac.datasource.PANDAS_DATAFRAME:
            generate_metadata_from_pandas_df(self.df, self.metadata_file)
        if self.metadata_generation and self.datasource == edac.datasource.HIVE_TABLE:
            generate_metadata_from_hive_table(
                self.spark_context, self.schema, self.table_name, self.metadata_file
            )
        return load_metadata_from_yaml(self.metadata_file)

    def _run_descriptive(self):
        analyzer = DescriptiveAnalysis(self.metadata, PreprocessorFactory.new_preprocessor(datasource))

    def run(self, report_path: Path):
        self.metadata = self.load_metadata()
        self._run_descriptive()
        self._categorical_cardinality_check()
        self._run_qualitative()
        self._run_quantitative()

        pass

    @classmethod
    def from_pandas_dataframe(
        cls, df: pd.DataFrame, metadata_file: Path, metadata_generation: bool = False
    ) -> EDA:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas dataframe for 'df', but got {type(df)}.")
        return cls(
            datasource=edac.datasource.PANDAS_DATAFRAME,
            metadata_file=metadata_file,
            df=df,
            metadata_generation=metadata_generation,
        )

    @classmethod
    def from_hive_table(
        cls,
        spark_context,
        schema: str,
        table_name: str,
        metadata_file: Path,
        metadata_generation: bool = False,
    ) -> EDA:
        return cls(
            datasource=edac.datasource.HIVE_TABLE,
            metadata_file=metadata_file,
            metadata_generation=metadata_generation,
            spark_context=spark_context,
            schema=schema,
            table_name=table_name,
        )
