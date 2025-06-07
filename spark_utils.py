from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when
from pyspark.sql.types import *
import pandas as pd
from typing import Optional, Dict, Any
import os

class SparkDataProcessor:
    def __init__(self):
        self.spark = None
        self.df = None
        self.initialize_spark()
    
    def initialize_spark(self):
        try:
            self.spark = SparkSession.builder \
                .appName("AutoML-Pipeline") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.driver.memory", "2g") \
                .config("spark.executor.memory", "2g") \
                .getOrCreate()
            
            # Set log level to reduce verbosity
            self.spark.sparkContext.setLogLevel("WARN")
            
        except Exception as e:
            print(f"Warning: Could not initialize Spark: {e}")
            self.spark = None
    
    def load_large_dataset(self, file_path: str, use_spark: bool = True) -> Optional[pd.DataFrame]:
        if not use_spark or self.spark is None:
            return self._load_with_pandas(file_path)
        
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            if file_size_mb < 100:
                return self._load_with_pandas(file_path)
            
            if file_path.endswith('.csv'):
                self.df = self.spark.read.csv(
                    file_path,
                    header=True,
                    inferSchema=True
                )
            elif file_path.endswith(('.xlsx', '.xls')):
                pandas_df = pd.read_excel(file_path)
                self.df = self.spark.createDataFrame(pandas_df)
            
            if self.df.count() > 1000000:
                sample_fraction = min(0.1, 100000 / self.df.count())
                sampled_df = self.df.sample(fraction=sample_fraction, seed=42)
                return sampled_df.toPandas()
            else:
                return self.df.toPandas()
                
        except Exception as e:
            print(f"Spark processing failed: {e}, falling back to pandas")
            return self._load_with_pandas(file_path)
    
    def _load_with_pandas(self, file_path: str) -> Optional[pd.DataFrame]:
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
        except Exception as e:
            print(f"Error loading with pandas: {e}")
            return None
    
    def get_spark_data_profile(self) -> Dict[str, Any]:
        if self.df is None:
            return {}
        
        try:
            total_rows = self.df.count()
            total_cols = len(self.df.columns)
            
            missing_counts = {}
            for col_name in self.df.columns:
                missing_count = self.df.filter(
                    col(col_name).isNull() | 
                    isnan(col(col_name)) | 
                    (col(col_name) == "")
                ).count()
                missing_counts[col_name] = missing_count
            
            dtypes = {field.name: str(field.dataType) for field in self.df.schema.fields}
            
            return {
                'total_rows': total_rows,
                'total_cols': total_cols,
                'missing_counts': missing_counts,
                'dtypes': dtypes,
                'schema': self.df.schema.json()
            }
            
        except Exception as e:
            print(f"Error in Spark profiling: {e}")
            return {}
    
    def cleanup(self):
        if self.spark:
            self.spark.stop()

# Global instance
spark_processor = SparkDataProcessor() 