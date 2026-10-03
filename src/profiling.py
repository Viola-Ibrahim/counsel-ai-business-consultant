"""
profiling.py
------------
Responsible for extracting information about the DataFrame.
Returning a dictionary contains the important info like(shape, categorical columns, numerical columns ...etc )
"""
 
import pandas as pd
 
 
def profile_data(df: pd.DataFrame) -> dict:
    null_num = df.isnull().sum()
    null_percent = (null_num / len(df) * 100).round(2)
 
    duplicates_num = df.duplicated().sum()
    duplicates_percent = round((duplicates_num / len(df) * 100), 2)
 
    return {
        'shape': {'rows': df.shape[0], 'columns': df.shape[1]},
        'categorical_columns': list(df.select_dtypes(['category', 'bool', 'object']).columns),
        'numerical_columns': list(df.select_dtypes(['number']).columns),
        'null_num': null_num.to_dict(),
        'null_percent': null_percent.to_dict(),
        'duplicates_num': duplicates_num,
        'duplicates_percent': duplicates_percent,
        'unique_num': df.nunique().to_dict(),
        'describe_numerical': df.select_dtypes(['number']).describe().to_dict(),
        'describe_categorical': df.select_dtypes(['category', 'bool', 'object']).describe().to_dict()
    }
 
