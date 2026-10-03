"""
analysis.py
------------
Responsible for performing actual data analysis operations on the DataFrame
using pandas. Contains descriptive stats, grouping, ratios, time-series,
correlation, and anomaly detection tools.
"""
 
import pandas as pd
 
# pandas >= 2.2 renamed the "end of period" aliases (M -> ME, Y -> YE, Q -> QE)
_FREQ_ALIASES = {"M": "ME", "Y": "YE", "A": "YE", "Q": "QE"}
 
 
def descriptive_analysis(df: pd.DataFrame, column: str) -> dict:
    """
    Basic descriptive statistics for a single column.
    Numeric column -> mean/median/min/max/std.
    Text/categorical column -> count, unique values, and the top 5 most frequent values.
 
    Example:
        descriptive_analysis(df, column="revenue")
    """
    series = df[column]
 
    if pd.api.types.is_numeric_dtype(series):
        return {
            "mean": series.mean(),
            "median": series.median(),
            "min": series.min(),
            "max": series.max(),
            "std": series.std(),
        }
 
    return {
        "count": int(series.count()),
        "unique": int(series.nunique()),
        "top_values": series.value_counts().head(5).to_dict(),
    }
 
 
def group_analysis(df: pd.DataFrame, metric: str, group_by: str, agg: str = "sum") -> pd.Series:
    """
    Group the DataFrame by `group_by` and aggregate `metric` using `agg`.
 
    Example:
        group_analysis(df, metric="revenue", group_by="region")
        -> total revenue per region, sorted from highest to lowest.
    """
    result = df.groupby(group_by, observed=True)[metric].agg(agg)
    result = result.sort_values(ascending=False)
    return result
 
 
def ratio_analysis(df: pd.DataFrame, numerator: str, denominator: str, group_by: str) -> pd.Series:
    """
    Ratio of the SUMS of two numeric columns per group (a correct, weighted ratio).
 
    Example:
        ratio_analysis(df, numerator="profit", denominator="net_sales", group_by="region")
        -> profit margin per region (total profit / total net sales).
    """
    sums = df.groupby(group_by, observed=True)[[numerator, denominator]].sum()
    ratio = sums[numerator] / sums[denominator].where(sums[denominator] != 0)
    ratio = ratio.sort_values(ascending=False).round(4)
    ratio.name = f"{numerator} / {denominator}"
    return ratio
 
 
def time_series_analysis(df: pd.DataFrame, date_column: str, metric: str, freq: str = "ME") -> pd.Series:
    """
    Aggregate `metric` over time, resampled by `freq` (e.g. "D", "ME", "YE").
    Old aliases ("M", "Y", "Q") are accepted and converted automatically.
 
    Example:
        time_series_analysis(df, date_column="order_date", metric="revenue", freq="ME")
        -> total revenue per month.
    """
    temp = df[[date_column, metric]].copy()
    temp[date_column] = pd.to_datetime(temp[date_column])
    temp = temp.set_index(date_column)
 
    try:
        return temp[metric].resample(_FREQ_ALIASES.get(freq, freq)).sum()
    except ValueError:  # older pandas that only knows the old aliases
        return temp[metric].resample(freq).sum()
 
 
def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix between all numerical columns.
 
    Example:
        correlation_analysis(df)
        -> DataFrame showing how numerical columns relate to each other.
    """
    numerical_df = df.select_dtypes(include="number")
    return numerical_df.corr()
 
 
def detect_anomalies(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalies (outliers) in a numerical column using the Z-score method.
    Any row where the Z-score exceeds `threshold` is flagged as an anomaly.
 
    Example:
        detect_anomalies(df, column="revenue")
        -> rows considered outliers based on revenue.
    """
    series = df[column]
    std = series.std()
    if not std or pd.isna(std):  # constant column: nothing can be an outlier
        return df.iloc[0:0]
 
    z_scores = (series - series.mean()) / std
    return df[z_scores.abs() > threshold]
 
