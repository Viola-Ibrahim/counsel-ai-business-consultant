"""
visualization.py
------------------
Responsible for turning analysis results into charts using matplotlib.
"""

import matplotlib.pyplot as plt
import pandas as pd


def create_bar_chart(data: pd.Series, title: str = "", xlabel: str = "", ylabel: str = "") -> plt.Figure:
    """
    Create a bar chart from a pandas Series (e.g. output of group_analysis).

    Example:
        result = group_analysis(df, metric="revenue", group_by="region")
        fig = create_bar_chart(result, title="Revenue by Region")
    """
    fig, ax = plt.subplots()
    data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def create_line_chart(data: pd.Series, title: str = "", xlabel: str = "", ylabel: str = "") -> plt.Figure:
    """
    Create a line chart from a pandas Series (e.g. output of time_series_analysis).

    Example:
        result = time_series_analysis(df, date_column="order_date", metric="revenue")
        fig = create_line_chart(result, title="Revenue Over Time")
    """
    fig, ax = plt.subplots()
    data.plot(kind="line", ax=ax, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def save_chart(fig: plt.Figure, path: str) -> None:
    """
    Save a chart figure to disk (e.g. for the report/dashboard).

    Example:
        save_chart(fig, "outputs/revenue_by_region.png")
    """
    fig.savefig(path)