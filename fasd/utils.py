import pandas as pd
import numpy as np
import warnings
import torch
import random


def postprocess(
    df_syn: pd.DataFrame,
    df_ori: pd.DataFrame,
    discrete_features: list,
    missing_indicators: pd.DataFrame = None,
    missing_suffix: str = "_missing",
):
    """
    Post-processes a synthetic DataFrame to align with the structure and types of the original DataFrame.
    """
    df_syn = df_syn.copy()
    df_ori = df_ori.copy()
    numerical_features = [x for x in df_ori.columns if x not in discrete_features]

    if missing_indicators is not None:
        df_syn = reinstate_nans(
            df=df_syn[df_ori.columns],
            missing_indicators=df_syn[missing_indicators.columns],
            missing_suffix=missing_suffix,
        )

    df_syn[numerical_features] = align_precision(
        df_ori[numerical_features], df_syn[numerical_features]
    )
    df_syn = align_dtypes(df_ori, df_syn)

    return df_syn


def preprocess(
    df: pd.DataFrame,
    missing_indicators: pd.DataFrame = None,
):
    """
    Preprocesses a DataFrame by randomly imputing missing values and appending missing indicators if provided.
    """
    df = df.copy()
    if missing_indicators is not None:
        df = random_impute(df)
        df = pd.concat([df, missing_indicators], axis=1)

    return df


def align_precision(df_ref: pd.DataFrame, df_to_align: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns the numeric precision of df_to_align to match df_ref.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        df_aligned = df_to_align.copy()

        for col in df_ref.columns.intersection(df_to_align.columns):
            if pd.api.types.is_numeric_dtype(
                df_ref[col]
            ) and pd.api.types.is_numeric_dtype(df_to_align[col]):
                if pd.api.types.is_integer_dtype(df_ref[col]):
                    df_aligned[col] = df_aligned[col].round().astype("Int64")
                elif pd.api.types.is_float_dtype(df_ref[col]):

                    def count_decimals(x):
                        if pd.isna(x) or not np.isfinite(x):
                            return 0
                        s = str(x)
                        if "." in s:
                            return len(s.split(".")[-1].rstrip("0"))
                        return 0

                    decimal_places = df_ref[col].dropna().map(count_decimals).max()
                    if pd.isna(decimal_places):
                        decimal_places = 0

                    df_aligned[col] = df_aligned[col].round(int(decimal_places))

    return df_aligned


def align_dtypes(df_ref: pd.DataFrame, df_to_align: pd.DataFrame):
    """
    Aligns the data types of df_to_align to match those of df_ref.
    """
    for col in df_ref.columns:
        if col in df_to_align.columns:
            df_to_align[col] = df_to_align[col].astype(df_ref[col].dtype)
    return df_to_align


def random_impute(df: pd.DataFrame):
    """
    Fills missing values in each column with random samples from the non-missing values of that column.
    """
    df_imputed = df.copy()
    for col in df_imputed.columns:
        if df_imputed[col].isnull().any():
            non_na_values = df_imputed[col].dropna().values
            nan_indices = df_imputed[col].index[df_imputed[col].isnull()]
            sampled_values = np.random.choice(
                non_na_values, size=len(nan_indices), replace=True
            )
            df_imputed.loc[nan_indices, col] = sampled_values
    return df_imputed


def reinstate_nans(
    df: pd.DataFrame, missing_indicators: pd.DataFrame, missing_suffix: str
):
    """
    Reintroduces NaN values into a DataFrame based on missing indicator columns.
    """
    df = df.copy()
    for col in missing_indicators.columns:
        original_col = col.replace(missing_suffix, "")
        is_missing = missing_indicators[col].astype(str) == "1"
        df.loc[is_missing, original_col] = np.nan
    return df


def get_discretes(df: pd.DataFrame, discrete_threshold: int = None):
    """
    Identifies discrete features based on type and/or unique value count.
    """
    if (discrete_threshold is None) or discrete_threshold == 0:
        discrete_threshold = -float("inf")

    discretes = []
    for col in df.columns:
        try:
            df[col].astype(float)
            if df[col].nunique() <= discrete_threshold:
                discretes.append(col)
        except:
            discretes.append(col)
    return discretes


def set_seed(seed: int = 42):
    """
    Sets random seed across Python, NumPy, and PyTorch (CPU and CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
