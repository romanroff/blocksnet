import pandas as pd
import numpy as np
from loguru import logger
from .schemas import Schema


def _impute_population(df: pd.DataFrame, total_population: int) -> pd.DataFrame:
    df = df.copy()
    weights = df["living_area"] / df["living_area"].sum()
    ideal = weights * total_population

    alloc = np.floor(ideal).astype(int)
    remainder = int(total_population - alloc.sum())
    fractions = ideal - alloc

    sorted_fractions = fractions.sort_values(ascending=False).head(remainder)
    fractions_idx = sorted_fractions.index

    alloc.loc[fractions_idx] += 1

    df["population"] = alloc
    return df


def impute_population(df: pd.DataFrame, total_population: int) -> pd.DataFrame:
    if not isinstance(total_population, int):
        raise TypeError("Total population must be int")
    if total_population <= 0:
        raise ValueError("Total population must be greater than 0")
    df = Schema(df)

    sum_population = int(df["population"].sum())
    delta_population = total_population - sum_population
    if delta_population < 0:
        raise ValueError(
            f"Total population must be greater than population sum, got:\n{total_population} - {sum_population} = {delta_population}"
        )
    if delta_population == 0:
        logger.warning("0 population to distribute")
        return df.fillna(0)

    idx = df[(df["population"].isna()) | ((df["population"] == 0) & (df["living_area"] > 0))].index
    if len(idx) == 0:
        logger.warning("No unknown population found")
        return df

    logger.info(f"Distributing {delta_population} population between {len(idx)} rows")

    sub_df = _impute_population(df.loc[idx], delta_population)
    df.loc[idx, df.columns] = sub_df
    return df
