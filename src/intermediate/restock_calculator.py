"""Restock recommendation logic for the Intermediate simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_PRODUCT_SUMMARY_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "starting_stock",
    "remaining_stock",
    "total_quantity_sold",
    "average_daily_sales",
}


def validate_product_summary(product_summary: pd.DataFrame) -> None:
    """Validate product summary columns needed for restocking.

    Parameters
    ----------
    product_summary : pandas.DataFrame
        Product-level monthly summary from the simulator.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing_columns = REQUIRED_PRODUCT_SUMMARY_COLUMNS.difference(
        product_summary.columns
    )
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Product summary is missing columns: {missing}")


def classify_stockout_risk(row: pd.Series) -> str:
    """Classify stockout risk using remaining stock and daily sales.

    Parameters
    ----------
    row : pandas.Series
        One row from the product summary plus reorder calculations.

    Returns
    -------
    str
        Risk label: ``High``, ``Medium``, or ``Low``.
    """
    remaining_stock = float(row.get("remaining_stock", 0))
    average_daily_sales = float(row.get("average_daily_sales", 0))
    reorder_point = float(row.get("reorder_point", 0))

    if remaining_stock <= 0:
        return "High"
    if average_daily_sales > 0 and remaining_stock <= reorder_point:
        return "High"
    if average_daily_sales > 0 and remaining_stock <= average_daily_sales * 14:
        return "Medium"
    return "Low"


def create_restock_recommendations(
    product_summary: pd.DataFrame,
    reorder_days: int = 7,
    cover_days: int = 14,
    minimum_reorder_point: int = 5,
) -> pd.DataFrame:
    """Create rule-based restock recommendations for next month.

    The main rule follows the project logic: when remaining stock is less than
    or equal to the reorder point, recommend restocking. Suggested restock
    quantity is the larger of 14 days of expected sales or the amount needed to
    refill the item back to starting stock.

    Parameters
    ----------
    product_summary : pandas.DataFrame
        Product-level monthly summary with sales and remaining stock.
    reorder_days : int, default=7
        Number of sales days used to compute the reorder point.
    cover_days : int, default=14
        Number of expected sales days to cover through restocking.
    minimum_reorder_point : int, default=5
        Minimum stock level that triggers a reorder check.

    Returns
    -------
    pandas.DataFrame
        Product-level restock recommendation table.
    """
    validate_product_summary(product_summary)

    restock = product_summary.copy()
    restock["average_daily_sales"] = pd.to_numeric(
        restock["average_daily_sales"], errors="coerce"
    ).fillna(0)
    restock["remaining_stock"] = pd.to_numeric(
        restock["remaining_stock"], errors="coerce"
    ).fillna(0)
    restock["starting_stock"] = pd.to_numeric(
        restock["starting_stock"], errors="coerce"
    ).fillna(0)

    restock["reorder_point"] = np.maximum(
        minimum_reorder_point,
        np.ceil(restock["average_daily_sales"] * reorder_days).astype(int),
    )
    restock["recommended_restock_quantity"] = np.maximum(
        np.ceil(restock["average_daily_sales"] * cover_days).astype(int),
        (restock["starting_stock"] - restock["remaining_stock"]).clip(lower=0),
    ).astype(int)
    restock["recommend_restock"] = (
        restock["remaining_stock"] <= restock["reorder_point"]
    )
    restock.loc[
        ~restock["recommend_restock"], "recommended_restock_quantity"
    ] = 0
    restock["stockout_risk"] = restock.apply(classify_stockout_risk, axis=1)

    restock["restock_reason"] = np.select(
        [
            restock["remaining_stock"] <= 0,
            restock["recommend_restock"],
            restock["stockout_risk"].eq("Medium"),
        ],
        [
            "Stock depleted during the month; restock immediately.",
            "Remaining stock is at or below the reorder point.",
            "Stock is above reorder point but may be tight within 14 days.",
        ],
        default="Stock level is sufficient for the next cycle.",
    )

    output_columns = [
        "product_id",
        "product_name",
        "category",
        "starting_stock",
        "total_quantity_sold",
        "average_daily_sales",
        "remaining_stock",
        "reorder_point",
        "recommend_restock",
        "recommended_restock_quantity",
        "stockout_risk",
        "restock_reason",
    ]
    return restock[output_columns].sort_values(
        ["recommend_restock", "stockout_risk", "recommended_restock_quantity"],
        ascending=[False, True, False],
    )
