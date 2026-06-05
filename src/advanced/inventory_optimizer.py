"""
inventory_optimizer.py

Advanced-level inventory optimization engine for the Sari-Sari Store Simulator.

Analyses five years of generated transaction data to produce monthly restock
recommendations for each product. Recommendations account for:

- Historical average monthly demand per product
- Remaining stock after the month's sales
- Average daily sales rate and projected stockout timeline
- Stockout risk classification (LOW / MEDIUM / HIGH / CRITICAL)
- Sales event impact on near-term demand
- Category-level demand patterns

Restock logic:
    restock_quantity = max(
        average_monthly_demand * RESTOCK_MONTHS_COVERAGE,
        MIN_RESTOCK_UNITS
    ) - current_stock

    Only recommended when current_stock < REORDER_THRESHOLD * avg_monthly_demand

Output per month:
    data/processed/advanced/year_YYYY/month_MM/restock_recommendations.csv

Aggregated output:
    data/processed/advanced/advanced_inventory_recommendations.csv
    SQLite table: advanced_inventory_recommendations
"""

from pathlib import Path
import pandas as pd
import numpy as np
import calendar
from datetime import date

from sqlalchemy import create_engine

try:
    from src.basic.data_loader import load_inventory
    from src.advanced.five_year_generator import START_YEAR, END_YEAR
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.basic.data_loader import load_inventory
    START_YEAR = 2022
    END_YEAR = 2026


# ---------------------------------------------------------------------------
# Optimiser constants
# ---------------------------------------------------------------------------

# How many months of forward demand to cover with a restock order
RESTOCK_MONTHS_COVERAGE = 1.5

# Reorder when stock falls below this fraction of average monthly demand
REORDER_THRESHOLD = 0.50

# Minimum units to recommend restocking (avoids trivially small orders)
MIN_RESTOCK_UNITS = 10

# Stockout risk thresholds expressed as days of stock remaining
STOCKOUT_RISK_THRESHOLDS = {
    "CRITICAL": 7,   # less than 1 week of stock
    "HIGH":     14,  # less than 2 weeks
    "MEDIUM":   21,  # less than 3 weeks
    "LOW":      999, # more than 3 weeks (no immediate concern)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_stockout_risk(
    current_stock: float,
    avg_daily_sales: float,
) -> str:
    """
    Classify stockout risk based on current stock and average daily sales.

    Parameters
    ----------
    current_stock : float
        Remaining units in stock.
    avg_daily_sales : float
        Average units sold per day over the historical period.

    Returns
    -------
    str
        One of: "CRITICAL", "HIGH", "MEDIUM", "LOW"
    """
    if avg_daily_sales <= 0:
        return "LOW"

    days_of_stock = current_stock / avg_daily_sales

    if days_of_stock <= STOCKOUT_RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif days_of_stock <= STOCKOUT_RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif days_of_stock <= STOCKOUT_RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    else:
        return "LOW"


def _build_recommendation_reason(
    risk: str,
    current_stock: float,
    avg_monthly_demand: float,
    restock_qty: float,
    has_upcoming_event: bool,
) -> str:
    """
    Build a human-readable reason string for a restock recommendation.

    Parameters
    ----------
    risk : str
        Stockout risk level.
    current_stock : float
        Remaining units.
    avg_monthly_demand : float
        Average monthly units sold historically.
    restock_qty : float
        Recommended restock quantity.
    has_upcoming_event : bool
        Whether a sales event is active or upcoming next month.

    Returns
    -------
    str
        Readable recommendation reason.
    """
    base = (
        f"Stock at {int(current_stock)} units vs "
        f"avg monthly demand of {avg_monthly_demand:.0f} units. "
        f"Stockout risk: {risk}."
    )

    if has_upcoming_event:
        base += " Upcoming sales event may increase demand — order early."

    if restock_qty > 0:
        base += f" Recommend ordering {int(restock_qty)} units."
    else:
        base += " Stock level sufficient; no restock needed."

    return base


# ---------------------------------------------------------------------------
# Monthly restock calculation
# ---------------------------------------------------------------------------

def calculate_monthly_restock(
    inventory: pd.DataFrame,
    product_summary: pd.DataFrame,
    historical_demand: pd.DataFrame,
    year: int,
    month: int,
    upcoming_events: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Calculate restock recommendations for one month.

    Parameters
    ----------
    inventory : pd.DataFrame
        Raw inventory DataFrame from load_inventory(). Used for product
        metadata (name, category, cost, price).
    product_summary : pd.DataFrame
        Product summary for the current month. Must contain:
        product_id, total_quantity_sold, remaining_stock.
    historical_demand : pd.DataFrame
        Aggregated demand history. Must contain:
        product_id, total_quantity_sold (summed over all prior months).
        Used to compute average monthly demand.
    year : int
        Current year (used to count elapsed months for averages).
    month : int
        Current month (1-12).
    upcoming_events : pd.DataFrame, optional
        Sales events active in the next month. If provided, products
        in affected categories get flagged for early restocking.

    Returns
    -------
    pd.DataFrame
        Restock recommendations with columns:
        product_id, product_name, category, unit_cost, unit_price,
        average_monthly_quantity_sold, current_stock,
        average_daily_sales, stockout_risk,
        recommended_restock_quantity, recommendation_reason
    """
    # Number of months elapsed so far (for computing monthly averages)
    months_elapsed = max(
        1,
        (year - START_YEAR) * 12 + month
    )

    # Compute average monthly demand per product from history
    avg_monthly = (
        historical_demand
        .groupby("product_id")["total_quantity_sold"]
        .sum()
        .div(months_elapsed)
        .reset_index(name="avg_monthly_demand")
    )

    # Merge current month product summary with historical averages
    merged = product_summary.merge(
        avg_monthly,
        on="product_id",
        how="left",
    )

    # Only merge inventory columns not already present in product_summary.
    # all_product_summaries (from monthly_outputs.py) already contains
    # product_name, category, unit_cost, unit_price — merging them again
    # creates _x/_y duplicates and breaks the column selector below.
    inv_cols_needed = [
        c for c in ["product_name", "category", "unit_cost", "unit_price"]
        if c not in merged.columns
    ]
    if inv_cols_needed:
        merged = merged.merge(
            inventory[["product_id"] + inv_cols_needed],
            on="product_id",
            how="left",
        )

    merged["avg_monthly_demand"] = merged["avg_monthly_demand"].fillna(
        merged["total_quantity_sold"]
    )

    days_in_month = calendar.monthrange(year, month)[1]
    merged["avg_daily_sales"] = (
        merged["avg_monthly_demand"] / days_in_month
    ).round(2)

    # Classify stockout risk
    merged["stockout_risk"] = merged.apply(
        lambda r: _classify_stockout_risk(
            r["remaining_stock"], r["avg_daily_sales"]
        ),
        axis=1,
    )

    # Determine which categories have upcoming events
    upcoming_categories = set()
    if upcoming_events is not None and not upcoming_events.empty:
        upcoming_categories = set(
            upcoming_events["affected_category"].unique()
        )

    # Calculate recommended restock quantity
    def _restock_qty(row):
        if row["remaining_stock"] <= row["avg_monthly_demand"] * REORDER_THRESHOLD:
            target = max(
                row["avg_monthly_demand"] * RESTOCK_MONTHS_COVERAGE,
                MIN_RESTOCK_UNITS,
            )
            qty = max(0, target - row["remaining_stock"])
            return int(np.ceil(qty))
        return 0

    merged["recommended_restock_quantity"] = merged.apply(
        _restock_qty, axis=1
    )

    # Build reason strings
    def _reason(row):
        has_event = row.get("category", "") in upcoming_categories
        return _build_recommendation_reason(
            risk=row["stockout_risk"],
            current_stock=row["remaining_stock"],
            avg_monthly_demand=row["avg_monthly_demand"],
            restock_qty=row["recommended_restock_quantity"],
            has_upcoming_event=has_event,
        )

    merged["recommendation_reason"] = merged.apply(_reason, axis=1)

    result = merged[[
        "product_id",
        "product_name",
        "category",
        "unit_cost",
        "unit_price",
        "avg_monthly_demand",
        "remaining_stock",
        "avg_daily_sales",
        "stockout_risk",
        "recommended_restock_quantity",
        "recommendation_reason",
    ]].copy()

    result = result.rename(columns={
        "avg_monthly_demand": "average_monthly_quantity_sold",
        "remaining_stock":    "current_stock",
        "avg_daily_sales":    "average_daily_sales",
    })

    result["average_monthly_quantity_sold"] = (
        result["average_monthly_quantity_sold"].round(2)
    )

    result["year"]  = year
    result["month"] = month

    return result.sort_values(
        ["stockout_risk", "recommended_restock_quantity"],
        ascending=[True, False],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_monthly_restock_csv(
    restock: pd.DataFrame,
    year: int,
    month: int,
    output_base_folder: str | Path,
) -> Path:
    """
    Save monthly restock recommendations to its year/month folder.

    Parameters
    ----------
    restock : pd.DataFrame
        Monthly restock recommendations DataFrame.
    year : int
        Year.
    month : int
        Month (1-12).
    output_base_folder : str or Path
        Root advanced output folder.

    Returns
    -------
    Path
        Path of the saved CSV.
    """
    folder = (
        Path(output_base_folder) / f"year_{year}" / f"month_{month:02d}"
    )
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / "restock_recommendations.csv"
    restock.to_csv(path, index=False)
    return path


def save_inventory_recommendations_to_sqlite(
    all_restock: pd.DataFrame,
    sqlite_db_path: str | Path,
) -> None:
    """
    Save aggregated inventory recommendations to SQLite.

    Table name: advanced_inventory_recommendations

    Parameters
    ----------
    all_restock : pd.DataFrame
        Combined restock recommendations for all months.
    sqlite_db_path : str or Path
        Path to the SQLite database.
    """
    sqlite_db_path = Path(sqlite_db_path)
    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{sqlite_db_path}")

    all_restock.to_sql(
        name="advanced_inventory_recommendations",
        con=engine,
        if_exists="replace",
        index=False,
    )

    print(
        f"SQLite table saved    : "
        f"advanced_inventory_recommendations → {sqlite_db_path}"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_inventory_optimizer(
    inventory_csv_path: str | Path = "data/raw/inventory.csv",
    output_base_folder: str | Path = "data/processed/advanced",
    sqlite_db_path: str | Path = "src/database/sari_sari_store.db",
    all_transactions: pd.DataFrame = None,
    all_product_summaries: pd.DataFrame = None,
    sales_events: pd.DataFrame = None,
    save_csv: bool = True,
    save_sqlite: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full inventory optimization workflow across all 60 months.

    For each month, computes average monthly demand from all prior months,
    assesses current stock levels, classifies stockout risk, and generates
    restock quantity recommendations.

    Parameters
    ----------
    inventory_csv_path : str or Path
        Path to data/raw/inventory.csv.
    output_base_folder : str or Path
        Root advanced output folder.
    sqlite_db_path : str or Path
        Path to the SQLite database.
    all_transactions : pd.DataFrame, optional
        Aggregated five-year transactions from run_five_year_generator().
        If None, falls back to inventory starting_stock for current_stock.
    all_product_summaries : pd.DataFrame, optional
        Aggregated monthly product summaries. Required columns:
        year, month, product_id, total_quantity_sold, remaining_stock.
        If None, optimizer uses transaction data to reconstruct summaries.
    sales_events : pd.DataFrame, optional
        Full sales events DataFrame from run_sales_event_generator().
    save_csv : bool
        Whether to save monthly restock CSVs.
    save_sqlite : bool
        Whether to save the aggregated table to SQLite.
    verbose : bool
        Whether to print progress to the console.

    Returns
    -------
    pd.DataFrame
        Aggregated inventory recommendations for all 60 months.
    """
    inventory = load_inventory(inventory_csv_path)

    if verbose:
        print("\n" + "=" * 72)
        print("ADVANCED GOAL: INVENTORY OPTIMIZER")
        print("=" * 72)
        print(f"Products        : {len(inventory)}")
        print(f"Period          : {START_YEAR}–{END_YEAR} (60 months)")
        print("-" * 72)

    all_restock_frames = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):

            # --- Build product summary for this month ---
            if all_product_summaries is not None and not all_product_summaries.empty:
                mask = (
                    (all_product_summaries["year"] == year)
                    & (all_product_summaries["month"] == month)
                )
                product_summary = all_product_summaries[mask].copy()
            elif all_transactions is not None and not all_transactions.empty:
                mask = (
                    (pd.to_datetime(
                        all_transactions["transaction_date"]
                    ).dt.year == year)
                    & (pd.to_datetime(
                        all_transactions["transaction_date"]
                    ).dt.month == month)
                )
                monthly_txns = all_transactions[mask]

                product_summary = (
                    monthly_txns
                    .groupby("product_id")["quantity_sold"]
                    .sum()
                    .reset_index(name="total_quantity_sold")
                )
                # Estimate remaining stock: starting_stock - quantity sold this month
                product_summary = product_summary.merge(
                    inventory[["product_id", "starting_stock"]],
                    on="product_id",
                    how="left",
                )
                product_summary["remaining_stock"] = (
                    product_summary["starting_stock"]
                    - product_summary["total_quantity_sold"]
                ).clip(lower=0)
            else:
                # Fallback: use inventory starting_stock as current stock
                product_summary = inventory[
                    ["product_id", "starting_stock"]
                ].copy()
                product_summary = product_summary.rename(
                    columns={"starting_stock": "remaining_stock"}
                )
                product_summary["total_quantity_sold"] = 0

            # --- Build historical demand (all months up to current) ---
            if all_transactions is not None and not all_transactions.empty:
                hist_mask = (
                    pd.to_datetime(
                        all_transactions["transaction_date"]
                    ).dt.year * 100
                    + pd.to_datetime(
                        all_transactions["transaction_date"]
                    ).dt.month
                ) <= (year * 100 + month)

                historical_demand = (
                    all_transactions[hist_mask]
                    .groupby("product_id")["quantity_sold"]
                    .sum()
                    .reset_index(name="total_quantity_sold")
                )
            else:
                historical_demand = product_summary[
                    ["product_id", "total_quantity_sold"]
                ].copy()

            # --- Upcoming events for next month ---
            if sales_events is not None and not sales_events.empty:
                next_month = month + 1 if month < 12 else 1
                next_year  = year if month < 12 else year + 1
                if next_year <= END_YEAR:
                    from src.advanced.sales_event_generator import (
                        get_events_for_month,
                    )
                    upcoming_events = get_events_for_month(
                        sales_events, next_year, next_month
                    )
                else:
                    upcoming_events = pd.DataFrame()
            else:
                upcoming_events = pd.DataFrame()

            # --- Run optimiser for this month ---
            # Drop year/month columns if present — they come from
            # all_product_summaries and conflict with result["year"] = year
            # assigned inside calculate_monthly_restock.
            cols_to_drop = [c for c in ["year", "month"] if c in product_summary.columns]
            if cols_to_drop:
                product_summary = product_summary.drop(columns=cols_to_drop)

            restock = calculate_monthly_restock(
                inventory=inventory,
                product_summary=product_summary,
                historical_demand=historical_demand,
                year=year,
                month=month,
                upcoming_events=upcoming_events,
            )

            if save_csv:
                save_monthly_restock_csv(
                    restock=restock,
                    year=year,
                    month=month,
                    output_base_folder=output_base_folder,
                )

            all_restock_frames.append(restock)

            if verbose:
                critical = (restock["stockout_risk"] == "CRITICAL").sum()
                high     = (restock["stockout_risk"] == "HIGH").sum()
                restock_needed = (
                    restock["recommended_restock_quantity"] > 0
                ).sum()
                print(
                    f"  {year}-{month:02d}  |  "
                    f"CRITICAL: {critical}  HIGH: {high}  |  "
                    f"Restock needed: {restock_needed} products"
                )

    all_restock = pd.concat(all_restock_frames, ignore_index=True)

    # Save aggregated CSV
    if save_csv:
        folder = Path(output_base_folder)
        folder.mkdir(parents=True, exist_ok=True)
        csv_path = folder / "advanced_inventory_recommendations.csv"
        all_restock.to_csv(csv_path, index=False)
        print(f"\nAggregated CSV saved  : {csv_path}")

    if save_sqlite:
        save_inventory_recommendations_to_sqlite(all_restock, sqlite_db_path)

    if verbose:
        print("-" * 72)
        total_restock = (all_restock["recommended_restock_quantity"] > 0).sum()
        print(f"Total restock recommendations : {total_restock:,}")
        print("Inventory optimization complete.")
        print("=" * 72)

    return all_restock


if __name__ == "__main__":
    run_inventory_optimizer()
