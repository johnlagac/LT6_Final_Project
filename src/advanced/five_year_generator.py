"""
five_year_generator.py

Advanced-level synthetic transaction generator for the Sari-Sari Store Simulator.

This module generates five years (2022-2026) of realistic daily transaction
data for each product in data/raw/inventory.csv. Demand weights are calibrated
against the actual baseline from the project's real transactions.csv
(137 total units sold across 15 products on a single day).

Design rules:
- Read only from data/raw/inventory.csv (never modify it)
- Save monthly transactions to data/processed/advanced/year_YYYY/month_MM/
- Each month resets starting_stock to inventory defaults before calculating sales
- Transaction IDs use format ADV-YYYYMM-NNNNN to avoid collision with Basic IDs
- Returns all DataFrames so notebooks can inspect results

Generated output per month:
    data/processed/advanced/year_YYYY/month_MM/transactions.csv

Aggregated five-year output:
    data/processed/advanced/advanced_transactions_5yr.csv
    SQLite table: advanced_transactions_5yr
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date, timedelta
import calendar

from sqlalchemy import create_engine

try:
    from src.basic.data_loader import load_inventory
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.basic.data_loader import load_inventory


# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

START_YEAR = 2022
END_YEAR = 2026

# Baseline daily units sold per product derived from real transactions.csv.
# These are the observed single-day quantities used to calibrate generation.
BASELINE_DAILY_UNITS = {
    "P001": 5,   # Coke 1L
    "P002": 3,   # Royal 1L
    "P003": 12,  # Bottled Water 500ml
    "P004": 9,   # Piattos Snack
    "P005": 5,   # Nova Snack
    "P006": 9,   # Lucky Me Beef Noodles
    "P007": 9,   # Pancit Canton Chilimansi
    "P008": 3,   # Canned Sardines
    "P009": 3,   # Corned Beef Small
    "P010": 24,  # Egg
    "P011": 3,   # Rice 1kg
    "P012": 23,  # Coffee Sachet
    "P013": 12,  # Shampoo Sachet
    "P014": 12,  # Laundry Detergent Sachet
    "P015": 5,   # Dishwashing Liquid Sachet
}

DEFAULT_DAILY_UNITS = 3

# Monthly seasonality multipliers (index 0 = January).
# Based on Philippine consumption patterns:
# - April-June: summer peak (beverages, snacks)
# - November-December: Christmas peak (all categories)
# - June-July: back-to-school (personal care, household)
MONTHLY_SEASONALITY = [
    1.00,  # Jan
    0.95,  # Feb
    1.00,  # Mar
    1.10,  # Apr
    1.15,  # May
    1.20,  # Jun
    1.18,  # Jul
    1.05,  # Aug
    1.00,  # Sep
    1.05,  # Oct
    1.20,  # Nov
    1.35,  # Dec
]

# Year-over-year growth rate applied cumulatively from 2022
ANNUAL_GROWTH_RATE = 0.05

# Day-of-week multipliers: sari-sari stores sell more on weekends
DAY_OF_WEEK_MULTIPLIER = {
    0: 1.00,  # Monday
    1: 0.95,  # Tuesday
    2: 0.95,  # Wednesday
    3: 1.00,  # Thursday
    4: 1.10,  # Friday
    5: 1.25,  # Saturday
    6: 1.20,  # Sunday
}

# Random noise: standard deviation as a fraction of expected daily demand
NOISE_STD_FRACTION = 0.20

# Probability that a product sells on any given day (by category)
DAILY_SALE_PROBABILITY = {
    "Beverage":      0.92,
    "Food":          0.88,
    "Snack":         0.85,
    "Personal Care": 0.75,
    "Household":     0.80,
}
DEFAULT_SALE_PROBABILITY = 0.80


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _growth_multiplier(year: int) -> float:
    """
    Return cumulative growth multiplier for a given year relative to 2022.

    Example: year=2024 → (1.05)^2 = 1.1025
    """
    return (1 + ANNUAL_GROWTH_RATE) ** (year - START_YEAR)


def _expected_daily_qty(
    product_id: str,
    category: str,
    sale_date: date,
    active_events: pd.DataFrame,
) -> float:
    """
    Compute expected daily units sold for one product on one date.

    Combines baseline demand, seasonality, day-of-week pattern,
    year-over-year growth, and any active sales event boosts.

    Parameters
    ----------
    product_id : str
        Product identifier, used to look up baseline demand.
    category : str
        Product category for event matching.
    sale_date : date
        The date being evaluated.
    active_events : pd.DataFrame
        Events active on this date. Required columns:
        affected_category, demand_multiplier.

    Returns
    -------
    float
        Expected units sold (before noise is applied).
    """
    base = BASELINE_DAILY_UNITS.get(product_id, DEFAULT_DAILY_UNITS)
    seasonality = MONTHLY_SEASONALITY[sale_date.month - 1]
    dow_mult = DAY_OF_WEEK_MULTIPLIER[sale_date.weekday()]
    growth = _growth_multiplier(sale_date.year)

    expected = base * seasonality * dow_mult * growth

    # Apply the highest demand multiplier from any active event for this category
    if not active_events.empty:
        matching = active_events[
            active_events["affected_category"] == category
        ]
        if not matching.empty:
            expected *= matching["demand_multiplier"].max()

    return expected


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_monthly_transactions(
    inventory: pd.DataFrame,
    year: int,
    month: int,
    sales_events: pd.DataFrame = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate one month of synthetic daily transactions for all products.

    Each row is one transaction (one customer buying one product).
    Multiple transactions per product per day are possible, matching
    the pattern seen in the real transactions.csv baseline.

    Parameters
    ----------
    inventory : pd.DataFrame
        Validated inventory DataFrame from load_inventory().
    year : int
        Year to simulate (2022-2026).
    month : int
        Month to simulate (1-12).
    sales_events : pd.DataFrame, optional
        Full sales events table. Filtered to this month internally.
        Required columns: start_date, end_date, affected_category,
        demand_multiplier. Pass None if no events.
    random_seed : int
        Base seed offset by year and month for reproducibility.

    Returns
    -------
    pd.DataFrame
        Monthly transactions with columns:
        transaction_id, transaction_date, product_id, quantity_sold
    """
    rng = np.random.default_rng(random_seed + year * 100 + month)

    if sales_events is None or sales_events.empty:
        events_df = pd.DataFrame(columns=[
            "start_date", "end_date", "affected_category", "demand_multiplier"
        ])
    else:
        events_df = sales_events.copy()
        # Normalise date columns for comparison
        events_df["start_date"] = pd.to_datetime(
            events_df["start_date"]
        ).dt.date
        events_df["end_date"] = pd.to_datetime(
            events_df["end_date"]
        ).dt.date

    days_in_month = calendar.monthrange(year, month)[1]
    all_dates = [
        date(year, month, d) for d in range(1, days_in_month + 1)
    ]

    rows = []
    counter = 1

    for sale_date in all_dates:
        # Filter events active on this date
        active_events = events_df[
            (events_df["start_date"] <= sale_date)
            & (events_df["end_date"] >= sale_date)
        ] if not events_df.empty else pd.DataFrame()

        for _, product in inventory.iterrows():
            product_id = product["product_id"]
            category = product["category"]

            sale_prob = DAILY_SALE_PROBABILITY.get(
                category, DEFAULT_SALE_PROBABILITY
            )

            if rng.random() > sale_prob:
                continue

            expected = _expected_daily_qty(
                product_id=product_id,
                category=category,
                sale_date=sale_date,
                active_events=active_events,
            )

            noise = rng.normal(0, expected * NOISE_STD_FRACTION)
            quantity = max(1, int(round(expected + noise)))

            rows.append({
                "transaction_id": f"ADV-{year}{month:02d}-{counter:05d}",
                "transaction_date": sale_date,
                "product_id": product_id,
                "quantity_sold": quantity,
            })
            counter += 1

    transactions = pd.DataFrame(rows, columns=[
        "transaction_id",
        "transaction_date",
        "product_id",
        "quantity_sold",
    ])

    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"]
    )

    return transactions


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_monthly_transactions(
    transactions: pd.DataFrame,
    year: int,
    month: int,
    output_base_folder: str | Path,
) -> Path:
    """
    Save a monthly transactions DataFrame to its year/month folder.

    Parameters
    ----------
    transactions : pd.DataFrame
        Monthly transactions from generate_monthly_transactions().
    year : int
        Year of the transactions.
    month : int
        Month of the transactions (1-12).
    output_base_folder : str or Path
        Root advanced output folder (data/processed/advanced).

    Returns
    -------
    Path
        Full path of the saved CSV file.
    """
    output_folder = (
        Path(output_base_folder) / f"year_{year}" / f"month_{month:02d}"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "transactions.csv"
    transactions.to_csv(output_path, index=False)

    return output_path


def save_five_year_transactions_csv(
    all_transactions: pd.DataFrame,
    output_base_folder: str | Path,
) -> Path:
    """
    Save the aggregated five-year transactions to the advanced root folder.

    Parameters
    ----------
    all_transactions : pd.DataFrame
        Combined transactions for all 60 months.
    output_base_folder : str or Path
        Root advanced output folder.

    Returns
    -------
    Path
        Full path of the saved CSV file.
    """
    output_folder = Path(output_base_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "advanced_transactions_5yr.csv"
    all_transactions.to_csv(output_path, index=False)

    return output_path


def save_five_year_transactions_to_sqlite(
    all_transactions: pd.DataFrame,
    sqlite_db_path: str | Path,
) -> None:
    """
    Save the aggregated five-year transactions DataFrame to SQLite.

    Table name: advanced_transactions_5yr

    Parameters
    ----------
    all_transactions : pd.DataFrame
        Combined transactions for all 60 months.
    sqlite_db_path : str or Path
        Path to the SQLite database file.
    """
    sqlite_db_path = Path(sqlite_db_path)
    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{sqlite_db_path}")

    all_transactions.to_sql(
        name="advanced_transactions_5yr",
        con=engine,
        if_exists="replace",
        index=False,
    )

    print(f"SQLite table saved    : advanced_transactions_5yr → {sqlite_db_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_five_year_generator(
    inventory_csv_path: str | Path = "data/raw/inventory.csv",
    output_base_folder: str | Path = "data/processed/advanced",
    sqlite_db_path: str | Path = "src/database/sari_sari_store.db",
    sales_events: pd.DataFrame = None,
    random_seed: int = 42,
    save_csv: bool = True,
    save_sqlite: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate and save five years of monthly synthetic transactions.

    Iterates January 2022 through December 2026 (60 months total).
    For each month, generates daily transactions for every product
    in inventory.csv and saves them to the year/month folder.
    Also saves one aggregated five-year CSV and SQLite table.

    Parameters
    ----------
    inventory_csv_path : str or Path
        Path to data/raw/inventory.csv.
    output_base_folder : str or Path
        Root output folder for advanced level.
    sqlite_db_path : str or Path
        Path to the SQLite database.
    sales_events : pd.DataFrame, optional
        Sales events DataFrame from run_sales_event_generator().
        Pass None to run without promotional boosts.
    random_seed : int
        Base seed for reproducible generation.
    save_csv : bool
        Whether to save monthly and aggregated CSV files.
    save_sqlite : bool
        Whether to save the aggregated table to SQLite.
    verbose : bool
        Whether to print progress to the console.

    Returns
    -------
    pd.DataFrame
        Aggregated five-year transactions DataFrame.
    """
    inventory = load_inventory(inventory_csv_path)

    if verbose:
        print("\n" + "=" * 72)
        print("ADVANCED GOAL: FIVE-YEAR TRANSACTION GENERATOR")
        print("=" * 72)
        print(f"Products loaded    : {len(inventory)}")
        print(f"Period             : {START_YEAR}–{END_YEAR} (60 months)")
        print(f"Output folder      : {output_base_folder}")
        print("-" * 72)

    all_monthly_frames = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            transactions = generate_monthly_transactions(
                inventory=inventory,
                year=year,
                month=month,
                sales_events=sales_events,
                random_seed=random_seed,
            )

            if save_csv:
                save_monthly_transactions(
                    transactions=transactions,
                    year=year,
                    month=month,
                    output_base_folder=output_base_folder,
                )

            all_monthly_frames.append(transactions)

            if verbose:
                print(
                    f"  {year}-{month:02d}  |  "
                    f"{len(transactions):>6,} transactions  |  "
                    f"{transactions['quantity_sold'].sum():>6,} units sold"
                )

    all_transactions = pd.concat(all_monthly_frames, ignore_index=True)

    if save_csv:
        csv_path = save_five_year_transactions_csv(
            all_transactions, output_base_folder
        )
        print(f"\nAggregated CSV saved  : {csv_path}")

    if save_sqlite:
        save_five_year_transactions_to_sqlite(
            all_transactions, sqlite_db_path
        )

    if verbose:
        print("-" * 72)
        print(f"Total transactions : {len(all_transactions):,}")
        print(f"Total units sold   : {all_transactions['quantity_sold'].sum():,}")
        print("Five-year generation complete.")
        print("=" * 72)

    return all_transactions


if __name__ == "__main__":
    run_five_year_generator()
