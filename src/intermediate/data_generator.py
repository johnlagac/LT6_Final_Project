"""Synthetic monthly transaction generator for the Intermediate simulator.

This module creates one month of sari-sari store transactions from the raw
inventory master file. The generator accepts benchmark assumptions so the mock
sales can reflect common sari-sari store conditions such as higher payday
traffic, stronger weekend demand, category differences, and affordable-item
turnover.
"""

from __future__ import annotations

from calendar import monthrange
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_INVENTORY_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "starting_stock",
    "unit_cost",
    "unit_price",
}

DEFAULT_BENCHMARK_INFO: dict[str, Any] = {
    "average_daily_customers": 45,
    "average_items_per_customer": 2.2,
    "weekend_multiplier": 1.18,
    "payday_multiplier": 1.35,
    "promo_multiplier": 1.10,
    "price_sensitivity": 0.18,
    "stock_safety_factor": 0.95,
    "max_units_per_transaction": 5,
    "category_multipliers": {
        "beverage": 1.30,
        "drinks": 1.30,
        "food": 1.25,
        "snacks": 1.25,
        "noodles": 1.20,
        "canned goods": 1.05,
        "household": 0.80,
        "personal care": 0.75,
        "others": 1.00,
    },
}


def validate_inventory_columns(inventory: pd.DataFrame) -> None:
    """Raise a clear error when the inventory file is missing columns.

    Parameters
    ----------
    inventory : pandas.DataFrame
        Product master data loaded from ``data/raw/inventory.csv``.

    Raises
    ------
    ValueError
        If at least one required inventory column is missing.
    """
    missing_columns = REQUIRED_INVENTORY_COLUMNS.difference(inventory.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Inventory file is missing columns: {missing}")


def normalize_benchmark_info(
    benchmark_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return benchmark information merged with safe default values.

    Parameters
    ----------
    benchmark_info : dict or None, optional
        Assumptions that influence generated data. Examples include
        ``average_daily_customers``, ``weekend_multiplier``,
        ``payday_multiplier``, and ``category_multipliers``.

    Returns
    -------
    dict
        Complete benchmark settings used by the generator.
    """
    normalized = DEFAULT_BENCHMARK_INFO.copy()
    normalized["category_multipliers"] = DEFAULT_BENCHMARK_INFO[
        "category_multipliers"
    ].copy()

    if not benchmark_info:
        return normalized

    for key, value in benchmark_info.items():
        if key == "category_multipliers" and isinstance(value, dict):
            normalized["category_multipliers"].update(value)
        else:
            normalized[key] = value

    return normalized


def get_category_multiplier(category: str, benchmark_info: dict[str, Any]) -> float:
    """Return the sales multiplier assigned to a product category.

    Parameters
    ----------
    category : str
        Product category from the inventory file.
    benchmark_info : dict
        Benchmark assumptions containing category multipliers.

    Returns
    -------
    float
        Demand multiplier for the product category.
    """
    category_text = str(category).strip().lower()
    multipliers = benchmark_info.get("category_multipliers", {})

    if category_text in multipliers:
        return float(multipliers[category_text])

    for keyword, multiplier in multipliers.items():
        if keyword in category_text:
            return float(multiplier)

    return float(multipliers.get("others", 1.0))


def get_day_multiplier(sale_date: pd.Timestamp, benchmark_info: dict[str, Any]) -> float:
    """Return the traffic multiplier for a calendar date.

    Parameters
    ----------
    sale_date : pandas.Timestamp
        Date being simulated.
    benchmark_info : dict
        Benchmark assumptions such as weekend and payday effects.

    Returns
    -------
    float
        Combined demand multiplier for the date.
    """
    multiplier = 1.0

    if sale_date.weekday() >= 5:
        multiplier *= float(benchmark_info.get("weekend_multiplier", 1.0))

    if sale_date.day in {15, 30}:
        multiplier *= float(benchmark_info.get("payday_multiplier", 1.0))

    return multiplier


def parse_month(month: str | int, year: int | None = None) -> pd.Timestamp:
    """Convert month input into the first day of that month.

    Parameters
    ----------
    month : str or int
        Month to simulate. Accepts ``"YYYY-MM"``, ``"YYYY-MM-DD"``, or an
        integer month when ``year`` is also supplied.
    year : int or None, optional
        Year used when ``month`` is given as an integer.

    Returns
    -------
    pandas.Timestamp
        First calendar day of the requested month.
    """
    if isinstance(month, int):
        if year is None:
            raise ValueError("A year is required when month is an integer.")
        return pd.Timestamp(year=int(year), month=int(month), day=1)

    parsed = pd.to_datetime(str(month), errors="raise")
    return pd.Timestamp(year=parsed.year, month=parsed.month, day=1)


def generate_monthly_transactions(
    inventory: pd.DataFrame,
    month: str | int = "2026-01",
    year: int | None = None,
    benchmark_info: dict[str, Any] | None = None,
    random_seed: int | None = 512,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate one month of sari-sari store transactions.

    Parameters
    ----------
    inventory : pandas.DataFrame
        Product master data with product IDs, names, categories, starting
        stock, unit cost, and unit price.
    month : str or int, default="2026-01"
        Month to simulate. Use ``"YYYY-MM"`` for clarity.
    year : int or None, optional
        Year used only when ``month`` is an integer.
    benchmark_info : dict or None, optional
        Sari-sari store benchmark assumptions that influence mock sales.
    random_seed : int or None, default=512
        Seed for reproducible output. Use ``None`` for random results.
    output_path : str, pathlib.Path, or None, optional
        Optional CSV path for saving generated transactions.

    Returns
    -------
    pandas.DataFrame
        Monthly transactions with ``transaction_id``, ``transaction_date``,
        ``product_id``, and ``quantity_sold``.
    """
    validate_inventory_columns(inventory)
    benchmark = normalize_benchmark_info(benchmark_info)
    rng = np.random.default_rng(random_seed)

    inventory_work = inventory.copy()
    inventory_work["starting_stock"] = pd.to_numeric(
        inventory_work["starting_stock"], errors="raise"
    ).astype(int)
    inventory_work["unit_price"] = pd.to_numeric(
        inventory_work["unit_price"], errors="raise"
    ).astype(float)

    month_start = parse_month(month, year)
    days_in_month = monthrange(month_start.year, month_start.month)[1]
    dates = pd.date_range(month_start, periods=days_in_month, freq="D")

    median_price = max(float(inventory_work["unit_price"].median()), 1.0)
    expected_daily_units_total = (
        float(benchmark.get("average_daily_customers", 45))
        * float(benchmark.get("average_items_per_customer", 2.2))
    )
    base_units_per_product = expected_daily_units_total / max(len(inventory_work), 1)
    price_sensitivity = float(benchmark.get("price_sensitivity", 0.18))
    max_units = int(benchmark.get("max_units_per_transaction", 5))

    running_stock = dict(
        zip(inventory_work["product_id"], inventory_work["starting_stock"])
    )
    transactions: list[dict[str, Any]] = []
    transaction_counter = 1

    for sale_date in dates:
        day_multiplier = get_day_multiplier(sale_date, benchmark)

        for _, product in inventory_work.iterrows():
            product_id = product["product_id"]
            current_stock = int(running_stock.get(product_id, 0))
            if current_stock <= 0:
                continue

            category_multiplier = get_category_multiplier(
                str(product["category"]), benchmark
            )
            price_ratio = median_price / max(float(product["unit_price"]), 1.0)
            price_multiplier = 1 + (price_ratio - 1) * price_sensitivity
            expected_units = max(
                base_units_per_product
                * day_multiplier
                * category_multiplier
                * price_multiplier,
                0.05,
            )

            quantity_sold = int(rng.poisson(expected_units))
            if quantity_sold <= 0:
                continue

            quantity_sold = min(quantity_sold, max_units, current_stock)
            running_stock[product_id] = current_stock - quantity_sold
            transactions.append(
                {
                    "transaction_id": f"IM-{month_start:%Y%m}-{transaction_counter:05d}",
                    "transaction_date": sale_date.date().isoformat(),
                    "product_id": product_id,
                    "quantity_sold": quantity_sold,
                }
            )
            transaction_counter += 1

    transactions_df = pd.DataFrame(
        transactions,
        columns=[
            "transaction_id",
            "transaction_date",
            "product_id",
            "quantity_sold",
        ],
    )

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        transactions_df.to_csv(output_file, index=False)

    return transactions_df
