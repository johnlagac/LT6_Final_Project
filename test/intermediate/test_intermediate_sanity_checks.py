"""Sanity checks for the Intermediate Sari-Sari Store simulator.

Run from the project root:

    python test/test_intermediate_sanity_checks.py

The checks are intentionally simple assert-based tests so they can be read in a
notebook or terminal without requiring pytest.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intermediate.monthly_simulator import (  # noqa: E402
    DATABASE_PATH,
    INTERMEDIATE_OUTPUT_DIR,
    RAW_INVENTORY_PATH,
    SQL_TABLE_NAMES,
    run_intermediate_monthly_simulator,
)


def file_hash(path: Path) -> str:
    """Return a SHA256 hash for a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(actual: float, expected: float, tolerance: float = 0.01) -> None:
    """Assert that two numeric values are almost equal."""
    assert abs(float(actual) - float(expected)) <= tolerance, (actual, expected)


def check_required_files_exist() -> None:
    """Check that the required raw inventory file exists."""
    assert RAW_INVENTORY_PATH.exists(), (
        "Missing data/raw/inventory.csv. Run this from the project root and "
        "confirm the raw inventory file is present."
    )


def check_output_files_exist() -> None:
    """Check that all expected Intermediate CSV files were created."""
    expected_files = [
        "monthly_transactions.csv",
        "inventory_before_monthly_sales.csv",
        "monthly_transaction_details.csv",
        "monthly_product_summary.csv",
        "monthly_ledger_summary.csv",
        "inventory_after_monthly_sales.csv",
        "restock_recommendations.csv",
        "monthly_dashboard_data.csv",
    ]
    for filename in expected_files:
        assert (INTERMEDIATE_OUTPUT_DIR / filename).exists(), filename


def check_formula_correctness(outputs: dict[str, pd.DataFrame]) -> None:
    """Check revenue, expense, profit, and stock formulas."""
    details = outputs["monthly_transaction_details"]
    product_summary = outputs["monthly_product_summary"]
    ledger = outputs["monthly_ledger_summary"].iloc[0]

    assert not details.empty, "Generated transaction details should not be empty."

    assert_close(
        (details["quantity_sold"] * details["unit_price"]).sum(),
        details["revenue"].sum(),
    )
    assert_close(
        (details["quantity_sold"] * details["unit_cost"]).sum(),
        details["expense"].sum(),
    )
    assert_close(
        (details["revenue"] - details["expense"]).sum(),
        details["gross_profit"].sum(),
    )

    product_check = product_summary.copy()
    expected_remaining = (
        product_check["starting_stock"] - product_check["total_quantity_sold"]
    )
    assert (
        product_check["remaining_stock"].astype(int).to_numpy()
        == expected_remaining.astype(int).to_numpy()
    ).all(), "Remaining stock should equal starting stock minus units sold."

    assert_close(ledger["total_revenue"], details["revenue"].sum())
    assert_close(ledger["total_expense"], details["expense"].sum())
    assert_close(ledger["gross_profit"], details["gross_profit"].sum())
    assert int(ledger["total_quantity_sold"]) == int(details["quantity_sold"].sum())


def check_restock_output(outputs: dict[str, pd.DataFrame]) -> None:
    """Check that restock recommendations have sensible fields."""
    restock = outputs["restock_recommendations"]
    required_columns = {
        "product_id",
        "product_name",
        "category",
        "remaining_stock",
        "reorder_point",
        "recommend_restock",
        "recommended_restock_quantity",
        "stockout_risk",
        "restock_reason",
    }
    assert required_columns.issubset(restock.columns)
    assert (restock["recommended_restock_quantity"] >= 0).all()

    recommended = restock[restock["recommend_restock"].astype(bool)]
    if not recommended.empty:
        assert (recommended["recommended_restock_quantity"] > 0).all()


def check_dashboard_output(outputs: dict[str, pd.DataFrame]) -> None:
    """Check that dashboard-ready data covers the required sections."""
    dashboard = outputs["monthly_dashboard_data"]
    required_groups = {
        "kpi",
        "daily_revenue_trend",
        "sales_by_category",
        "top_selling_products",
        "lowest_stock_products",
    }
    assert required_groups.issubset(set(dashboard["metric_group"])), dashboard


def check_sqlite_tables_exist() -> None:
    """Check that all Intermediate tables were written to SQLite."""
    assert DATABASE_PATH.exists(), "SQLite database was not created."
    with sqlite3.connect(DATABASE_PATH) as connection:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type = 'table';",
            connection,
        )["name"].tolist()
    for table_name in SQL_TABLE_NAMES.values():
        assert table_name in tables, f"Missing SQLite table: {table_name}"


def main() -> None:
    """Run all Intermediate sanity checks."""
    check_required_files_exist()
    before_hash = file_hash(RAW_INVENTORY_PATH)

    outputs = run_intermediate_monthly_simulator(
        month="2026-01",
        benchmark_info={
            "average_daily_customers": 50,
            "weekend_multiplier": 1.20,
            "payday_multiplier": 1.40,
            "category_multipliers": {
                "beverage": 1.35,
                "snacks": 1.30,
                "food": 1.20,
            },
        },
        random_seed=512,
        save_outputs=True,
        create_charts=True,
    )

    after_hash = file_hash(RAW_INVENTORY_PATH)
    assert before_hash == after_hash, "Raw inventory.csv should not be modified."

    check_output_files_exist()
    check_formula_correctness(outputs)
    check_restock_output(outputs)
    check_dashboard_output(outputs)
    check_sqlite_tables_exist()

    print("All Intermediate sanity checks passed.")


if __name__ == "__main__":
    main()
