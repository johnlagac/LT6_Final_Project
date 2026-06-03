"""Monthly simulator for the Intermediate Sari-Sari Store project level.

The main public function is ``run_intermediate_monthly_simulator``. It loads
``data/raw/inventory.csv``, generates or loads one month of transactions,
calculates revenue, expense, gross profit, remaining stock, restocking needs,
and writes CSV plus SQLite outputs using level-specific names.
"""

from __future__ import annotations

from calendar import monthrange
from pathlib import Path
from typing import Any

import pandas as pd

from src.intermediate.dashboard_report import (
    build_monthly_dashboard_data,
    save_dashboard_charts,
)
from src.intermediate.data_generator import (
    generate_monthly_transactions,
    parse_month,
    validate_inventory_columns,
)
from src.intermediate.restock_calculator import create_restock_recommendations

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_INVENTORY_PATH = PROJECT_ROOT / "data" / "raw" / "inventory.csv"
INTERMEDIATE_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "intermediate"
DATABASE_PATH = PROJECT_ROOT / "src" / "database" / "sari_sari_store.db"

REQUIRED_TRANSACTION_COLUMNS = {
    "transaction_id",
    "transaction_date",
    "product_id",
    "quantity_sold",
}

CSV_OUTPUT_FILES = {
    "monthly_transactions": "monthly_transactions.csv",
    "inventory_before_monthly_sales": "inventory_before_monthly_sales.csv",
    "monthly_transaction_details": "monthly_transaction_details.csv",
    "monthly_product_summary": "monthly_product_summary.csv",
    "monthly_ledger_summary": "monthly_ledger_summary.csv",
    "inventory_after_monthly_sales": "inventory_after_monthly_sales.csv",
    "restock_recommendations": "restock_recommendations.csv",
    "monthly_dashboard_data": "monthly_dashboard_data.csv",
}

SQL_TABLE_NAMES = {
    "monthly_transactions": "intermediate_monthly_transactions",
    "inventory_before_monthly_sales": "intermediate_inventory_before_monthly_sales",
    "monthly_transaction_details": "intermediate_monthly_transaction_details",
    "monthly_product_summary": "intermediate_monthly_product_summary",
    "monthly_ledger_summary": "intermediate_monthly_ledger_summary",
    "inventory_after_monthly_sales": "intermediate_inventory_after_monthly_sales",
    "restock_recommendations": "intermediate_restock_recommendations",
    "monthly_dashboard_data": "intermediate_monthly_dashboard_data",
}


def load_intermediate_inventory(
    inventory_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the raw inventory file used by the Intermediate simulator.

    The function first attempts to reuse ``src.basic.data_loader.load_inventory``
    when available. If the Basic loader has a different signature or is not
    available, it safely falls back to ``pandas.read_csv``. This keeps the
    Intermediate level compatible with the existing Basic project while still
    being testable as a standalone module.

    Parameters
    ----------
    inventory_path : str, pathlib.Path, or None, optional
        Location of the raw inventory file. Defaults to
        ``data/raw/inventory.csv`` under the project root.

    Returns
    -------
    pandas.DataFrame
        Validated inventory data.
    """
    path = Path(inventory_path) if inventory_path is not None else RAW_INVENTORY_PATH

    try:
        from src.basic.data_loader import load_inventory

        try:
            inventory = load_inventory(path)
        except TypeError:
            inventory = load_inventory()
    except Exception:
        inventory = pd.read_csv(path)

    validate_inventory_columns(inventory)
    return inventory.copy()


def validate_transactions(transactions: pd.DataFrame) -> None:
    """Validate monthly transaction columns and values.

    Parameters
    ----------
    transactions : pandas.DataFrame
        Monthly transactions to validate.

    Raises
    ------
    ValueError
        If required columns are missing or quantity values are invalid.
    """
    missing_columns = REQUIRED_TRANSACTION_COLUMNS.difference(
        transactions.columns
    )
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Transactions file is missing columns: {missing}")

    invalid_quantities = pd.to_numeric(
        transactions["quantity_sold"], errors="coerce"
    ).isna() | (pd.to_numeric(transactions["quantity_sold"], errors="coerce") < 0)
    if invalid_quantities.any():
        raise ValueError("Transactions contain missing or negative quantities.")


def load_or_generate_monthly_transactions(
    inventory: pd.DataFrame,
    month: str | int = "2026-01",
    year: int | None = None,
    transactions_path: str | Path | None = None,
    benchmark_info: dict[str, Any] | None = None,
    random_seed: int | None = 512,
) -> pd.DataFrame:
    """Load monthly transactions or generate them when no file is supplied.

    Parameters
    ----------
    inventory : pandas.DataFrame
        Inventory data used when synthetic transactions are generated.
    month : str or int, default="2026-01"
        Month to simulate.
    year : int or None, optional
        Year used when ``month`` is an integer.
    transactions_path : str, pathlib.Path, or None, optional
        Optional monthly transaction CSV. When omitted, synthetic transactions
        are generated.
    benchmark_info : dict or None, optional
        Sari-sari store benchmark assumptions for synthetic data generation.
    random_seed : int or None, default=512
        Seed for reproducible synthetic output.

    Returns
    -------
    pandas.DataFrame
        Monthly transaction table.
    """
    if transactions_path is not None and Path(transactions_path).exists():
        transactions = pd.read_csv(transactions_path)
    else:
        transactions = generate_monthly_transactions(
            inventory=inventory,
            month=month,
            year=year,
            benchmark_info=benchmark_info,
            random_seed=random_seed,
        )

    validate_transactions(transactions)
    transactions = transactions.copy()
    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"]
    ).dt.date.astype(str)
    transactions["quantity_sold"] = pd.to_numeric(
        transactions["quantity_sold"], errors="raise"
    ).astype(int)
    return transactions


def build_monthly_transaction_details(
    transactions: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Join transactions with inventory and calculate financial fields.

    Parameters
    ----------
    transactions : pandas.DataFrame
        Monthly transactions with product IDs and quantities sold.
    inventory : pandas.DataFrame
        Raw product master data.

    Returns
    -------
    pandas.DataFrame
        Transaction details with revenue, expense, gross profit, product-level
        total quantity sold, and remaining stock.
    """
    validate_transactions(transactions)
    validate_inventory_columns(inventory)

    details = transactions.merge(inventory, on="product_id", how="left")
    missing_products = details[details["product_name"].isna()]["product_id"].unique()
    if len(missing_products) > 0:
        missing = ", ".join(map(str, missing_products))
        raise ValueError(f"Transactions contain unknown product IDs: {missing}")

    numeric_columns = ["quantity_sold", "starting_stock", "unit_cost", "unit_price"]
    for column in numeric_columns:
        details[column] = pd.to_numeric(details[column], errors="raise")

    total_sold = details.groupby("product_id")["quantity_sold"].transform("sum")
    details["total_quantity_sold"] = total_sold.astype(int)
    details["remaining_stock"] = (
        details["starting_stock"] - details["total_quantity_sold"]
    ).astype(int)

    oversold = details[details["remaining_stock"] < 0]["product_id"].unique()
    if len(oversold) > 0:
        product_list = ", ".join(map(str, oversold))
        raise ValueError(
            "Monthly transactions sold more units than starting stock for "
            f"product IDs: {product_list}"
        )

    details["revenue"] = details["quantity_sold"] * details["unit_price"]
    details["expense"] = details["quantity_sold"] * details["unit_cost"]
    details["gross_profit"] = details["revenue"] - details["expense"]

    output_columns = [
        "transaction_id",
        "transaction_date",
        "product_id",
        "product_name",
        "category",
        "quantity_sold",
        "unit_cost",
        "unit_price",
        "revenue",
        "expense",
        "gross_profit",
        "starting_stock",
        "total_quantity_sold",
        "remaining_stock",
    ]
    return details[output_columns].sort_values(
        ["transaction_date", "transaction_id"]
    )


def build_monthly_product_summary(
    transaction_details: pd.DataFrame,
    inventory: pd.DataFrame,
    month: str | int = "2026-01",
    year: int | None = None,
) -> pd.DataFrame:
    """Create product-level monthly sales and inventory summary.

    Parameters
    ----------
    transaction_details : pandas.DataFrame
        Transaction-level details with computed financial fields.
    inventory : pandas.DataFrame
        Raw inventory master data.
    month : str or int, default="2026-01"
        Month being summarized.
    year : int or None, optional
        Year used when ``month`` is an integer.

    Returns
    -------
    pandas.DataFrame
        Product-level sales, profit, and stock summary.
    """
    month_start = parse_month(month, year)
    days_in_month = monthrange(month_start.year, month_start.month)[1]

    if transaction_details.empty:
        sales_summary = pd.DataFrame(
            columns=[
                "product_id",
                "total_quantity_sold",
                "total_revenue",
                "total_expense",
                "gross_profit",
                "transaction_count",
            ]
        )
    else:
        sales_summary = (
            transaction_details.groupby("product_id", as_index=False)
            .agg(
                total_quantity_sold=("quantity_sold", "sum"),
                total_revenue=("revenue", "sum"),
                total_expense=("expense", "sum"),
                gross_profit=("gross_profit", "sum"),
                transaction_count=("transaction_id", "nunique"),
            )
        )

    product_summary = inventory.merge(sales_summary, on="product_id", how="left")
    fill_zero_columns = [
        "total_quantity_sold",
        "total_revenue",
        "total_expense",
        "gross_profit",
        "transaction_count",
    ]
    for column in fill_zero_columns:
        product_summary[column] = product_summary[column].fillna(0)

    product_summary["total_quantity_sold"] = product_summary[
        "total_quantity_sold"
    ].astype(int)
    product_summary["transaction_count"] = product_summary[
        "transaction_count"
    ].astype(int)
    product_summary["remaining_stock"] = (
        product_summary["starting_stock"] - product_summary["total_quantity_sold"]
    ).astype(int)
    product_summary["average_daily_sales"] = (
        product_summary["total_quantity_sold"] / days_in_month
    ).round(2)
    product_summary["sell_through_rate"] = (
        product_summary["total_quantity_sold"]
        / product_summary["starting_stock"].replace(0, pd.NA)
    ).fillna(0).round(4)
    product_summary["gross_margin_rate"] = (
        product_summary["gross_profit"]
        / product_summary["total_revenue"].replace(0, pd.NA)
    ).fillna(0).round(4)

    output_columns = [
        "product_id",
        "product_name",
        "category",
        "starting_stock",
        "unit_cost",
        "unit_price",
        "total_quantity_sold",
        "total_revenue",
        "total_expense",
        "gross_profit",
        "gross_margin_rate",
        "transaction_count",
        "remaining_stock",
        "average_daily_sales",
        "sell_through_rate",
    ]
    return product_summary[output_columns].sort_values("product_id")


def build_monthly_ledger_summary(
    transaction_details: pd.DataFrame,
    month: str | int = "2026-01",
    year: int | None = None,
) -> pd.DataFrame:
    """Create one-row monthly ledger summary.

    Parameters
    ----------
    transaction_details : pandas.DataFrame
        Transaction-level details with computed financial fields.
    month : str or int, default="2026-01"
        Month being summarized.
    year : int or None, optional
        Year used when ``month`` is an integer.

    Returns
    -------
    pandas.DataFrame
        One-row monthly financial ledger summary.
    """
    month_start = parse_month(month, year)
    days_in_month = monthrange(month_start.year, month_start.month)[1]
    month_end = month_start + pd.offsets.MonthEnd(0)

    total_revenue = float(transaction_details["revenue"].sum()) if not transaction_details.empty else 0.0
    total_expense = float(transaction_details["expense"].sum()) if not transaction_details.empty else 0.0
    gross_profit = float(transaction_details["gross_profit"].sum()) if not transaction_details.empty else 0.0
    total_quantity_sold = int(transaction_details["quantity_sold"].sum()) if not transaction_details.empty else 0
    transaction_count = int(transaction_details["transaction_id"].nunique()) if not transaction_details.empty else 0
    unique_products_sold = int(transaction_details["product_id"].nunique()) if not transaction_details.empty else 0
    gross_margin_rate = gross_profit / total_revenue if total_revenue else 0.0

    return pd.DataFrame(
        [
            {
                "month": month_start.strftime("%Y-%m"),
                "month_start": month_start.date().isoformat(),
                "month_end": month_end.date().isoformat(),
                "days_in_month": days_in_month,
                "transaction_count": transaction_count,
                "unique_products_sold": unique_products_sold,
                "total_quantity_sold": total_quantity_sold,
                "total_revenue": round(total_revenue, 2),
                "total_expense": round(total_expense, 2),
                "gross_profit": round(gross_profit, 2),
                "gross_margin_rate": round(gross_margin_rate, 4),
            }
        ]
    )


def build_inventory_after_sales(product_summary: pd.DataFrame) -> pd.DataFrame:
    """Create inventory-after-sales file for the next-month restock process.

    Parameters
    ----------
    product_summary : pandas.DataFrame
        Product-level monthly summary.

    Returns
    -------
    pandas.DataFrame
        Inventory table with updated starting stock for the next period.
    """
    inventory_after = product_summary[
        [
            "product_id",
            "product_name",
            "category",
            "remaining_stock",
            "unit_cost",
            "unit_price",
        ]
    ].copy()
    inventory_after = inventory_after.rename(
        columns={"remaining_stock": "starting_stock"}
    )
    return inventory_after.sort_values("product_id")


def save_intermediate_outputs(
    outputs: dict[str, pd.DataFrame],
    output_dir: str | Path = INTERMEDIATE_OUTPUT_DIR,
    database_path: str | Path = DATABASE_PATH,
) -> None:
    """Save Intermediate DataFrames as CSV files and SQLite tables.

    Parameters
    ----------
    outputs : dict[str, pandas.DataFrame]
        Dictionary of output names and DataFrames.
    output_dir : str or pathlib.Path, default=INTERMEDIATE_OUTPUT_DIR
        Folder for generated Intermediate CSV files.
    database_path : str or pathlib.Path, default=DATABASE_PATH
        SQLite database path used by all project levels.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    database_file = Path(database_path)
    database_file.parent.mkdir(parents=True, exist_ok=True)

    connection = None
    try:
        from sqlalchemy import create_engine

        connection = create_engine(f"sqlite:///{database_file}")
    except ImportError:
        import sqlite3

        connection = sqlite3.connect(database_file)

    try:
        for output_name, dataframe in outputs.items():
            if output_name in CSV_OUTPUT_FILES:
                dataframe.to_csv(output_path / CSV_OUTPUT_FILES[output_name], index=False)
            if output_name in SQL_TABLE_NAMES:
                dataframe.to_sql(
                    SQL_TABLE_NAMES[output_name],
                    con=connection,
                    if_exists="replace",
                    index=False,
                )
    finally:
        if hasattr(connection, "close"):
            connection.close()


def run_intermediate_monthly_simulator(
    month: str | int = "2026-01",
    year: int | None = None,
    inventory_path: str | Path | None = None,
    transactions_path: str | Path | None = None,
    benchmark_info: dict[str, Any] | None = None,
    random_seed: int | None = 512,
    output_dir: str | Path = INTERMEDIATE_OUTPUT_DIR,
    database_path: str | Path = DATABASE_PATH,
    save_outputs: bool = True,
    create_charts: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the full Intermediate monthly simulator.

    Parameters
    ----------
    month : str or int, default="2026-01"
        Month to simulate or summarize.
    year : int or None, optional
        Year used when ``month`` is an integer.
    inventory_path : str, pathlib.Path, or None, optional
        Raw inventory CSV path. Defaults to ``data/raw/inventory.csv``.
    transactions_path : str, pathlib.Path, or None, optional
        Optional monthly transaction CSV. If not supplied, transactions are
        generated synthetically.
    benchmark_info : dict or None, optional
        Benchmark assumptions influencing synthetic transaction generation.
    random_seed : int or None, default=512
        Seed for reproducible synthetic transactions.
    output_dir : str or pathlib.Path, default=INTERMEDIATE_OUTPUT_DIR
        Folder where Intermediate CSV outputs will be saved.
    database_path : str or pathlib.Path, default=DATABASE_PATH
        SQLite database path for level-specific output tables.
    save_outputs : bool, default=True
        When True, save CSV and SQLite outputs.
    create_charts : bool, default=True
        When True, save simple dashboard PNG charts under the output folder.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary containing all major Intermediate output DataFrames.
    """
    inventory = load_intermediate_inventory(inventory_path)
    transactions = load_or_generate_monthly_transactions(
        inventory=inventory,
        month=month,
        year=year,
        transactions_path=transactions_path,
        benchmark_info=benchmark_info,
        random_seed=random_seed,
    )
    transaction_details = build_monthly_transaction_details(transactions, inventory)
    product_summary = build_monthly_product_summary(
        transaction_details, inventory, month=month, year=year
    )
    ledger_summary = build_monthly_ledger_summary(
        transaction_details, month=month, year=year
    )
    inventory_after_sales = build_inventory_after_sales(product_summary)
    restock_recommendations = create_restock_recommendations(product_summary)
    dashboard_data = build_monthly_dashboard_data(
        transaction_details=transaction_details,
        product_summary=product_summary,
        ledger_summary=ledger_summary,
        restock_recommendations=restock_recommendations,
    )

    outputs = {
        "monthly_transactions": transactions,
        "inventory_before_monthly_sales": inventory.copy(),
        "monthly_transaction_details": transaction_details,
        "monthly_product_summary": product_summary,
        "monthly_ledger_summary": ledger_summary,
        "inventory_after_monthly_sales": inventory_after_sales,
        "restock_recommendations": restock_recommendations,
        "monthly_dashboard_data": dashboard_data,
    }

    if save_outputs:
        save_intermediate_outputs(outputs, output_dir, database_path)
        if create_charts:
            save_dashboard_charts(transaction_details, product_summary, output_dir)

    return outputs


if __name__ == "__main__":
    run_intermediate_monthly_simulator()
