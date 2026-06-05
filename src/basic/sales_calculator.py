"""
sales_calculator.py

Basic Goal:
One-day Sari-Sari Store Sales Calculator.

This module:
- loads inventory.csv
- loads transactions.csv
- joins transactions with inventory
- calculates revenue, expenses, and gross profit
- calculates remaining stock
- creates a daily ledger summary
- saves outputs to CSV
- saves tables to SQLite using SQLAlchemy

Expected project structure:

LT6_Final_Project/
│
├── src/
│   ├── __init__.py
│   └── basic/
│       ├── __init__.py
│       ├── data_loader.py
│       └── sales_calculator.py
│
├── data/
│   └── raw/
│       ├── inventory.csv
│       └── transactions.csv
│
└── test/

Expected input files:
- data/raw/inventory.csv
- data/raw/transactions.csv

Generated output files:
- data/processed/basic/daily_transaction_details.csv
- data/processed/basic/daily_product_summary.csv
- data/processed/basic/daily_ledger_summary.csv

Generated SQLite database:
- src/database/sari_sari_store.db
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

try:
    from src.basic.data_loader import (
        load_inventory,
        load_transactions,
        validate_transactions_match_inventory,
    )
except ModuleNotFoundError:
    from data_loader import (
        load_inventory,
        load_transactions,
        validate_transactions_match_inventory,
    )


def create_sales_details(
    inventory: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join transactions with inventory and calculate transaction-level totals.

    Parameters
    ----------
    inventory:
        DataFrame loaded from inventory.csv.

    transactions:
        DataFrame loaded from transactions.csv.

    Returns
    -------
    pd.DataFrame
        Transaction-level details with product information and calculated fields.
    """

    sales_details = transactions.merge(
        inventory,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    sales_details["revenue"] = (
        sales_details["quantity_sold"] * sales_details["unit_price"]
    )

    sales_details["expense"] = (
        sales_details["quantity_sold"] * sales_details["unit_cost"]
    )

    sales_details["gross_profit"] = (
        sales_details["revenue"] - sales_details["expense"]
    )

    return sales_details


def create_product_summary(sales_details: pd.DataFrame) -> pd.DataFrame:
    """
    Create product-level summary for the day.

    This calculates:
    - starting stock
    - total quantity sold
    - total revenue
    - total expense
    - total gross profit
    - remaining stock
    - stock status
    """

    product_summary = (
        sales_details
        .groupby(
            [
                "product_id",
                "product_name",
                "category",
            ],
            as_index=False,
        )
        .agg(
            starting_stock=("starting_stock", "max"),
            unit_cost=("unit_cost", "max"),
            unit_price=("unit_price", "max"),
            total_quantity_sold=("quantity_sold", "sum"),
            total_revenue=("revenue", "sum"),
            total_expense=("expense", "sum"),
            total_gross_profit=("gross_profit", "sum"),
        )
    )

    product_summary["remaining_stock"] = (
        product_summary["starting_stock"]
        - product_summary["total_quantity_sold"]
    )

    product_summary["stock_status"] = product_summary["remaining_stock"].apply(
        lambda stock: "OK" if stock > 0 else "INSUFFICIENT STOCK"
    )

    return product_summary


def create_ledger_summary(
    sales_details: pd.DataFrame,
    product_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a one-row daily ledger summary.

    The ledger gives the final check for:
    - total transactions
    - total quantity sold
    - total revenue
    - total expenses
    - gross profit
    - stock issues
    - ledger status
    """

    transaction_dates = sales_details["transaction_date"].dt.date.unique()

    if len(transaction_dates) > 1:
        transaction_date = "MULTIPLE_DATES_FOUND"
    else:
        transaction_date = str(transaction_dates[0])

    total_revenue = sales_details["revenue"].sum()
    total_expense = sales_details["expense"].sum()
    gross_profit = total_revenue - total_expense

    insufficient_stock_products = (
        product_summary["remaining_stock"] < 1
    ).sum()

    ledger_status = (
        "BALANCED"
        if insufficient_stock_products == 0 and len(transaction_dates) == 1
        else "CHECK_LEDGER"
    )

    ledger_summary = pd.DataFrame(
        [
            {
                "transaction_date": transaction_date,
                "number_of_transactions": len(sales_details),
                "unique_products_sold": sales_details["product_id"].nunique(),
                "total_quantity_sold": sales_details["quantity_sold"].sum(),
                "total_revenue": total_revenue,
                "total_expense": total_expense,
                "gross_profit": gross_profit,
                "insufficient_stock_products": insufficient_stock_products,
                "ledger_status": ledger_status,
            }
        ]
    )

    return ledger_summary


def print_basic_report(
    ledger_summary: pd.DataFrame,
    product_summary: pd.DataFrame,
) -> None:
    """
    Print the Basic-level ledger and product summary.
    """

    ledger = ledger_summary.iloc[0]

    print("\n" + "=" * 72)
    print("BASIC GOAL: ONE-DAY SARI-SARI STORE SALES CALCULATOR")
    print("=" * 72)

    print(f"Transaction Date             : {ledger['transaction_date']}")
    print(f"Number of Transactions       : {ledger['number_of_transactions']}")
    print(f"Unique Products Sold         : {ledger['unique_products_sold']}")
    print(f"Total Quantity Sold          : {ledger['total_quantity_sold']}")
    print(f"Total Revenue                : PHP {ledger['total_revenue']:,.2f}")
    print(f"Total Expenses               : PHP {ledger['total_expense']:,.2f}")
    print(f"Gross Profit                 : PHP {ledger['gross_profit']:,.2f}")
    print(f"Insufficient Stock Products  : {ledger['insufficient_stock_products']}")
    print(f"Ledger Status                : {ledger['ledger_status']}")

    print("=" * 72)

    print("\nPRODUCT SUMMARY")
    print("-" * 72)
    print(product_summary.to_string(index=False))


def save_output_csv_files(
    sales_details: pd.DataFrame,
    product_summary: pd.DataFrame,
    ledger_summary: pd.DataFrame,
    output_folder: str | Path,
) -> None:
    """
    Save Basic-level output CSV files.

    Files created:
    - daily_transaction_details.csv
    - daily_product_summary.csv
    - daily_ledger_summary.csv
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    sales_details_path = output_folder / "daily_transaction_details.csv"
    product_summary_path = output_folder / "daily_product_summary.csv"
    ledger_summary_path = output_folder / "daily_ledger_summary.csv"

    sales_details.to_csv(sales_details_path, index=False)
    product_summary.to_csv(product_summary_path, index=False)
    ledger_summary.to_csv(ledger_summary_path, index=False)

    print("\nCSV outputs saved:")
    print(f"- {sales_details_path}")
    print(f"- {product_summary_path}")
    print(f"- {ledger_summary_path}")


def save_tables_to_sqlite(
    inventory: pd.DataFrame,
    transactions: pd.DataFrame,
    sales_details: pd.DataFrame,
    product_summary: pd.DataFrame,
    ledger_summary: pd.DataFrame,
    sqlite_db_path: str | Path,
) -> None:
    """
    Save input and output tables to SQLite using SQLAlchemy.

    Tables created:
    - inventory
    - transactions
    - daily_transaction_details
    - daily_product_summary
    - daily_ledger_summary
    """

    sqlite_db_path = Path(sqlite_db_path)
    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{sqlite_db_path}", connect_args={"timeout": 30})

    inventory.to_sql(
        name="inventory",
        con=engine,
        if_exists="replace",
        index=False,
    )

    transactions.to_sql(
        name="transactions",
        con=engine,
        if_exists="replace",
        index=False,
    )

    sales_details.to_sql(
        name="daily_transaction_details",
        con=engine,
        if_exists="replace",
        index=False,
    )

    product_summary.to_sql(
        name="daily_product_summary",
        con=engine,
        if_exists="replace",
        index=False,
    )

    ledger_summary.to_sql(
        name="daily_ledger_summary",
        con=engine,
        if_exists="replace",
        index=False,
    )

    print(f"\nSQLite database saved to: {sqlite_db_path}")

    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                ORDER BY name;
                """
            )
        )

        print("\nSQLite tables created:")
        for row in result:
            print(f"- {row[0]}")

    engine.dispose()


def run_basic_sales_calculator(
    inventory_csv_path: str | Path = "data/raw/inventory.csv",
    transactions_csv_path: str | Path = "data/raw/transactions.csv",
    output_folder: str | Path = "data/processed/basic",
    sqlite_db_path: str | Path = "src/database/sari_sari_store.db",
    save_csv_outputs: bool = True,
    save_sqlite_database: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full Basic-level workflow.

    Parameters
    ----------
    inventory_csv_path:
        Path to inventory.csv.

    transactions_csv_path:
        Path to transactions.csv.

    output_folder:
        Folder where output CSV files will be saved.

    sqlite_db_path:
        Path where the SQLite database will be created.

    save_csv_outputs:
        Whether to save calculated outputs as CSV files.

    save_sqlite_database:
        Whether to save tables into SQLite.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        sales_details, product_summary, ledger_summary
    """

    inventory = load_inventory(inventory_csv_path)
    transactions = load_transactions(transactions_csv_path)

    validate_transactions_match_inventory(
        transactions=transactions,
        inventory=inventory,
    )

    sales_details = create_sales_details(
        inventory=inventory,
        transactions=transactions,
    )

    product_summary = create_product_summary(sales_details)

    ledger_summary = create_ledger_summary(
        sales_details=sales_details,
        product_summary=product_summary,
    )

    print_basic_report(
        ledger_summary=ledger_summary,
        product_summary=product_summary,
    )

    if save_csv_outputs:
        save_output_csv_files(
            sales_details=sales_details,
            product_summary=product_summary,
            ledger_summary=ledger_summary,
            output_folder=output_folder,
        )

    if save_sqlite_database:
        save_tables_to_sqlite(
            inventory=inventory,
            transactions=transactions,
            sales_details=sales_details,
            product_summary=product_summary,
            ledger_summary=ledger_summary,
            sqlite_db_path=sqlite_db_path,
        )

    return sales_details, product_summary, ledger_summary


if __name__ == "__main__":
    run_basic_sales_calculator()
