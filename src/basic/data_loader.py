"""
data_loader.py

Basic-level data loading and validation module for the Sari-Sari Store Simulator.

This module handles the two input CSV files used in the Basic goal:

1. data/raw/inventory.csv
2. data/raw/transactions.csv

inventory.csv contains product-level information:
- product_id
- product_name
- category
- starting_stock
- unit_cost
- unit_price

transactions.csv contains one day's sales activity:
- transaction_id
- transaction_date
- product_id
- quantity_sold

The purpose of this module is to:
- load CSV files
- clean column names
- validate required columns
- validate data types
- check for missing or invalid values
- make sure every transaction product exists in inventory
"""

from pathlib import Path
import pandas as pd


INVENTORY_REQUIRED_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "starting_stock",
    "unit_cost",
    "unit_price",
]


TRANSACTIONS_REQUIRED_COLUMNS = [
    "transaction_id",
    "transaction_date",
    "product_id",
    "quantity_sold",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names into Python/SQL-friendly names.

    Examples
    --------
    "Product Name" becomes "product_name"
    "Unit-Price" becomes "unit_price"
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    file_label: str,
) -> None:
    """
    Check whether a DataFrame contains all required columns.

    Parameters
    ----------
    df:
        DataFrame to validate.

    required_columns:
        List of required column names.

    file_label:
        Name of the file being checked. Used for clearer error messages.
    """

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file_label} is missing required columns: "
            + ", ".join(missing_columns)
        )


def load_inventory(inventory_csv_path: str | Path) -> pd.DataFrame:
    """
    Load and validate inventory.csv.

    Required columns:
    - product_id
    - product_name
    - category
    - starting_stock
    - unit_cost
    - unit_price

    Returns
    -------
    pd.DataFrame
        Cleaned and validated inventory DataFrame.
    """

    inventory_csv_path = Path(inventory_csv_path)

    if not inventory_csv_path.exists():
        raise FileNotFoundError(
            f"Inventory CSV file not found: {inventory_csv_path}"
        )

    inventory = pd.read_csv(inventory_csv_path)
    inventory = clean_column_names(inventory)

    validate_required_columns(
        df=inventory,
        required_columns=INVENTORY_REQUIRED_COLUMNS,
        file_label="inventory.csv",
    )

    # Clean text columns
    inventory["product_id"] = inventory["product_id"].astype(str).str.strip()
    inventory["product_name"] = inventory["product_name"].astype(str).str.strip()
    inventory["category"] = inventory["category"].astype(str).str.strip()

    # Validate text columns are not empty
    text_columns = [
        "product_id",
        "product_name",
        "category",
    ]

    for column in text_columns:
        if inventory[column].eq("").any():
            raise ValueError(
                f"inventory.csv contains blank values in column: {column}"
            )

    # Convert and validate numeric columns
    numeric_columns = [
        "starting_stock",
        "unit_cost",
        "unit_price",
    ]

    for column in numeric_columns:
        inventory[column] = pd.to_numeric(
            inventory[column],
            errors="coerce",
        )

    if inventory[numeric_columns].isna().any().any():
        raise ValueError(
            "inventory.csv contains invalid or missing numeric values."
        )

    for column in numeric_columns:
        if (inventory[column] < 0).any():
            raise ValueError(
                f"inventory.csv contains negative values in column: {column}"
            )

    # Product IDs should be unique in inventory
    if inventory["product_id"].duplicated().any():
        duplicate_ids = (
            inventory.loc[inventory["product_id"].duplicated(), "product_id"]
            .unique()
            .tolist()
        )

        raise ValueError(
            "inventory.csv contains duplicate product_id values: "
            + ", ".join(duplicate_ids)
        )

    return inventory


def load_transactions(transactions_csv_path: str | Path) -> pd.DataFrame:
    """
    Load and validate transactions.csv.

    Required columns:
    - transaction_id
    - transaction_date
    - product_id
    - quantity_sold

    Returns
    -------
    pd.DataFrame
        Cleaned and validated transactions DataFrame.
    """

    transactions_csv_path = Path(transactions_csv_path)

    if not transactions_csv_path.exists():
        raise FileNotFoundError(
            f"Transactions CSV file not found: {transactions_csv_path}"
        )

    transactions = pd.read_csv(transactions_csv_path)
    transactions = clean_column_names(transactions)

    validate_required_columns(
        df=transactions,
        required_columns=TRANSACTIONS_REQUIRED_COLUMNS,
        file_label="transactions.csv",
    )

    # Clean text columns
    transactions["transaction_id"] = (
        transactions["transaction_id"].astype(str).str.strip()
    )

    transactions["product_id"] = (
        transactions["product_id"].astype(str).str.strip()
    )

    # Validate text columns are not empty
    text_columns = [
        "transaction_id",
        "product_id",
    ]

    for column in text_columns:
        if transactions[column].eq("").any():
            raise ValueError(
                f"transactions.csv contains blank values in column: {column}"
            )

    # Convert date column
    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"],
        errors="coerce",
    )

    if transactions["transaction_date"].isna().any():
        raise ValueError(
            "transactions.csv contains invalid transaction_date values."
        )

    # Convert quantity_sold column
    transactions["quantity_sold"] = pd.to_numeric(
        transactions["quantity_sold"],
        errors="coerce",
    )

    if transactions["quantity_sold"].isna().any():
        raise ValueError(
            "transactions.csv contains invalid quantity_sold values."
        )

    if (transactions["quantity_sold"] < 0).any():
        raise ValueError(
            "transactions.csv contains negative quantity_sold values."
        )

    # Transaction IDs should be unique
    if transactions["transaction_id"].duplicated().any():
        duplicate_ids = (
            transactions.loc[
                transactions["transaction_id"].duplicated(),
                "transaction_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "transactions.csv contains duplicate transaction_id values: "
            + ", ".join(duplicate_ids)
        )

    return transactions


def validate_transactions_match_inventory(
    transactions: pd.DataFrame,
    inventory: pd.DataFrame,
) -> None:
    """
    Check that every product_id in transactions.csv exists in inventory.csv.

    Parameters
    ----------
    transactions:
        Validated transactions DataFrame.

    inventory:
        Validated inventory DataFrame.
    """

    transaction_product_ids = set(transactions["product_id"])
    inventory_product_ids = set(inventory["product_id"])

    missing_product_ids = transaction_product_ids - inventory_product_ids

    if missing_product_ids:
        raise ValueError(
            "transactions.csv contains product_id values not found in inventory.csv: "
            + ", ".join(sorted(missing_product_ids))
        )
