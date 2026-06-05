"""
database.py

Database utilities for the Sari-Sari Store Simulator.

This file handles:
- Creating a SQLAlchemy engine
- Saving pandas DataFrames to SQLite tables
- Running simple SQL checks
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def get_engine(sqlite_db_path: str = "database/sari_sari_store.db") -> Engine:
    """
    Create and return a SQLAlchemy engine for a SQLite database.
    """

    sqlite_db_path = Path(sqlite_db_path)
    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{sqlite_db_path}")

    return engine


def save_dataframe_to_table(
    df: pd.DataFrame,
    table_name: str,
    engine: Engine,
    if_exists: str = "replace",
) -> None:
    """
    Save a pandas DataFrame into a SQLite table.

    Parameters
    ----------
    df:
        DataFrame to save.

    table_name:
        SQLite table name.

    engine:
        SQLAlchemy engine.

    if_exists:
        "replace", "append", or "fail".
    """

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=False,
    )


def list_tables(engine: Engine) -> list[str]:
    """
    Return a list of table names from the SQLite database.
    """

    query = text(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name;
        """
    )

    with engine.connect() as conn:
        result = conn.execute(query)
        tables = [row[0] for row in result]

    return tables


def print_tables(engine: Engine) -> None:
    """
    Print all tables in the SQLite database.
    """

    tables = list_tables(engine)

    print("\nSQLite tables created:")
    for table in tables:
        print(f"- {table}")
