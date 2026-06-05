"""
monthly_outputs.py

Generates the per-month output files that were missing from the Advanced
pipeline to satisfy the project requirement that each monthly folder contains
all seven files:

    transactions.csv              ← already saved by five_year_generator.py
    transaction_details.csv       ← NEW (this module)
    product_summary.csv           ← NEW (this module)
    ledger_summary.csv            ← NEW (this module)
    restock_recommendations.csv   ← already saved by inventory_optimizer.py
    customer_feedback.csv         ← already saved by feedback_generator.py
    sales_events.csv              ← already saved by sales_event_generator.py

Also saves:
    inventory_before_monthly_sales.csv  ← NEW (this module)
        Snapshot of inventory state at the START of each month, before
        any sales are deducted. This mirrors the Intermediate-level
        inventory_before_monthly_sales.csv so both levels are consistent.

Aggregated SQLite tables saved here:
    advanced_transaction_details_5yr
    advanced_monthly_product_summary
    advanced_monthly_ledger_summary

Design rules:
- Reads from data/raw/inventory.csv (never modifies it)
- All calculations use the same formulas as Intermediate:
      revenue      = quantity_sold * unit_price
      expense      = quantity_sold * unit_cost
      gross_profit = revenue - expense
      remaining_stock = starting_stock - total_quantity_sold (per month)
- Each month uses inventory.starting_stock as the opening stock level
  because the Advanced generator resets stock monthly
- Returns all DataFrames for notebook inspection
"""

from pathlib import Path
from calendar import monthrange

import pandas as pd
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
# Per-month calculation functions
# ---------------------------------------------------------------------------

def build_transaction_details(
    transactions: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join monthly transactions with inventory and calculate financial fields.

    Matches the schema used by Intermediate's build_monthly_transaction_details().

    Parameters
    ----------
    transactions : pd.DataFrame
        Monthly transactions with columns:
        transaction_id, transaction_date, product_id, quantity_sold.
    inventory : pd.DataFrame
        Raw inventory DataFrame from load_inventory().

    Returns
    -------
    pd.DataFrame
        Transaction-level details with columns:
        transaction_id, transaction_date, product_id, product_name,
        category, quantity_sold, unit_cost, unit_price,
        revenue, expense, gross_profit, starting_stock,
        total_quantity_sold, remaining_stock
    """
    if transactions.empty:
        return pd.DataFrame(columns=[
            "transaction_id", "transaction_date", "product_id",
            "product_name", "category", "quantity_sold",
            "unit_cost", "unit_price", "revenue", "expense",
            "gross_profit", "starting_stock",
            "total_quantity_sold", "remaining_stock",
        ])

    details = transactions.merge(inventory, on="product_id", how="left")

    details["revenue"]      = details["quantity_sold"] * details["unit_price"]
    details["expense"]      = details["quantity_sold"] * details["unit_cost"]
    details["gross_profit"] = details["revenue"] - details["expense"]

    # Total sold per product this month (for remaining_stock)
    total_sold = (
        details.groupby("product_id")["quantity_sold"].transform("sum")
    )
    details["total_quantity_sold"] = total_sold.astype(int)
    details["remaining_stock"] = (
        details["starting_stock"] - details["total_quantity_sold"]
    ).clip(lower=0).astype(int)

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
    ).reset_index(drop=True)


def build_product_summary(
    transaction_details: pd.DataFrame,
    inventory: pd.DataFrame,
    year: int,
    month: int,
) -> pd.DataFrame:
    """
    Create product-level monthly sales and inventory summary.

    Matches the schema used by Intermediate's build_monthly_product_summary().

    Parameters
    ----------
    transaction_details : pd.DataFrame
        Output of build_transaction_details() for this month.
    inventory : pd.DataFrame
        Raw inventory DataFrame from load_inventory().
    year : int
        Year of the month being summarised.
    month : int
        Month number (1-12).

    Returns
    -------
    pd.DataFrame
        Product-level summary with columns:
        product_id, product_name, category, starting_stock,
        unit_cost, unit_price, total_quantity_sold, total_revenue,
        total_expense, gross_profit, gross_margin_rate,
        transaction_count, remaining_stock,
        average_daily_sales, sell_through_rate, year, month
    """
    days_in_month = monthrange(year, month)[1]

    if transaction_details.empty:
        sales_summary = pd.DataFrame(columns=[
            "product_id", "total_quantity_sold", "total_revenue",
            "total_expense", "gross_profit", "transaction_count",
        ])
    else:
        sales_summary = (
            transaction_details
            .groupby("product_id", as_index=False)
            .agg(
                total_quantity_sold=("quantity_sold",    "sum"),
                total_revenue      =("revenue",          "sum"),
                total_expense      =("expense",          "sum"),
                gross_profit       =("gross_profit",     "sum"),
                transaction_count  =("transaction_id",   "nunique"),
            )
        )

    product_summary = inventory.merge(sales_summary, on="product_id", how="left")

    fill_zero = [
        "total_quantity_sold", "total_revenue", "total_expense",
        "gross_profit", "transaction_count",
    ]
    for col in fill_zero:
        product_summary[col] = product_summary[col].fillna(0)

    product_summary["remaining_stock"] = (
        product_summary["starting_stock"] - product_summary["total_quantity_sold"]
    ).clip(lower=0).astype(int)

    product_summary["gross_margin_rate"] = product_summary.apply(
        lambda r: round(r["gross_profit"] / r["total_revenue"], 4)
        if r["total_revenue"] > 0 else 0.0,
        axis=1,
    )

    product_summary["average_daily_sales"] = (
        product_summary["total_quantity_sold"] / days_in_month
    ).round(2)

    product_summary["sell_through_rate"] = product_summary.apply(
        lambda r: round(r["total_quantity_sold"] / r["starting_stock"], 4)
        if r["starting_stock"] > 0 else 0.0,
        axis=1,
    )

    product_summary["year"]  = year
    product_summary["month"] = month

    output_columns = [
        "product_id", "product_name", "category",
        "starting_stock", "unit_cost", "unit_price",
        "total_quantity_sold", "total_revenue", "total_expense",
        "gross_profit", "gross_margin_rate",
        "transaction_count", "remaining_stock",
        "average_daily_sales", "sell_through_rate",
        "year", "month",
    ]

    return product_summary[output_columns].sort_values("product_id").reset_index(drop=True)


def build_ledger_summary(
    transaction_details: pd.DataFrame,
    year: int,
    month: int,
) -> pd.DataFrame:
    """
    Create a one-row monthly ledger summary.

    Schema is a *superset* of Intermediate's build_monthly_ledger_summary():
    all Intermediate columns are present, and the Advanced level additionally
    emits numeric ``year`` and ``month_num`` columns. Those two columns are
    required by build_advanced_dashboard_data() to sort 60 rows across five
    years; the Intermediate level only summarises one month at a time and
    therefore does not need them.

    Cross-level reconciliation that compares only the shared columns
    (``month``, ``total_revenue``, ``total_expense``, ``gross_profit``,
    ``gross_margin_rate``, ``total_quantity_sold``, ``transaction_count``,
    ``unique_products_sold``, ``month_start``, ``month_end``,
    ``days_in_month``) remains valid.

    Parameters
    ----------
    transaction_details : pd.DataFrame
        Output of build_transaction_details() for this month.
    year : int
        Year of the month being summarised.
    month : int
        Month number (1-12).

    Returns
    -------
    pd.DataFrame
        One-row ledger with columns:
        month (label "YYYY-MM"), year (int), month_num (int 1-12),
        month_start, month_end, days_in_month,
        transaction_count, unique_products_sold,
        total_quantity_sold, total_revenue, total_expense,
        gross_profit, gross_margin_rate
    """
    import calendar
    days_in_month = monthrange(year, month)[1]
    month_start   = f"{year}-{month:02d}-01"
    month_end     = f"{year}-{month:02d}-{days_in_month:02d}"
    month_label   = f"{year}-{month:02d}"

    if transaction_details.empty:
        total_revenue       = 0.0
        total_expense       = 0.0
        gross_profit        = 0.0
        total_quantity_sold = 0
        transaction_count   = 0
        unique_products     = 0
    else:
        total_revenue       = float(transaction_details["revenue"].sum())
        total_expense       = float(transaction_details["expense"].sum())
        gross_profit        = float(transaction_details["gross_profit"].sum())
        total_quantity_sold = int(transaction_details["quantity_sold"].sum())
        transaction_count   = int(transaction_details["transaction_id"].nunique())
        unique_products     = int(transaction_details["product_id"].nunique())

    gross_margin_rate = (
        round(gross_profit / total_revenue, 4) if total_revenue > 0 else 0.0
    )

    return pd.DataFrame([{
        "month":               month_label,
        "year":                year,
        "month_num":           month,
        "month_start":         month_start,
        "month_end":           month_end,
        "days_in_month":       days_in_month,
        "transaction_count":   transaction_count,
        "unique_products_sold": unique_products,
        "total_quantity_sold": total_quantity_sold,
        "total_revenue":       round(total_revenue, 2),
        "total_expense":       round(total_expense, 2),
        "gross_profit":        round(gross_profit, 2),
        "gross_margin_rate":   gross_margin_rate,
    }])


def build_inventory_before_sales(
    inventory: pd.DataFrame,
    year: int,
    month: int,
) -> pd.DataFrame:
    """
    Build the inventory snapshot at the START of a month, before any sales.

    This mirrors the Intermediate-level inventory_before_monthly_sales.csv.
    Because the Advanced generator resets stock to starting_stock each month,
    this is simply inventory with year/month metadata attached.

    Parameters
    ----------
    inventory : pd.DataFrame
        Raw inventory DataFrame from load_inventory().
    year : int
        Year of the month.
    month : int
        Month number (1-12).

    Returns
    -------
    pd.DataFrame
        Inventory snapshot with columns:
        product_id, product_name, category,
        starting_stock, unit_cost, unit_price, year, month
    """
    snapshot = inventory[[
        "product_id", "product_name", "category",
        "starting_stock", "unit_cost", "unit_price",
    ]].copy()
    snapshot["year"]  = year
    snapshot["month"] = month
    return snapshot.sort_values("product_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _monthly_folder(
    output_base_folder: str | Path,
    year: int,
    month: int,
) -> Path:
    """Return (and create) the year/month output folder."""
    folder = Path(output_base_folder) / f"year_{year}" / f"month_{month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_monthly_outputs(
    transaction_details: pd.DataFrame,
    product_summary: pd.DataFrame,
    ledger_summary: pd.DataFrame,
    inventory_before: pd.DataFrame,
    year: int,
    month: int,
    output_base_folder: str | Path,
) -> dict:
    """
    Save the four per-month output files to the year/month folder.

    Parameters
    ----------
    transaction_details : pd.DataFrame
        Output of build_transaction_details().
    product_summary : pd.DataFrame
        Output of build_product_summary().
    ledger_summary : pd.DataFrame
        Output of build_ledger_summary().
    inventory_before : pd.DataFrame
        Output of build_inventory_before_sales().
    year : int
        Year of the month.
    month : int
        Month number (1-12).
    output_base_folder : str or Path
        Root advanced output folder.

    Returns
    -------
    dict
        Mapping of file label to saved Path.
    """
    folder = _monthly_folder(output_base_folder, year, month)

    paths = {}

    p = folder / "transaction_details.csv"
    transaction_details.to_csv(p, index=False)
    paths["transaction_details"] = p

    p = folder / "product_summary.csv"
    product_summary.to_csv(p, index=False)
    paths["product_summary"] = p

    p = folder / "ledger_summary.csv"
    ledger_summary.to_csv(p, index=False)
    paths["ledger_summary"] = p

    p = folder / "inventory_before_monthly_sales.csv"
    inventory_before.to_csv(p, index=False)
    paths["inventory_before_monthly_sales"] = p

    return paths


def save_aggregated_outputs_to_sqlite(
    all_transaction_details: pd.DataFrame,
    all_product_summaries: pd.DataFrame,
    all_ledger_summaries: pd.DataFrame,
    sqlite_db_path: str | Path,
) -> None:
    """
    Save aggregated five-year outputs to SQLite.

    Table names:
        advanced_transaction_details_5yr
        advanced_monthly_product_summary
        advanced_monthly_ledger_summary

    Parameters
    ----------
    all_transaction_details : pd.DataFrame
        Combined transaction details for all 60 months.
    all_product_summaries : pd.DataFrame
        Combined product summaries for all 60 months.
    all_ledger_summaries : pd.DataFrame
        Combined ledger summaries for all 60 months.
    sqlite_db_path : str or Path
        Path to the SQLite database.
    """
    sqlite_db_path = Path(sqlite_db_path)
    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{sqlite_db_path}", connect_args={"timeout": 30})

    all_transaction_details.to_sql(
        name="advanced_transaction_details_5yr",
        con=engine, if_exists="replace", index=False,
    )
    all_product_summaries.to_sql(
        name="advanced_monthly_product_summary",
        con=engine, if_exists="replace", index=False,
    )
    all_ledger_summaries.to_sql(
        name="advanced_monthly_ledger_summary",
        con=engine, if_exists="replace", index=False,
    )

    engine.dispose()

    print(f"SQLite tables saved   : advanced_transaction_details_5yr")
    print(f"                        advanced_monthly_product_summary")
    print(f"                        advanced_monthly_ledger_summary")
    print(f"                        → {sqlite_db_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_monthly_outputs(
    inventory_csv_path: str | Path = "data/raw/inventory.csv",
    output_base_folder: str | Path = "data/processed/advanced",
    sqlite_db_path: str | Path = "src/database/sari_sari_store.db",
    all_transactions: pd.DataFrame = None,
    save_csv: bool = True,
    save_sqlite: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Generate and save the missing monthly output files for all 60 months.

    For each month computes transaction_details, product_summary,
    ledger_summary, and inventory_before_monthly_sales, then saves them
    to the year/month folder and accumulates aggregated SQLite tables.

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
        If None, the function reads monthly CSVs from the output folder.
    save_csv : bool
        Whether to save per-month CSV files.
    save_sqlite : bool
        Whether to save aggregated SQLite tables.
    verbose : bool
        Whether to print progress to the console.

    Returns
    -------
    dict
        Keys:
        - "all_transaction_details" : pd.DataFrame (all 60 months combined)
        - "all_product_summaries"   : pd.DataFrame (all 60 months combined)
        - "all_ledger_summaries"    : pd.DataFrame (all 60 months combined)
    """
    inventory = load_inventory(inventory_csv_path)

    if verbose:
        print("\n" + "=" * 72)
        print("ADVANCED GOAL: MONTHLY OUTPUTS (transaction details, product summary,")
        print("               ledger summary, inventory before sales)")
        print("=" * 72)
        print(f"Products        : {len(inventory)}")
        print(f"Period          : {START_YEAR}–{END_YEAR} (60 months)")
        print("-" * 72)

    txns = all_transactions.copy() if all_transactions is not None else None
    if txns is not None:
        txns["transaction_date"] = pd.to_datetime(txns["transaction_date"])

    detail_frames  = []
    summary_frames = []
    ledger_frames  = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):

            # --- Get transactions for this month ---
            if txns is not None:
                mask = (
                    (txns["transaction_date"].dt.year  == year) &
                    (txns["transaction_date"].dt.month == month)
                )
                monthly_txns = txns[mask].copy()
            else:
                # Fall back to reading the monthly CSV from disk
                csv_path = (
                    Path(output_base_folder)
                    / f"year_{year}" / f"month_{month:02d}"
                    / "transactions.csv"
                )
                if csv_path.exists():
                    monthly_txns = pd.read_csv(csv_path)
                    monthly_txns["transaction_date"] = pd.to_datetime(
                        monthly_txns["transaction_date"]
                    )
                else:
                    monthly_txns = pd.DataFrame(columns=[
                        "transaction_id", "transaction_date",
                        "product_id", "quantity_sold",
                    ])

            # --- Compute outputs ---
            transaction_details = build_transaction_details(
                monthly_txns, inventory
            )
            product_summary = build_product_summary(
                transaction_details, inventory, year, month
            )
            ledger_summary = build_ledger_summary(
                transaction_details, year, month
            )
            inventory_before = build_inventory_before_sales(
                inventory, year, month
            )

            # --- Save monthly CSVs ---
            if save_csv:
                save_monthly_outputs(
                    transaction_details=transaction_details,
                    product_summary=product_summary,
                    ledger_summary=ledger_summary,
                    inventory_before=inventory_before,
                    year=year,
                    month=month,
                    output_base_folder=output_base_folder,
                )

            detail_frames.append(transaction_details)
            summary_frames.append(product_summary)
            ledger_frames.append(ledger_summary)

            if verbose:
                rev = ledger_summary["total_revenue"].iloc[0]
                gp  = ledger_summary["gross_profit"].iloc[0]
                print(
                    f"  {year}-{month:02d}  |  "
                    f"revenue PHP {rev:>10,.2f}  |  "
                    f"profit PHP {gp:>9,.2f}"
                )

    all_transaction_details = pd.concat(detail_frames,  ignore_index=True)
    all_product_summaries   = pd.concat(summary_frames, ignore_index=True)
    all_ledger_summaries    = pd.concat(ledger_frames,  ignore_index=True)

    # Save aggregated CSVs to root advanced folder
    if save_csv:
        root = Path(output_base_folder)
        root.mkdir(parents=True, exist_ok=True)
        all_transaction_details.to_csv(
            root / "advanced_transaction_details_5yr.csv", index=False
        )
        all_product_summaries.to_csv(
            root / "advanced_monthly_product_summary.csv", index=False
        )
        all_ledger_summaries.to_csv(
            root / "advanced_monthly_ledger_summary.csv", index=False
        )

    if save_sqlite:
        save_aggregated_outputs_to_sqlite(
            all_transaction_details=all_transaction_details,
            all_product_summaries=all_product_summaries,
            all_ledger_summaries=all_ledger_summaries,
            sqlite_db_path=sqlite_db_path,
        )

    if verbose:
        total_rev = all_ledger_summaries["total_revenue"].sum()
        total_gp  = all_ledger_summaries["gross_profit"].sum()
        print("-" * 72)
        print(f"5-year total revenue : PHP {total_rev:,.2f}")
        print(f"5-year gross profit  : PHP {total_gp:,.2f}")
        print("Monthly outputs complete.")
        print("=" * 72)

    return {
        "all_transaction_details": all_transaction_details,
        "all_product_summaries":   all_product_summaries,
        "all_ledger_summaries":    all_ledger_summaries,
    }


if __name__ == "__main__":
    run_monthly_outputs()
