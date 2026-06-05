"""
advanced_dashboard.py

Advanced-level dashboard and analytics engine for the Sari-Sari Store Simulator.

Builds a normalized dashboard table from five years of monthly data. The
key addition over the Intermediate dashboard is event peak detection: months
where a grocery-wide sales event ran are flagged so the dashboard can
visually reflect the demand spikes that events cause.

The dashboard output follows the same normalized structure as
Intermediate's build_monthly_dashboard_data() so both levels can be
queried and compared in SQL Magic notebooks:

    metric_group | metric_name | dimension | value | rank

Metric groups produced:
    kpi_summary            - five-year totals and averages
    monthly_revenue_trend  - monthly revenue for all 60 months
    monthly_unit_trend     - monthly units sold for all 60 months
    event_peak_months      - months with active sales events (flagged)
    top_selling_products   - top N products by total five-year units sold
    lowest_stock_products  - products with lowest avg remaining stock
    sales_by_category      - five-year revenue and units by category
    annual_summary         - year-by-year revenue, expense, profit

Output files:
    data/processed/advanced/advanced_monthly_dashboard_data.csv
    SQLite table: advanced_monthly_dashboard_data
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

try:
    from src.advanced.five_year_generator import START_YEAR, END_YEAR
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    START_YEAR = 2022
    END_YEAR = 2026


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(
    rows: list,
    metric_group: str,
    metric_name: str,
    dimension: str,
    value: object,
    rank: int | None = None,
    event_flag: bool = False,
) -> None:
    """Append one normalized dashboard row."""
    rows.append({
        "metric_group": metric_group,
        "metric_name":  metric_name,
        "dimension":    dimension,
        "value":        value,
        "rank":         rank,
        "event_flag":   event_flag,
    })


def _has_event(
    sales_events: pd.DataFrame,
    year: int,
    month: int,
) -> bool:
    """
    Return True if any sales event was active during the given month.

    Parameters
    ----------
    sales_events : pd.DataFrame
        Full sales events table. Required columns: start_date, end_date.
    year : int
        Target year.
    month : int
        Target month (1-12).
    """
    if sales_events is None or sales_events.empty:
        return False

    import calendar
    last_day    = calendar.monthrange(year, month)[1]
    month_start = pd.Timestamp(f"{year}-{month:02d}-01")
    month_end   = pd.Timestamp(f"{year}-{month:02d}-{last_day:02d}")

    events = sales_events.copy()
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"]   = pd.to_datetime(events["end_date"])

    active = events[
        (events["start_date"] <= month_end) &
        (events["end_date"]   >= month_start)
    ]
    return not active.empty


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_advanced_dashboard_data(
    all_ledger_summaries: pd.DataFrame,
    all_product_summaries: pd.DataFrame,
    all_transaction_details: pd.DataFrame,
    sales_events: pd.DataFrame = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Build a normalized five-year dashboard table.

    Parameters
    ----------
    all_ledger_summaries : pd.DataFrame
        Combined monthly ledger summaries from run_monthly_outputs().
        Required columns: month (label "YYYY-MM"), year (int),
        month_num (int 1-12), total_revenue, total_expense, gross_profit,
        gross_margin_rate, total_quantity_sold, transaction_count,
        unique_products_sold.
    all_product_summaries : pd.DataFrame
        Combined monthly product summaries from run_monthly_outputs().
        Required columns: product_id, product_name, category,
        total_quantity_sold, total_revenue, remaining_stock, year, month.
    all_transaction_details : pd.DataFrame
        Combined transaction details from run_monthly_outputs().
        Required columns: transaction_date, product_id, category,
        quantity_sold, revenue.
    sales_events : pd.DataFrame, optional
        Full sales events DataFrame from run_sales_event_generator().
        Used to flag event-peak months in the trend data.
    top_n : int
        Number of products to include in top-selling and low-stock sections.

    Returns
    -------
    pd.DataFrame
        Normalized dashboard data with columns:
        metric_group, metric_name, dimension, value, rank, event_flag
    """
    rows = []

    # ------------------------------------------------------------------
    # 1. Five-year KPI summary
    # ------------------------------------------------------------------
    if not all_ledger_summaries.empty:
        total_revenue    = float(all_ledger_summaries["total_revenue"].sum())
        total_expense    = float(all_ledger_summaries["total_expense"].sum())
        total_gp         = float(all_ledger_summaries["gross_profit"].sum())
        total_units      = int(all_ledger_summaries["total_quantity_sold"].sum())
        total_txns       = int(all_ledger_summaries["transaction_count"].sum())
        avg_monthly_rev  = float(all_ledger_summaries["total_revenue"].mean())
        avg_monthly_gp   = float(all_ledger_summaries["gross_profit"].mean())
        overall_margin   = round(total_gp / total_revenue, 4) if total_revenue else 0.0

        for name, val in [
            ("total_revenue_5yr",      round(total_revenue, 2)),
            ("total_expense_5yr",      round(total_expense, 2)),
            ("total_gross_profit_5yr", round(total_gp, 2)),
            ("total_units_sold_5yr",   total_units),
            ("total_transactions_5yr", total_txns),
            ("avg_monthly_revenue",    round(avg_monthly_rev, 2)),
            ("avg_monthly_gross_profit", round(avg_monthly_gp, 2)),
            ("overall_gross_margin_rate", overall_margin),
        ]:
            _row(rows, "kpi_summary", name, "five_year", val)

    # ------------------------------------------------------------------
    # 2. Monthly revenue and unit trends (60 data points each)
    #    event_flag=True on months where a sales event was active
    # ------------------------------------------------------------------
    if not all_ledger_summaries.empty:
        ledger_sorted = all_ledger_summaries.sort_values(
            ["year", "month_num"]
        ).reset_index(drop=True)

        for _, lr in ledger_sorted.iterrows():
            year  = int(lr["year"])
            month = int(lr["month_num"])
            label = f"{year}-{month:02d}"
            flag  = _has_event(sales_events, year, month)

            _row(
                rows, "monthly_revenue_trend", "monthly_revenue",
                label, round(float(lr["total_revenue"]), 2),
                event_flag=flag,
            )
            _row(
                rows, "monthly_unit_trend", "monthly_units_sold",
                label, int(lr["total_quantity_sold"]),
                event_flag=flag,
            )
            _row(
                rows, "monthly_gross_profit_trend", "monthly_gross_profit",
                label, round(float(lr["gross_profit"]), 2),
                event_flag=flag,
            )

    # ------------------------------------------------------------------
    # 3. Event peak months — summary of which months had events
    # ------------------------------------------------------------------
    if sales_events is not None and not sales_events.empty:
        events = sales_events.copy()
        events["start_date"] = pd.to_datetime(events["start_date"])
        events["year"]  = events["start_date"].dt.year
        events["month"] = events["start_date"].dt.month

        for _, ev in events.iterrows():
            label = f"{int(ev['year'])}-{int(ev['month']):02d}"
            _row(
                rows, "event_peak_months", ev["event_name"],
                label, round(float(ev["discount_rate"]) * 100, 1),
                event_flag=True,
            )

    # ------------------------------------------------------------------
    # 4. Top-selling products (five-year total units)
    # ------------------------------------------------------------------
    if not all_product_summaries.empty:
        product_totals = (
            all_product_summaries
            .groupby(["product_id", "product_name"], as_index=False)
            .agg(
                total_units  =("total_quantity_sold", "sum"),
                total_revenue=("total_revenue",       "sum"),
            )
            .sort_values("total_units", ascending=False)
        )
        for rank, (_, pr) in enumerate(
            product_totals.head(top_n).iterrows(), start=1
        ):
            _row(
                rows, "top_selling_products", "total_units_sold_5yr",
                str(pr["product_name"]), int(pr["total_units"]), rank,
            )
            _row(
                rows, "top_selling_products", "total_revenue_5yr",
                str(pr["product_name"]),
                round(float(pr["total_revenue"]), 2), rank,
            )

    # ------------------------------------------------------------------
    # 5. Lowest average remaining stock (most at-risk products)
    # ------------------------------------------------------------------
    if not all_product_summaries.empty:
        avg_stock = (
            all_product_summaries
            .groupby(["product_id", "product_name"], as_index=False)
            ["remaining_stock"]
            .mean()
            .sort_values("remaining_stock")
        )
        for rank, (_, ps) in enumerate(
            avg_stock.head(top_n).iterrows(), start=1
        ):
            _row(
                rows, "lowest_stock_products", "avg_remaining_stock",
                str(ps["product_name"]),
                round(float(ps["remaining_stock"]), 1), rank,
            )

    # ------------------------------------------------------------------
    # 6. Sales by category (five-year revenue and units)
    # ------------------------------------------------------------------
    if not all_transaction_details.empty:
        cat_sales = (
            all_transaction_details
            .groupby("category", as_index=False)
            .agg(
                category_revenue=("revenue",       "sum"),
                category_units  =("quantity_sold", "sum"),
            )
            .sort_values("category_revenue", ascending=False)
        )
        for rank, (_, cs) in enumerate(cat_sales.iterrows(), start=1):
            _row(
                rows, "sales_by_category", "category_revenue_5yr",
                str(cs["category"]),
                round(float(cs["category_revenue"]), 2), rank,
            )
            _row(
                rows, "sales_by_category", "category_units_5yr",
                str(cs["category"]), int(cs["category_units"]), rank,
            )

    # ------------------------------------------------------------------
    # 7. Annual summary (year-by-year financials)
    # ------------------------------------------------------------------
    if not all_ledger_summaries.empty:
        annual = (
            all_ledger_summaries
            .groupby("year", as_index=False)
            .agg(
                annual_revenue=("total_revenue",       "sum"),
                annual_expense=("total_expense",       "sum"),
                annual_profit =("gross_profit",        "sum"),
                annual_units  =("total_quantity_sold", "sum"),
            )
            .sort_values("year")
        )
        for _, yr in annual.iterrows():
            year_label = str(int(yr["year"]))
            _row(rows, "annual_summary", "annual_revenue",
                 year_label, round(float(yr["annual_revenue"]), 2))
            _row(rows, "annual_summary", "annual_expense",
                 year_label, round(float(yr["annual_expense"]), 2))
            _row(rows, "annual_summary", "annual_gross_profit",
                 year_label, round(float(yr["annual_profit"]), 2))
            _row(rows, "annual_summary", "annual_units_sold",
                 year_label, int(yr["annual_units"]))

    return pd.DataFrame(
        rows,
        columns=[
            "metric_group", "metric_name", "dimension",
            "value", "rank", "event_flag",
        ],
    )


def print_dashboard_summary(dashboard: pd.DataFrame) -> None:
    """
    Print a high-level summary of the advanced dashboard to the console.

    Parameters
    ----------
    dashboard : pd.DataFrame
        Output of build_advanced_dashboard_data().
    """
    print("\n" + "=" * 72)
    print("ADVANCED DASHBOARD SUMMARY")
    print("=" * 72)

    kpis = dashboard[dashboard["metric_group"] == "kpi_summary"]
    for _, row in kpis.iterrows():
        val = row["value"]
        if isinstance(val, float):
            print(f"  {row['metric_name']:<35} : {val:>14,.2f}")
        else:
            print(f"  {row['metric_name']:<35} : {val:>14,}")

    event_months = dashboard[
        (dashboard["metric_group"] == "monthly_revenue_trend")
        & dashboard["event_flag"].astype(bool)
    ]
    print(f"\n  Event-peak months flagged : {len(event_months)}")

    print("\n  Top-selling products (five-year units):")
    top = dashboard[
        (dashboard["metric_group"] == "top_selling_products") &
        (dashboard["metric_name"]  == "total_units_sold_5yr")
    ].sort_values("rank")
    for _, r in top.iterrows():
        print(f"    {int(r['rank'])}. {r['dimension']:<30} {int(r['value']):>8,} units")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_dashboard_csv(
    dashboard: pd.DataFrame,
    output_base_folder: str | Path,
) -> Path:
    """
    Save the advanced dashboard data to the advanced root folder.

    Parameters
    ----------
    dashboard : pd.DataFrame
        Output of build_advanced_dashboard_data().
    output_base_folder : str or Path
        Root advanced output folder.

    Returns
    -------
    Path
        Path of the saved CSV.
    """
    folder = Path(output_base_folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / "advanced_monthly_dashboard_data.csv"
    dashboard.to_csv(path, index=False)
    return path


def save_dashboard_to_sqlite(
    dashboard: pd.DataFrame,
    sqlite_db_path: str | Path,
) -> None:
    """
    Save the advanced dashboard DataFrame to SQLite.

    Table name: advanced_monthly_dashboard_data

    Parameters
    ----------
    dashboard : pd.DataFrame
        Output of build_advanced_dashboard_data().
    sqlite_db_path : str or Path
        Path to the SQLite database.
    """
    sqlite_db_path = Path(sqlite_db_path)
    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{sqlite_db_path}", connect_args={"timeout": 30})

    dashboard.to_sql(
        name="advanced_monthly_dashboard_data",
        con=engine, if_exists="replace", index=False,
    )
    engine.dispose()

    print(
        f"SQLite table saved    : "
        f"advanced_monthly_dashboard_data → {sqlite_db_path}"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_advanced_dashboard(
    output_base_folder: str | Path = "data/processed/advanced",
    sqlite_db_path: str | Path = "src/database/sari_sari_store.db",
    all_ledger_summaries: pd.DataFrame = None,
    all_product_summaries: pd.DataFrame = None,
    all_transaction_details: pd.DataFrame = None,
    sales_events: pd.DataFrame = None,
    top_n: int = 5,
    save_csv: bool = True,
    save_sqlite: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the advanced dashboard generation workflow.

    Builds a normalized five-year dashboard table including monthly trend
    data, event peak flags, top products, category breakdowns, and annual
    summaries. Saves to CSV and SQLite.

    Parameters
    ----------
    output_base_folder : str or Path
        Root advanced output folder.
    sqlite_db_path : str or Path
        Path to the SQLite database.
    all_ledger_summaries : pd.DataFrame
        Monthly ledger summaries from run_monthly_outputs().
    all_product_summaries : pd.DataFrame
        Monthly product summaries from run_monthly_outputs().
    all_transaction_details : pd.DataFrame
        Five-year transaction details from run_monthly_outputs().
    sales_events : pd.DataFrame, optional
        Sales events from run_sales_event_generator(). Used to flag
        event-peak months in the trend data.
    top_n : int
        Number of products in top-selling and low-stock sections.
    save_csv : bool
        Whether to save the dashboard CSV.
    save_sqlite : bool
        Whether to save to SQLite.
    verbose : bool
        Whether to print a summary to the console.

    Returns
    -------
    pd.DataFrame
        Normalized dashboard DataFrame.
    """
    if verbose:
        print("\n" + "=" * 72)
        print("ADVANCED GOAL: DASHBOARD AND ANALYTICS")
        print("=" * 72)

    # Fall back to reading aggregated CSVs if DataFrames were not passed
    root = Path(output_base_folder)

    if all_ledger_summaries is None:
        p = root / "advanced_monthly_ledger_summary.csv"
        if p.exists():
            all_ledger_summaries = pd.read_csv(p)
            if verbose:
                print(f"Loaded ledger summaries from: {p}")
        else:
            raise FileNotFoundError(
                f"{p} not found. Run run_monthly_outputs() first."
            )

    if all_product_summaries is None:
        p = root / "advanced_monthly_product_summary.csv"
        if p.exists():
            all_product_summaries = pd.read_csv(p)
        else:
            raise FileNotFoundError(
                f"{p} not found. Run run_monthly_outputs() first."
            )

    if all_transaction_details is None:
        p = root / "advanced_transaction_details_5yr.csv"
        if p.exists():
            all_transaction_details = pd.read_csv(p)
        else:
            raise FileNotFoundError(
                f"{p} not found. Run run_monthly_outputs() first."
            )

    dashboard = build_advanced_dashboard_data(
        all_ledger_summaries=all_ledger_summaries,
        all_product_summaries=all_product_summaries,
        all_transaction_details=all_transaction_details,
        sales_events=sales_events,
        top_n=top_n,
    )

    if verbose:
        print_dashboard_summary(dashboard)

    if save_csv:
        path = save_dashboard_csv(dashboard, output_base_folder)
        print(f"\nCSV saved             : {path}")

    if save_sqlite:
        save_dashboard_to_sqlite(dashboard, sqlite_db_path)

    if verbose:
        print(f"\nDashboard rows total  : {len(dashboard):,}")
        event_peaks = dashboard["event_flag"].astype(bool).sum()
        print(f"Event-flagged rows    : {event_peaks:,}")
        print("Dashboard generation complete.")
        print("=" * 72)

    return dashboard


if __name__ == "__main__":
    run_advanced_dashboard()
