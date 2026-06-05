"""
pricing_strategy.py

Advanced-level pricing strategy engine for the Sari-Sari Store Simulator.

Analyses five years of sales history to recommend selling prices for each
product. Recommendations are grounded in actual product margins from
inventory.csv (margins range from 16% for Rice to 42.9% for Shampoo Sachet).

Pricing logic (in priority order):
1. If demand trend is UP and gross margin is LOW → slight price increase
2. If demand trend is DOWN and remaining stock is HIGH → slight price decrease
3. If gross margin is already HIGH → hold price, protect margin
4. Otherwise → adjust by demand trend multiplier

Demand trend is derived by comparing the most recent 6-month average demand
to the prior 6-month average. Threshold: >5% increase = "increasing",
<-5% = "decreasing", otherwise "stable".

Output:
    data/processed/advanced/advanced_pricing_recommendations.csv
    SQLite table: advanced_pricing_recommendations

Suggested output columns:
    product_id, product_name, category,
    current_unit_cost, current_unit_price,
    average_monthly_quantity_sold,
    gross_margin, demand_trend,
    recommended_price, pricing_reason
"""

from pathlib import Path
import pandas as pd
import numpy as np

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
# Pricing constants
# ---------------------------------------------------------------------------

# Gross margin thresholds
LOW_MARGIN_THRESHOLD  = 0.25   # below 25% → margin is low
HIGH_MARGIN_THRESHOLD = 0.35   # above 35% → margin is high

# Demand trend thresholds (fractional change in avg monthly units)
DEMAND_UP_THRESHOLD   =  0.05  # >5% increase → "increasing"
DEMAND_DOWN_THRESHOLD = -0.05  # <-5% decrease → "decreasing"

# Price adjustment fractions
PRICE_INCREASE_RATE   = 0.05   # +5% when demand up and margin low
PRICE_DECREASE_RATE   = 0.05   # -5% when demand down and stock high
TREND_ADJUSTMENT_RATE = 0.03   # ±3% for general trend adjustment

# Minimum price floor: never recommend below cost + 10% markup
MIN_MARKUP_FRACTION = 0.10

# Number of months to use for "recent" vs "prior" trend comparison
TREND_RECENT_MONTHS = 6
TREND_PRIOR_MONTHS  = 6

# Stock level threshold for "high stock" condition (fraction of avg monthly)
HIGH_STOCK_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_gross_margin(unit_cost: float, unit_price: float) -> float:
    """
    Compute gross margin as a fraction of selling price.

    gross_margin = (unit_price - unit_cost) / unit_price

    Parameters
    ----------
    unit_cost : float
        Cost per unit (PHP).
    unit_price : float
        Selling price per unit (PHP).

    Returns
    -------
    float
        Gross margin fraction (e.g. 0.375 for 37.5%).
    """
    if unit_price <= 0:
        return 0.0
    return (unit_price - unit_cost) / unit_price


def _compute_demand_trend(
    monthly_units: pd.Series,
    recent_months: int = TREND_RECENT_MONTHS,
    prior_months: int  = TREND_PRIOR_MONTHS,
) -> tuple[str, float]:
    """
    Classify demand trend by comparing recent vs prior period averages.

    Parameters
    ----------
    monthly_units : pd.Series
        Time-ordered Series of monthly units sold (oldest first).
    recent_months : int
        Number of most recent months to use as the "recent" window.
    prior_months : int
        Number of months before the recent window to use as "prior".

    Returns
    -------
    tuple[str, float]
        (trend_label, pct_change) where trend_label is one of:
        "increasing", "stable", "decreasing"
        and pct_change is the fractional change (e.g. 0.08 = +8%).
    """
    n = len(monthly_units)

    if n < 2:
        return "stable", 0.0

    recent_window = monthly_units.iloc[-recent_months:]
    prior_end     = max(0, n - recent_months)
    prior_start   = max(0, prior_end - prior_months)
    prior_window  = monthly_units.iloc[prior_start:prior_end]

    recent_avg = recent_window.mean()
    prior_avg  = prior_window.mean()

    if prior_avg <= 0:
        return "stable", 0.0

    pct_change = (recent_avg - prior_avg) / prior_avg

    if pct_change > DEMAND_UP_THRESHOLD:
        return "increasing", round(pct_change, 4)
    elif pct_change < DEMAND_DOWN_THRESHOLD:
        return "decreasing", round(pct_change, 4)
    else:
        return "stable", round(pct_change, 4)


def _recommend_price(
    unit_cost: float,
    unit_price: float,
    gross_margin: float,
    demand_trend: str,
    current_stock: float,
    avg_monthly_demand: float,
) -> tuple[float, str]:
    """
    Compute a recommended price and reason string for one product.

    Priority logic:
    1. Demand UP + margin LOW  → increase price to improve margin
    2. Demand DOWN + stock HIGH → decrease price to move inventory
    3. Margin HIGH              → hold price, protect margin
    4. Default                 → nudge price by demand trend

    Recommended price is always floored at cost + MIN_MARKUP_FRACTION.

    Parameters
    ----------
    unit_cost : float
        Cost per unit.
    unit_price : float
        Current selling price per unit.
    gross_margin : float
        Current gross margin fraction.
    demand_trend : str
        One of: "increasing", "stable", "decreasing".
    current_stock : float
        Current remaining stock.
    avg_monthly_demand : float
        Average monthly units sold.

    Returns
    -------
    tuple[float, str]
        (recommended_price, pricing_reason)
    """
    min_price = round(unit_cost * (1 + MIN_MARKUP_FRACTION), 2)

    stock_is_high = (
        avg_monthly_demand > 0
        and current_stock >= avg_monthly_demand * HIGH_STOCK_THRESHOLD
    )

    if demand_trend == "increasing" and gross_margin < LOW_MARGIN_THRESHOLD:
        new_price = unit_price * (1 + PRICE_INCREASE_RATE)
        reason = (
            f"Demand increasing and gross margin is low "
            f"({gross_margin:.1%}). "
            f"Raising price by {PRICE_INCREASE_RATE:.0%} to improve margin."
        )

    elif demand_trend == "decreasing" and stock_is_high:
        new_price = unit_price * (1 - PRICE_DECREASE_RATE)
        reason = (
            f"Demand decreasing and stock is high "
            f"({int(current_stock)} units). "
            f"Lowering price by {PRICE_DECREASE_RATE:.0%} to move inventory."
        )

    elif gross_margin >= HIGH_MARGIN_THRESHOLD:
        new_price = unit_price
        reason = (
            f"Gross margin is healthy ({gross_margin:.1%}). "
            f"Holding current price."
        )

    elif demand_trend == "increasing":
        new_price = unit_price * (1 + TREND_ADJUSTMENT_RATE)
        reason = (
            f"Demand is increasing. "
            f"Slight price increase of {TREND_ADJUSTMENT_RATE:.0%}."
        )

    elif demand_trend == "decreasing":
        new_price = unit_price * (1 - TREND_ADJUSTMENT_RATE)
        reason = (
            f"Demand is declining. "
            f"Slight price decrease of {TREND_ADJUSTMENT_RATE:.0%} to stay competitive."
        )

    else:
        new_price = unit_price
        reason = "Demand stable and margin acceptable. Holding current price."

    recommended_price = round(max(new_price, min_price), 2)

    return recommended_price, reason


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def build_pricing_recommendations(
    inventory: pd.DataFrame,
    all_transactions: pd.DataFrame,
    all_product_summaries: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build a pricing recommendations table from five years of transaction data.

    For each product, computes:
    - Average monthly quantity sold across all 60 months
    - Demand trend (recent 6 months vs prior 6 months)
    - Gross margin from current cost and price
    - Recommended selling price with reasoning

    Parameters
    ----------
    inventory : pd.DataFrame
        Validated inventory DataFrame from load_inventory().
    all_transactions : pd.DataFrame
        Aggregated five-year transactions from run_five_year_generator().
        Required columns: transaction_date, product_id, quantity_sold.
    all_product_summaries : pd.DataFrame, optional
        Monthly product summaries with year, month, product_id,
        total_quantity_sold, remaining_stock. If provided, used to get
        the most recent month's remaining_stock. Otherwise defaults to
        inventory starting_stock.

    Returns
    -------
    pd.DataFrame
        Pricing recommendations with columns:
        product_id, product_name, category,
        current_unit_cost, current_unit_price,
        average_monthly_quantity_sold,
        gross_margin, demand_trend,
        recommended_price, pricing_reason
    """
    txns = all_transactions.copy()
    txns["transaction_date"] = pd.to_datetime(txns["transaction_date"])
    txns["year"]  = txns["transaction_date"].dt.year
    txns["month"] = txns["transaction_date"].dt.month

    # Monthly units per product
    monthly_by_product = (
        txns.groupby(["product_id", "year", "month"])["quantity_sold"]
        .sum()
        .reset_index(name="monthly_units")
    )

    rows = []

    for _, product in inventory.iterrows():
        product_id   = product["product_id"]
        product_name = product["product_name"]
        category     = product["category"]
        unit_cost    = product["unit_cost"]
        unit_price   = product["unit_price"]

        prod_history = monthly_by_product[
            monthly_by_product["product_id"] == product_id
        ].sort_values(["year", "month"])

        monthly_units_series = prod_history["monthly_units"]

        if monthly_units_series.empty:
            avg_monthly = 0.0
            demand_trend = "stable"
        else:
            avg_monthly = monthly_units_series.mean()
            demand_trend, _ = _compute_demand_trend(monthly_units_series)

        gross_margin = _compute_gross_margin(unit_cost, unit_price)

        # Get most recent stock level
        if all_product_summaries is not None and not all_product_summaries.empty:
            prod_summaries = all_product_summaries[
                all_product_summaries["product_id"] == product_id
            ]
            if not prod_summaries.empty:
                latest = prod_summaries.sort_values(
                    ["year", "month"]
                ).iloc[-1]
                current_stock = latest.get("remaining_stock", product["starting_stock"])
            else:
                current_stock = product["starting_stock"]
        else:
            current_stock = product["starting_stock"]

        recommended_price, pricing_reason = _recommend_price(
            unit_cost=unit_cost,
            unit_price=unit_price,
            gross_margin=gross_margin,
            demand_trend=demand_trend,
            current_stock=current_stock,
            avg_monthly_demand=avg_monthly,
        )

        rows.append({
            "product_id":                   product_id,
            "product_name":                 product_name,
            "category":                     category,
            "current_unit_cost":            unit_cost,
            "current_unit_price":           unit_price,
            "average_monthly_quantity_sold": round(avg_monthly, 2),
            "gross_margin":                 round(gross_margin, 4),
            "demand_trend":                 demand_trend,
            "recommended_price":            recommended_price,
            "pricing_reason":               pricing_reason,
        })

    recommendations = pd.DataFrame(rows, columns=[
        "product_id",
        "product_name",
        "category",
        "current_unit_cost",
        "current_unit_price",
        "average_monthly_quantity_sold",
        "gross_margin",
        "demand_trend",
        "recommended_price",
        "pricing_reason",
    ])

    return recommendations.sort_values("product_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_pricing_recommendations_csv(
    recommendations: pd.DataFrame,
    output_base_folder: str | Path,
) -> Path:
    """
    Save pricing recommendations to the advanced root folder.

    Parameters
    ----------
    recommendations : pd.DataFrame
        Pricing recommendations DataFrame.
    output_base_folder : str or Path
        Root advanced output folder.

    Returns
    -------
    Path
        Path of the saved CSV.
    """
    folder = Path(output_base_folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / "advanced_pricing_recommendations.csv"
    recommendations.to_csv(path, index=False)
    return path


def save_pricing_recommendations_to_sqlite(
    recommendations: pd.DataFrame,
    sqlite_db_path: str | Path,
) -> None:
    """
    Save pricing recommendations to SQLite.

    Table name: advanced_pricing_recommendations

    Parameters
    ----------
    recommendations : pd.DataFrame
        Pricing recommendations DataFrame.
    sqlite_db_path : str or Path
        Path to the SQLite database.
    """
    sqlite_db_path = Path(sqlite_db_path)
    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{sqlite_db_path}")

    recommendations.to_sql(
        name="advanced_pricing_recommendations",
        con=engine,
        if_exists="replace",
        index=False,
    )

    print(
        f"SQLite table saved    : "
        f"advanced_pricing_recommendations → {sqlite_db_path}"
    )


def print_pricing_report(recommendations: pd.DataFrame) -> None:
    """
    Print a summary of pricing recommendations to the console.

    Parameters
    ----------
    recommendations : pd.DataFrame
        Pricing recommendations DataFrame.
    """
    print("\n" + "=" * 72)
    print("PRICING RECOMMENDATIONS SUMMARY")
    print("=" * 72)

    increases = (
        recommendations["recommended_price"]
        > recommendations["current_unit_price"]
    ).sum()
    decreases = (
        recommendations["recommended_price"]
        < recommendations["current_unit_price"]
    ).sum()
    holds = (
        recommendations["recommended_price"]
        == recommendations["current_unit_price"]
    ).sum()

    print(f"Price increases recommended : {increases}")
    print(f"Price decreases recommended : {decreases}")
    print(f"Prices held (no change)     : {holds}")

    print("\nDetailed recommendations:")
    print("-" * 72)

    display = recommendations[[
        "product_id",
        "product_name",
        "current_unit_price",
        "recommended_price",
        "gross_margin",
        "demand_trend",
    ]].copy()

    display["gross_margin"] = display["gross_margin"].map("{:.1%}".format)
    display["price_change"] = (
        recommendations["recommended_price"]
        - recommendations["current_unit_price"]
    ).map(lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}")

    print(display.to_string(index=False))
    print("=" * 72)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pricing_strategy(
    inventory_csv_path: str | Path = "data/raw/inventory.csv",
    output_base_folder: str | Path = "data/processed/advanced",
    sqlite_db_path: str | Path = "src/database/sari_sari_store.db",
    all_transactions: pd.DataFrame = None,
    all_product_summaries: pd.DataFrame = None,
    save_csv: bool = True,
    save_sqlite: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full pricing strategy workflow.

    Loads five-year transaction data, computes demand trends and margins
    for each product, and generates a pricing recommendations table.

    Parameters
    ----------
    inventory_csv_path : str or Path
        Path to data/raw/inventory.csv.
    output_base_folder : str or Path
        Root advanced output folder.
    sqlite_db_path : str or Path
        Path to the SQLite database.
    all_transactions : pd.DataFrame, optional
        Aggregated five-year transactions. If None, the function will
        attempt to read from the aggregated CSV in output_base_folder.
    all_product_summaries : pd.DataFrame, optional
        Aggregated monthly product summaries (for current stock lookup).
    save_csv : bool
        Whether to save the recommendations CSV.
    save_sqlite : bool
        Whether to save to SQLite.
    verbose : bool
        Whether to print a report to the console.

    Returns
    -------
    pd.DataFrame
        Pricing recommendations DataFrame.
    """
    inventory = load_inventory(inventory_csv_path)

    # Fall back to reading aggregated CSV if no DataFrame was passed in
    if all_transactions is None:
        fallback_csv = (
            Path(output_base_folder) / "advanced_transactions_5yr.csv"
        )
        if fallback_csv.exists():
            all_transactions = pd.read_csv(fallback_csv)
            if verbose:
                print(
                    f"\nLoaded transactions from: {fallback_csv} "
                    f"({len(all_transactions):,} rows)"
                )
        else:
            raise FileNotFoundError(
                "No transactions DataFrame provided and "
                f"{fallback_csv} does not exist. "
                "Run run_five_year_generator() first."
            )

    if verbose:
        print("\n" + "=" * 72)
        print("ADVANCED GOAL: PRICING STRATEGY ENGINE")
        print("=" * 72)
        print(f"Products loaded        : {len(inventory)}")
        print(f"Transaction records    : {len(all_transactions):,}")
        print("-" * 72)

    recommendations = build_pricing_recommendations(
        inventory=inventory,
        all_transactions=all_transactions,
        all_product_summaries=all_product_summaries,
    )

    if verbose:
        print_pricing_report(recommendations)

    if save_csv:
        path = save_pricing_recommendations_csv(
            recommendations, output_base_folder
        )
        print(f"\nCSV saved             : {path}")

    if save_sqlite:
        save_pricing_recommendations_to_sqlite(
            recommendations, sqlite_db_path
        )

    return recommendations


if __name__ == "__main__":
    run_pricing_strategy()
