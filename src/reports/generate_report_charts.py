"""
generate_report_charts.py

One-shot artifact generator. Runs each level's pipeline and produces a tidy
set of line graphs and bar/pie charts under `reports/figures/`, suitable for
the final presentation deck.

Each chart is titled, has axis labels with PHP units where relevant, and is
saved at 150 DPI as PNG.

Run from project root:
    PYTHONPATH=$PWD python -m src.reports.generate_report_charts
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.advanced.advanced_runner import run_advanced_pipeline
from src.basic.sales_calculator import run_basic_sales_calculator
from src.intermediate.monthly_simulator import (
    run_intermediate_monthly_simulator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  saved {fig_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# BASIC artifacts
# ---------------------------------------------------------------------------

def generate_basic_charts() -> None:
    print("\n[BASIC]")
    out = _ensure_dir(FIGURES_DIR / "basic")
    # Chart-only run — don't overwrite the basic CSVs or shared SQLite DB.
    sales_details, product_summary, ledger_summary = run_basic_sales_calculator(
        save_csv_outputs=False,
        save_sqlite_database=False,
    )

    # 1. Revenue vs expense vs profit (single-day bar)
    led = ledger_summary.iloc[0]
    plt.figure(figsize=(7, 4.2))
    plt.bar(
        ["Revenue", "Expense", "Gross Profit"],
        [float(led["total_revenue"]), float(led["total_expense"]),
         float(led["gross_profit"])],
        color=["#2563EB", "#DC2626", "#16A34A"],
    )
    plt.title("Daily Ledger Summary (One Day)")
    plt.ylabel("PHP")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    _save(out / "01_daily_ledger_summary.png")

    # 2. Revenue by product (bar)
    ps = product_summary.sort_values("total_revenue", ascending=False)
    plt.figure(figsize=(10, 4.5))
    plt.bar(ps["product_name"], ps["total_revenue"], color="#2563EB")
    plt.title("Revenue by Product (One Day)")
    plt.xlabel("Product")
    plt.ylabel("Revenue (PHP)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    _save(out / "02_revenue_by_product.png")

    # 3. Remaining stock by product (line, sorted by stock)
    ps_stock = product_summary.sort_values("remaining_stock")
    plt.figure(figsize=(10, 4.5))
    plt.plot(
        ps_stock["product_name"], ps_stock["remaining_stock"],
        marker="o", color="#16A34A",
    )
    plt.title("Remaining Stock by Product (One Day)")
    plt.xlabel("Product")
    plt.ylabel("Units Remaining")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5)
    _save(out / "03_remaining_stock.png")


# ---------------------------------------------------------------------------
# INTERMEDIATE artifacts
# ---------------------------------------------------------------------------

def generate_intermediate_charts() -> None:
    print("\n[INTERMEDIATE]")
    out = _ensure_dir(FIGURES_DIR / "intermediate")
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
        save_outputs=False,
        create_charts=False,
    )
    details = outputs["monthly_transaction_details"].copy()
    ps = outputs["monthly_product_summary"]
    restock = outputs["restock_recommendations"]

    details["transaction_date"] = pd.to_datetime(details["transaction_date"])

    # 1. Daily revenue trend (line)
    daily = (
        details.groupby(details["transaction_date"].dt.date)["revenue"]
        .sum()
        .reset_index(name="revenue")
    )
    plt.figure(figsize=(12, 4))
    plt.plot(daily["transaction_date"], daily["revenue"],
             marker="o", color="#2563EB")
    plt.title("Daily Revenue Trend — January 2026")
    plt.xlabel("Date")
    plt.ylabel("Revenue (PHP)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5)
    _save(out / "01_daily_revenue_trend.png")

    # 2. Daily units sold trend (line)
    daily_units = (
        details.groupby(details["transaction_date"].dt.date)["quantity_sold"]
        .sum()
        .reset_index(name="units")
    )
    plt.figure(figsize=(12, 4))
    plt.plot(daily_units["transaction_date"], daily_units["units"],
             marker="o", color="#7C3AED")
    plt.title("Daily Units Sold — January 2026")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5)
    _save(out / "02_daily_units_sold.png")

    # 3. Revenue by category (bar)
    cat = (
        details.groupby("category", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )
    plt.figure(figsize=(8, 4.5))
    plt.bar(cat["category"], cat["revenue"], color="#0EA5E9")
    plt.title("Revenue by Category — January 2026")
    plt.xlabel("Category")
    plt.ylabel("Revenue (PHP)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    _save(out / "03_revenue_by_category.png")

    # 4. Top 10 products by units sold (horizontal bar)
    top = (
        ps.sort_values("total_quantity_sold", ascending=True).tail(10)
    )
    plt.figure(figsize=(9, 5))
    plt.barh(top["product_name"], top["total_quantity_sold"], color="#16A34A")
    plt.title("Top 10 Products by Units Sold — January 2026")
    plt.xlabel("Units Sold")
    plt.ylabel("Product")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    _save(out / "04_top_products_units.png")

    # 5. Restock plan (bar of recommended qty)
    rs = (
        restock[restock["recommend_restock"].astype(bool)]
        .sort_values("recommended_restock_quantity", ascending=True)
    )
    if not rs.empty:
        plt.figure(figsize=(9, 5))
        plt.barh(
            rs["product_name"], rs["recommended_restock_quantity"],
            color="#F59E0B",
        )
        plt.title("Restock Plan for Next Month — January 2026")
        plt.xlabel("Recommended Units to Restock")
        plt.ylabel("Product")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        _save(out / "05_restock_plan.png")


# ---------------------------------------------------------------------------
# ADVANCED artifacts
# ---------------------------------------------------------------------------

def generate_advanced_charts() -> None:
    print("\n[ADVANCED]")
    out = _ensure_dir(FIGURES_DIR / "advanced")
    res = run_advanced_pipeline(
        inventory_csv_path=PROJECT_ROOT / "data" / "raw" / "inventory.csv",
        output_base_folder=PROJECT_ROOT / "data" / "processed" / "advanced",
        sqlite_db_path=PROJECT_ROOT / "src" / "database" / "sari_sari_store.db",
        random_seed=42,
        save_csv=True,
        save_sqlite=True,
        verbose=False,
    )
    ledger = res["all_ledger_summaries"].copy()
    dashboard = res["dashboard"].copy()
    txns = res["all_transactions"].copy()
    events = res["sales_events"].copy()
    pricing = res["pricing_recommendations"].copy()
    restock = res["all_restock"].copy()
    feedback = res["all_feedback"].copy()

    # 1. Monthly revenue trend across 5 years (line)
    led = ledger.sort_values(["year", "month_num"])
    labels = [f"{int(r['year'])}-{int(r['month_num']):02d}"
              for _, r in led.iterrows()]
    plt.figure(figsize=(14, 4.5))
    plt.plot(range(len(led)), led["total_revenue"],
             color="#2563EB", linewidth=1.4)
    plt.title("Monthly Revenue — 5 Years (2022–2026)")
    plt.xlabel("Month")
    plt.ylabel("Revenue (PHP)")
    step = max(1, len(led) // 12)
    plt.xticks(range(0, len(led), step), labels[::step],
               rotation=35, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5)
    _save(out / "01_monthly_revenue_5yr.png")

    # 2. Monthly revenue with event-month markers
    plt.figure(figsize=(14, 4.5))
    plt.plot(range(len(led)), led["total_revenue"],
             color="#2563EB", linewidth=1.2, label="Monthly revenue")
    # Reuse the dashboard event_flag mapped onto the same monthly ordering
    trend = dashboard[
        (dashboard["metric_group"] == "monthly_revenue_trend")
        & (dashboard["metric_name"] == "monthly_revenue")
    ].copy().reset_index(drop=True)
    trend["value"] = pd.to_numeric(trend["value"])
    event_idx = trend.index[trend["event_flag"].astype(bool)].tolist()
    plt.scatter(
        event_idx, trend.loc[event_idx, "value"],
        color="#F97316", s=35, zorder=5, label="Sales event active",
    )
    plt.title("Monthly Revenue with Grocery-Wide Sales Events Flagged")
    plt.xlabel("Month")
    plt.ylabel("Revenue (PHP)")
    plt.xticks(range(0, len(led), step), labels[::step],
               rotation=35, ha="right")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    _save(out / "02_monthly_revenue_event_peaks.png")

    # 3. Annual profit (bar)
    annual = (
        ledger.groupby("year", as_index=False)
        .agg(profit=("gross_profit", "sum"),
             revenue=("total_revenue", "sum"))
        .sort_values("year")
    )
    plt.figure(figsize=(8, 4.5))
    plt.bar(annual["year"].astype(str), annual["profit"], color="#16A34A")
    plt.title("Annual Gross Profit — 2022 to 2026")
    plt.xlabel("Year")
    plt.ylabel("Gross Profit (PHP)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    _save(out / "03_annual_profit.png")

    # 4. Event-day vs non-event-day daily units (bar)
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])
    event_days: set = set()
    for _, ev in events.iterrows():
        for d in pd.date_range(ev["start_date"], ev["end_date"]):
            event_days.add(d.date())
    txns["transaction_date"] = pd.to_datetime(txns["transaction_date"])
    daily = txns.groupby("transaction_date")["quantity_sold"].sum()
    daily_df = daily.reset_index(name="units")
    daily_df["is_event"] = daily_df["transaction_date"].dt.date.isin(
        event_days
    )
    avg = daily_df.groupby("is_event")["units"].mean().reset_index()
    labels = ["Non-event day" if not v else "Event day" for v in avg["is_event"]]
    plt.figure(figsize=(6, 4.5))
    plt.bar(labels, avg["units"], color=["#94A3B8", "#F97316"])
    plt.title("Average Daily Units — Event vs Non-Event Days")
    plt.ylabel("Avg Units per Day")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    _save(out / "04_event_lift.png")

    # 5. Pricing recommendation: current vs recommended (grouped bar)
    pr = pricing.sort_values("product_id")
    x = range(len(pr))
    plt.figure(figsize=(13, 4.5))
    plt.bar([i - 0.2 for i in x], pr["current_unit_price"], width=0.4,
            color="#94A3B8", label="Current price")
    plt.bar([i + 0.2 for i in x], pr["recommended_price"], width=0.4,
            color="#2563EB", label="Recommended price")
    plt.title("Pricing Strategy — Current vs Recommended (PHP)")
    plt.xlabel("Product")
    plt.ylabel("Price (PHP)")
    plt.xticks(x, pr["product_name"], rotation=40, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    _save(out / "05_pricing_recommendation.png")

    # 6. Stockout risk distribution (bar)
    risk = restock.groupby("stockout_risk").size().reset_index(name="count")
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    risk["order"] = risk["stockout_risk"].apply(
        lambda r: order.index(r) if r in order else 99
    )
    risk = risk.sort_values("order")
    plt.figure(figsize=(7, 4.5))
    plt.bar(risk["stockout_risk"], risk["count"],
            color=["#DC2626", "#F97316", "#EAB308", "#16A34A"])
    plt.title("Stockout Risk Distribution Across 60 Months")
    plt.xlabel("Risk Level")
    plt.ylabel("Product-Month Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    _save(out / "06_stockout_risk.png")

    # 7. Customer feedback sentiment distribution (bar)
    sent = (
        feedback.groupby("sentiment").size().reset_index(name="count")
    )
    plt.figure(figsize=(6, 4.5))
    plt.bar(sent["sentiment"], sent["count"],
            color=["#16A34A", "#94A3B8", "#DC2626"])
    plt.title("Customer Feedback Sentiment (5 Years)")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    _save(out / "07_feedback_sentiment.png")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_dir(FIGURES_DIR)
    print(f"Writing chart artifacts under: "
          f"{FIGURES_DIR.relative_to(PROJECT_ROOT)}")
    generate_basic_charts()
    generate_intermediate_charts()
    generate_advanced_charts()
    print("\nAll report charts generated.")


if __name__ == "__main__":
    main()
