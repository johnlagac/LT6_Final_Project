"""Dashboard-ready analytics outputs for the Intermediate simulator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _append_metric(
    rows: list[dict[str, object]],
    metric_group: str,
    metric_name: str,
    dimension: str,
    value: object,
    rank: int | None = None,
) -> None:
    """Append one normalized dashboard row to an output list."""
    rows.append(
        {
            "metric_group": metric_group,
            "metric_name": metric_name,
            "dimension": dimension,
            "value": value,
            "rank": rank,
        }
    )


def build_monthly_dashboard_data(
    transaction_details: pd.DataFrame,
    product_summary: pd.DataFrame,
    ledger_summary: pd.DataFrame,
    restock_recommendations: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build a dashboard-ready summary table for monthly store analytics.

    Parameters
    ----------
    transaction_details : pandas.DataFrame
        Row-level transactions with revenue, expense, and gross profit.
    product_summary : pandas.DataFrame
        Product-level sales and inventory results.
    ledger_summary : pandas.DataFrame
        One-row monthly financial summary.
    restock_recommendations : pandas.DataFrame
        Product-level restock recommendation table.
    top_n : int, default=5
        Number of products to show for top-selling and low-stock sections.

    Returns
    -------
    pandas.DataFrame
        Normalized dashboard data with KPI, trend, category, product, and
        restock rows.
    """
    rows: list[dict[str, object]] = []

    if not ledger_summary.empty:
        ledger = ledger_summary.iloc[0]
        for metric in [
            "total_revenue",
            "total_expense",
            "gross_profit",
            "gross_margin_rate",
            "total_quantity_sold",
            "transaction_count",
            "unique_products_sold",
        ]:
            _append_metric(rows, "kpi", metric, "month_total", ledger.get(metric))

    if not transaction_details.empty:
        daily_revenue = (
            transaction_details.groupby("transaction_date", as_index=False)
            .agg(daily_revenue=("revenue", "sum"), daily_units=("quantity_sold", "sum"))
            .sort_values("transaction_date")
        )
        for _, row in daily_revenue.iterrows():
            _append_metric(
                rows,
                "daily_revenue_trend",
                "daily_revenue",
                str(row["transaction_date"]),
                round(float(row["daily_revenue"]), 2),
            )
            _append_metric(
                rows,
                "daily_unit_trend",
                "daily_units",
                str(row["transaction_date"]),
                int(row["daily_units"]),
            )

        category_sales = (
            transaction_details.groupby("category", as_index=False)
            .agg(category_revenue=("revenue", "sum"), category_units=("quantity_sold", "sum"))
            .sort_values("category_revenue", ascending=False)
        )
        for rank, (_, row) in enumerate(category_sales.iterrows(), start=1):
            _append_metric(
                rows,
                "sales_by_category",
                "category_revenue",
                str(row["category"]),
                round(float(row["category_revenue"]), 2),
                rank,
            )
            _append_metric(
                rows,
                "sales_by_category",
                "category_units",
                str(row["category"]),
                int(row["category_units"]),
                rank,
            )

    top_selling = product_summary.sort_values(
        "total_quantity_sold", ascending=False
    ).head(top_n)
    for rank, (_, row) in enumerate(top_selling.iterrows(), start=1):
        _append_metric(
            rows,
            "top_selling_products",
            "quantity_sold",
            str(row["product_name"]),
            int(row["total_quantity_sold"]),
            rank,
        )

    lowest_stock = product_summary.sort_values(
        ["remaining_stock", "total_quantity_sold"], ascending=[True, False]
    ).head(top_n)
    for rank, (_, row) in enumerate(lowest_stock.iterrows(), start=1):
        _append_metric(
            rows,
            "lowest_stock_products",
            "remaining_stock",
            str(row["product_name"]),
            int(row["remaining_stock"]),
            rank,
        )

    restock_needed = restock_recommendations[
        restock_recommendations["recommend_restock"].astype(bool)
    ].sort_values("recommended_restock_quantity", ascending=False)
    for rank, (_, row) in enumerate(restock_needed.iterrows(), start=1):
        _append_metric(
            rows,
            "restock_recommendations",
            "recommended_restock_quantity",
            str(row["product_name"]),
            int(row["recommended_restock_quantity"]),
            rank,
        )

    return pd.DataFrame(
        rows,
        columns=["metric_group", "metric_name", "dimension", "value", "rank"],
    )


def save_dashboard_charts(
    transaction_details: pd.DataFrame,
    product_summary: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 5,
) -> dict[str, Path]:
    """Save simple visualization files for notebook or report use.

    Parameters
    ----------
    transaction_details : pandas.DataFrame
        Row-level transactions with revenue and quantity sold.
    product_summary : pandas.DataFrame
        Product-level monthly sales and inventory summary.
    output_dir : str or pathlib.Path
        Folder where chart image files will be saved.
    top_n : int, default=5
        Number of products to include in product charts.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping of chart names to saved PNG file paths.
    """
    import matplotlib.pyplot as plt

    chart_dir = Path(output_dir) / "visualizations"
    chart_dir.mkdir(parents=True, exist_ok=True)
    saved_charts: dict[str, Path] = {}

    if not transaction_details.empty:
        daily_revenue = (
            transaction_details.groupby("transaction_date", as_index=False)["revenue"]
            .sum()
            .sort_values("transaction_date")
        )
        plt.figure(figsize=(10, 5))
        plt.plot(
            pd.to_datetime(daily_revenue["transaction_date"]),
            daily_revenue["revenue"],
            marker="o",
        )
        plt.title("Daily Revenue Trend")
        plt.xlabel("Transaction Date")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = chart_dir / "daily_revenue_trend.png"
        plt.savefig(path, dpi=150)
        plt.close()
        saved_charts["daily_revenue_trend"] = path

        category_revenue = (
            transaction_details.groupby("category", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
        )
        plt.figure(figsize=(9, 5))
        plt.bar(category_revenue["category"], category_revenue["revenue"])
        plt.title("Revenue by Category")
        plt.xlabel("Category")
        plt.ylabel("Revenue")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = chart_dir / "revenue_by_category.png"
        plt.savefig(path, dpi=150)
        plt.close()
        saved_charts["revenue_by_category"] = path

    if not product_summary.empty:
        top_products = product_summary.sort_values(
            "total_quantity_sold", ascending=False
        ).head(top_n)
        plt.figure(figsize=(9, 5))
        plt.bar(top_products["product_name"], top_products["total_quantity_sold"])
        plt.title("Top-Selling Products")
        plt.xlabel("Product")
        plt.ylabel("Units Sold")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = chart_dir / "top_selling_products.png"
        plt.savefig(path, dpi=150)
        plt.close()
        saved_charts["top_selling_products"] = path

        low_stock = product_summary.sort_values("remaining_stock").head(top_n)
        plt.figure(figsize=(9, 5))
        plt.bar(low_stock["product_name"], low_stock["remaining_stock"])
        plt.title("Lowest-Stock Products")
        plt.xlabel("Product")
        plt.ylabel("Remaining Stock")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = chart_dir / "lowest_stock_products.png"
        plt.savefig(path, dpi=150)
        plt.close()
        saved_charts["lowest_stock_products"] = path

    return saved_charts
