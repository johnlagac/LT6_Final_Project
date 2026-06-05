"""
test_intermediate.py

Pytest anchor for the Intermediate-level "How to Test" requirements from the
project PDF:

    * Comparison of revenues generated to real data
    * Accuracy of information reflected in Dashboard and Inventory

Complements `test_intermediate_sanity_checks.py` (the script-style end-to-end
check) with focused unit tests on the building blocks.

Run from project root:
    PYTHONPATH=$PWD python -m pytest test/intermediate/test_intermediate.py -v
"""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.intermediate.monthly_simulator import (
    run_intermediate_monthly_simulator,
)
from src.intermediate.data_generator import (
    DEFAULT_BENCHMARK_INFO,
    generate_monthly_transactions,
    normalize_benchmark_info,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_INVENTORY_PATH = PROJECT_ROOT / "data" / "raw" / "inventory.csv"


def _load_inventory() -> pd.DataFrame:
    df = pd.read_csv(RAW_INVENTORY_PATH)
    df["starting_stock"] = pd.to_numeric(df["starting_stock"])
    df["unit_cost"] = pd.to_numeric(df["unit_cost"])
    df["unit_price"] = pd.to_numeric(df["unit_price"])
    return df


# ===========================================================================
# Synthetic revenue vs benchmark (PDF: "Comparison of revenues generated to
# real data")
# ===========================================================================

class TestSyntheticRevenueVsBenchmark(unittest.TestCase):
    """Sari-sari store PSA benchmark figures the team agreed on (chat
    2026-05-15, file `00000877-sari_sari_PSA_benchmark_data.md`):
    average store does ~PHP 7k-15k revenue per month; with the multipliers
    used in our generator the expected window is roughly PHP 8k-25k. We
    assert the generated month falls inside a wider sanity band so the
    test is stable across random seeds."""

    EXPECTED_REVENUE_MIN_PHP = 5_000.0
    EXPECTED_REVENUE_MAX_PHP = 30_000.0

    def setUp(self):
        self.inventory = _load_inventory()
        self.benchmark = {
            "average_daily_customers": 50,
            "weekend_multiplier": 1.20,
            "payday_multiplier": 1.40,
            "category_multipliers": {
                "beverage": 1.35,
                "snacks": 1.30,
                "food": 1.20,
            },
        }

    def test_generated_month_revenue_within_benchmark_window(self):
        """Synthetic January 2026 revenue falls inside the PH benchmark
        window. This is the headline "compare to real data" check."""
        txns = generate_monthly_transactions(
            inventory=self.inventory,
            month="2026-01",
            benchmark_info=self.benchmark,
            random_seed=512,
        )
        details = txns.merge(
            self.inventory[["product_id", "unit_price"]],
            on="product_id",
        )
        revenue = float(
            (details["quantity_sold"] * details["unit_price"]).sum()
        )
        self.assertGreaterEqual(revenue, self.EXPECTED_REVENUE_MIN_PHP)
        self.assertLessEqual(revenue, self.EXPECTED_REVENUE_MAX_PHP)

    def test_weekend_lift_is_visible_in_generated_data(self):
        """Weekend multiplier > 1 must produce strictly more units on
        Sat/Sun than the per-day average."""
        txns = generate_monthly_transactions(
            inventory=self.inventory,
            month="2026-01",
            benchmark_info=self.benchmark,
            random_seed=512,
        )
        txns["transaction_date"] = pd.to_datetime(txns["transaction_date"])
        txns["dow"] = txns["transaction_date"].dt.dayofweek
        weekend = txns[txns["dow"].isin([5, 6])]["quantity_sold"].sum()
        weekday = txns[~txns["dow"].isin([5, 6])]["quantity_sold"].sum()
        weekend_days = txns[txns["dow"].isin([5, 6])][
            "transaction_date"].nunique()
        weekday_days = txns[~txns["dow"].isin([5, 6])][
            "transaction_date"].nunique()
        weekend_per_day = weekend / max(weekend_days, 1)
        weekday_per_day = weekday / max(weekday_days, 1)
        self.assertGreater(weekend_per_day, weekday_per_day)

    def test_benchmark_normalization_keeps_defaults_for_missing_keys(self):
        """A partial benchmark dict still produces a full settings dict."""
        partial = {"average_daily_customers": 60}
        normalized = normalize_benchmark_info(partial)
        for key in DEFAULT_BENCHMARK_INFO:
            self.assertIn(key, normalized)
        self.assertEqual(normalized["average_daily_customers"], 60)


# ===========================================================================
# Dashboard + inventory accuracy (PDF: "Accuracy of information reflected in
# Dashboard and Inventory")
# ===========================================================================

class TestDashboardAndInventoryAccuracy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.outputs = run_intermediate_monthly_simulator(
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

    def test_inventory_before_matches_raw_inventory(self):
        """The 'inventory before sales' file mirrors the raw inventory
        master untouched (key columns + values)."""
        raw = _load_inventory().set_index("product_id")
        before = self.outputs["inventory_before_monthly_sales"].set_index(
            "product_id"
        )
        for col in ("product_name", "category", "starting_stock",
                    "unit_cost", "unit_price"):
            self.assertTrue(
                (raw[col].sort_index() == before[col].sort_index()).all(),
                f"inventory_before disagrees with raw on column: {col}",
            )

    def test_dashboard_kpi_revenue_matches_ledger(self):
        """Dashboard `kpi.total_revenue` equals the ledger total."""
        ledger = self.outputs["monthly_ledger_summary"].iloc[0]
        dash = self.outputs["monthly_dashboard_data"]
        kpi_rev = float(
            dash[(dash["metric_group"] == "kpi")
                 & (dash["metric_name"] == "total_revenue")
                 ]["value"].iloc[0]
        )
        self.assertAlmostEqual(kpi_rev, float(ledger["total_revenue"]), 2)

    def test_dashboard_kpi_profit_matches_ledger(self):
        ledger = self.outputs["monthly_ledger_summary"].iloc[0]
        dash = self.outputs["monthly_dashboard_data"]
        kpi_gp = float(
            dash[(dash["metric_group"] == "kpi")
                 & (dash["metric_name"] == "gross_profit")
                 ]["value"].iloc[0]
        )
        self.assertAlmostEqual(kpi_gp, float(ledger["gross_profit"]), 2)

    def test_dashboard_daily_revenue_sums_to_kpi_total(self):
        """Sum of `daily_revenue_trend` rows reconciles to the KPI total."""
        dash = self.outputs["monthly_dashboard_data"]
        daily = dash[
            (dash["metric_group"] == "daily_revenue_trend")
            & (dash["metric_name"] == "daily_revenue")
        ]
        daily_sum = pd.to_numeric(daily["value"]).sum()
        kpi_rev = float(
            dash[(dash["metric_group"] == "kpi")
                 & (dash["metric_name"] == "total_revenue")
                 ]["value"].iloc[0]
        )
        self.assertAlmostEqual(daily_sum, kpi_rev, places=2)

    def test_dashboard_category_revenue_sums_to_kpi_total(self):
        dash = self.outputs["monthly_dashboard_data"]
        cat = dash[
            (dash["metric_group"] == "sales_by_category")
            & (dash["metric_name"] == "category_revenue")
        ]
        self.assertGreater(len(cat), 0)
        cat_sum = pd.to_numeric(cat["value"]).sum()
        kpi_rev = float(
            dash[(dash["metric_group"] == "kpi")
                 & (dash["metric_name"] == "total_revenue")
                 ]["value"].iloc[0]
        )
        self.assertAlmostEqual(cat_sum, kpi_rev, places=2)

    def test_product_summary_remaining_stock_invariant(self):
        """Inventory accuracy: per product, remaining = starting - sold."""
        ps = self.outputs["monthly_product_summary"]
        expected = ps["starting_stock"] - ps["total_quantity_sold"]
        self.assertTrue(
            (ps["remaining_stock"].astype(int).to_numpy()
             == expected.astype(int).to_numpy()).all()
        )

    def test_restock_quantities_non_negative(self):
        restock = self.outputs["restock_recommendations"]
        self.assertTrue(
            (restock["recommended_restock_quantity"] >= 0).all()
        )

    def test_restock_recommended_only_when_below_reorder_point(self):
        """Restock flag is True only when remaining_stock <= reorder_point."""
        restock = self.outputs["restock_recommendations"]
        flagged = restock[restock["recommend_restock"].astype(bool)]
        if not flagged.empty:
            self.assertTrue(
                (flagged["remaining_stock"] <= flagged["reorder_point"]).all()
            )

    def test_dashboard_top_products_sorted_correctly(self):
        """Top-selling products are sorted by units, ascending rank."""
        dash = self.outputs["monthly_dashboard_data"]
        top = dash[
            (dash["metric_group"] == "top_selling_products")
            & (dash["metric_name"] == "total_quantity_sold")
        ].copy()
        if len(top) >= 2:
            top["value_num"] = pd.to_numeric(top["value"])
            top_sorted = top.sort_values("rank")
            values = top_sorted["value_num"].tolist()
            self.assertEqual(values, sorted(values, reverse=True))


if __name__ == "__main__":
    unittest.main()
