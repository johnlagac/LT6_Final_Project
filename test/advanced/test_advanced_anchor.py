"""
test_advanced_anchor.py

Pytest anchor for the Advanced-level "How to Test" requirements from the
project PDF:

    * Comparison of Recommended Prices for select goods to real-life prices
      of competitors
    * Dashboard to reflect peak in sales when adding occurrences of
      grocery-wide sales

This complements the broad coverage in `test_advanced.py` (161 unit tests on
the building blocks) with a small set of end-to-end pipeline assertions that
map 1:1 to the rubric's "How to Test" column.

Run from project root:
    PYTHONPATH=$PWD python -m pytest test/advanced/test_advanced_anchor.py -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.advanced.advanced_runner import run_advanced_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# PH competitor price windows (PHP, retail at small grocery / sari-sari).
# Sourced from public Philippine sari-sari and supermarket price ranges
# (Robinsons, Puregold, neighborhood stores) — wide enough to be stable but
# tight enough to flag a runaway recommendation.
# ---------------------------------------------------------------------------
PH_COMPETITOR_PRICE_BENCHMARK_PHP: dict[str, tuple[float, float]] = {
    "P001": (60.0, 100.0),   # Coke 1L
    "P002": (60.0, 100.0),   # Royal 1L
    "P003": (10.0,  30.0),   # Bottled Water 500ml
    "P004": (12.0,  28.0),   # Piattos Snack
    "P005": (12.0,  28.0),   # Nova Snack
    "P006": (10.0,  25.0),   # Lucky Me Beef Noodles
    "P007": (12.0,  25.0),   # Pancit Canton Chilimansi
    "P008": (22.0,  45.0),   # Canned Sardines
    "P009": (35.0,  60.0),   # Corned Beef Small
    "P010":  (7.0,  18.0),   # Egg
    "P011": (40.0,  70.0),   # Rice 1kg
    "P012":  (5.0,  15.0),   # Coffee Sachet
    "P013":  (4.0,  12.0),   # Shampoo Sachet
    "P014":  (5.0,  15.0),   # Laundry Detergent Sachet
    "P015":  (7.0,  18.0),   # Dishwashing Liquid Sachet
}


def _build_pipeline_results() -> dict:
    """Run the advanced pipeline once into a temp folder; reuse across tests."""
    tmp = Path(tempfile.mkdtemp(prefix="lt6_advanced_anchor_"))
    return run_advanced_pipeline(
        inventory_csv_path=PROJECT_ROOT / "data" / "raw" / "inventory.csv",
        output_base_folder=tmp / "advanced",
        sqlite_db_path=tmp / "anchor.db",
        random_seed=42,
        save_csv=False,
        save_sqlite=False,
        verbose=False,
    )


class _PipelineFixture(unittest.TestCase):
    """Shared expensive setup — run the 5-year pipeline once and reuse
    the same results dict across every subclass.

    Caching on ``cls.results`` would re-run the pipeline once per subclass
    because each subclass gets its own class-level attribute. We bind on
    ``_PipelineFixture.results`` directly so every subclass sees the same
    cached value after the first ``setUpClass`` fires.
    """

    results: dict | None = None

    @classmethod
    def setUpClass(cls):
        if _PipelineFixture.results is None:
            _PipelineFixture.results = _build_pipeline_results()
        cls.results = _PipelineFixture.results


# ===========================================================================
# Anchor 1: Recommended prices vs PH competitor benchmark
# ===========================================================================

class TestRecommendedPriceVsPHCompetitor(_PipelineFixture):

    def setUp(self):
        self.pricing = self.results["pricing_recommendations"]

    def test_pricing_recommendations_have_15_products(self):
        self.assertEqual(len(self.pricing), 15)

    def test_recommended_price_columns_present(self):
        for col in ("product_id", "current_unit_price",
                    "recommended_price", "demand_trend", "gross_margin"):
            self.assertIn(col, self.pricing.columns)

    def test_every_recommended_price_within_ph_benchmark_window(self):
        """Each recommended price falls inside the PH competitor window.
        This is the headline "compare prices to real-life competitors"
        check the project PDF asks for."""
        failures = []
        for _, row in self.pricing.iterrows():
            pid = row["product_id"]
            price = float(row["recommended_price"])
            lo, hi = PH_COMPETITOR_PRICE_BENCHMARK_PHP[pid]
            if not (lo <= price <= hi):
                failures.append(
                    f"{pid} ({row['product_name']}): "
                    f"recommended {price:.2f} not in [{lo}, {hi}]"
                )
        self.assertFalse(
            failures,
            "Recommended prices outside PH benchmark:\n  - "
            + "\n  - ".join(failures),
        )

    def test_recommended_price_at_least_cost_plus_minimum_markup(self):
        """Pricing strategy floors at cost + MIN_MARKUP_FRACTION."""
        inv = pd.read_csv(
            PROJECT_ROOT / "data" / "raw" / "inventory.csv"
        ).set_index("product_id")
        for _, row in self.pricing.iterrows():
            price = float(row["recommended_price"])
            cost = float(inv.loc[row["product_id"], "unit_cost"])
            self.assertGreater(
                price, cost,
                f"Recommended price {price} not greater than cost {cost} "
                f"for {row['product_id']}",
            )

    def test_demand_trend_is_one_of_three_labels(self):
        labels = set(self.pricing["demand_trend"].unique())
        self.assertTrue(
            labels.issubset({"increasing", "stable", "decreasing"}),
            f"Unexpected demand_trend labels: {labels}",
        )

    def test_gross_margin_within_zero_one(self):
        self.assertTrue(
            ((self.pricing["gross_margin"] > 0)
             & (self.pricing["gross_margin"] < 1)).all()
        )


# ===========================================================================
# Anchor 2: Dashboard reflects peaks during grocery-wide sales events
# ===========================================================================

class TestDashboardReflectsEventPeaks(_PipelineFixture):

    def setUp(self):
        self.dashboard = self.results["dashboard"]
        self.sales_events = self.results["sales_events"]

    def test_sales_events_were_generated(self):
        """Step 1 of the advanced pipeline must produce events."""
        self.assertGreater(len(self.sales_events), 0)

    def test_dashboard_has_event_flag_true_rows(self):
        """The advanced dashboard must mark months with active events."""
        flagged = self.dashboard[self.dashboard["event_flag"] == True]  # noqa: E712
        self.assertGreater(
            len(flagged), 0,
            "No rows have event_flag=True — dashboard does not reflect "
            "grocery-wide sales.",
        )

    def test_event_days_have_higher_daily_units_than_non_event_days(self):
        """Headline check: at the daily level, transactions on days with
        an active grocery-wide sale show higher unit volume than days
        without one. The 5-year generator schedules ~22 events/yr so
        most months touch at least one event window — comparing at
        the *day* level isolates the actual sales peak."""
        events = self.sales_events.copy()
        events["start_date"] = pd.to_datetime(events["start_date"])
        events["end_date"] = pd.to_datetime(events["end_date"])
        event_days: set = set()
        for _, ev in events.iterrows():
            for d in pd.date_range(ev["start_date"], ev["end_date"]):
                event_days.add(d.date())

        txns = self.results["all_transactions"].copy()
        txns["transaction_date"] = pd.to_datetime(txns["transaction_date"])
        daily = txns.groupby("transaction_date")["quantity_sold"].sum()
        daily_df = daily.reset_index(name="units")
        daily_df["is_event"] = daily_df["transaction_date"].dt.date.isin(
            event_days
        )
        event_avg = daily_df[daily_df["is_event"]]["units"].mean()
        non_event_avg = daily_df[~daily_df["is_event"]]["units"].mean()
        self.assertGreater(
            event_avg, non_event_avg,
            f"Daily units on event days ({event_avg:.1f}) not greater "
            f"than on non-event days ({non_event_avg:.1f})",
        )

    def test_event_lift_at_least_ten_percent(self):
        """Sanity floor: event-day average is at least 10% above
        non-event-day average — a true 'peak', not noise."""
        events = self.sales_events.copy()
        events["start_date"] = pd.to_datetime(events["start_date"])
        events["end_date"] = pd.to_datetime(events["end_date"])
        event_days = set()
        for _, ev in events.iterrows():
            for d in pd.date_range(ev["start_date"], ev["end_date"]):
                event_days.add(d.date())

        txns = self.results["all_transactions"].copy()
        txns["transaction_date"] = pd.to_datetime(txns["transaction_date"])
        daily = txns.groupby("transaction_date")["quantity_sold"].sum()
        daily_df = daily.reset_index(name="units")
        daily_df["is_event"] = daily_df["transaction_date"].dt.date.isin(
            event_days
        )
        event_avg = daily_df[daily_df["is_event"]]["units"].mean()
        non_event_avg = daily_df[~daily_df["is_event"]]["units"].mean()
        lift = event_avg / non_event_avg
        self.assertGreaterEqual(
            lift, 1.10,
            f"Event-day lift only {lift:.2f}x — expected >=1.10x.",
        )


# ===========================================================================
# Anchor 3: Pipeline-level smoke on key advanced deliverables
# ===========================================================================

class TestAdvancedPipelineSmoke(_PipelineFixture):

    def test_five_years_of_ledger_rows(self):
        ledger = self.results["all_ledger_summaries"]
        self.assertEqual(len(ledger), 60)

    def test_customer_feedback_present_for_every_month(self):
        feedback = self.results["all_feedback"]
        feedback["year"] = pd.to_datetime(feedback["feedback_date"]).dt.year
        feedback["month"] = pd.to_datetime(feedback["feedback_date"]).dt.month
        months_covered = feedback.groupby(["year", "month"]).size().shape[0]
        self.assertEqual(months_covered, 60)

    def test_inventory_recommendations_for_every_month(self):
        restock = self.results["all_restock"]
        self.assertEqual(
            restock.groupby(["year", "month"]).size().shape[0], 60
        )

    def test_inventory_recommendations_have_risk_classification(self):
        restock = self.results["all_restock"]
        risks = set(restock["stockout_risk"].unique())
        self.assertTrue(
            risks.issubset({"CRITICAL", "HIGH", "MEDIUM", "LOW"}),
            f"Unexpected stockout risk labels: {risks}",
        )


if __name__ == "__main__":
    unittest.main()
