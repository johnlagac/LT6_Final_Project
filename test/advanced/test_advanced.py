"""
test_advanced.py

Unit tests for the Advanced-level Sari-Sari Store Simulator modules.

Tests cover every major public function across all five files:
    - five_year_generator.py
    - sales_event_generator.py
    - feedback_generator.py
    - inventory_optimizer.py
    - pricing_strategy.py

Each test uses a minimal synthetic inventory (3 products) to keep
execution fast. No files are written to disk and no SQLite connections
are opened unless the test explicitly calls a save function.

Run from project root:
    python -m pytest test/test_advanced.py -v

Or without pytest:
    python test/test_advanced.py
"""

import sys
import unittest
import tempfile
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Make sure project root is on the path so src.* imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced.five_year_generator import (
    _growth_multiplier,
    _expected_daily_qty,
    generate_monthly_transactions,
    save_monthly_transactions,
    save_five_year_transactions_csv,
    START_YEAR,
    END_YEAR,
    ANNUAL_GROWTH_RATE,
    BASELINE_DAILY_UNITS,
    MONTHLY_SEASONALITY,
)
from src.advanced.sales_event_generator import (
    _demand_multiplier,
    _safe_date,
    generate_sales_events,
    get_events_for_month,
    EVENT_TEMPLATES,
)
from src.advanced.feedback_generator import (
    _sentiment_from_rating,
    _pick_comment,
    generate_monthly_feedback,
    SENTIMENT_WEIGHTS,
    FEEDBACK_PER_PRODUCT_MIN,
    FEEDBACK_PER_PRODUCT_MAX,
)
from src.advanced.inventory_optimizer import (
    _classify_stockout_risk,
    _build_recommendation_reason,
    calculate_monthly_restock,
    REORDER_THRESHOLD,
    RESTOCK_MONTHS_COVERAGE,
    MIN_RESTOCK_UNITS,
)
from src.advanced.pricing_strategy import (
    _compute_gross_margin,
    _compute_demand_trend,
    _recommend_price,
    build_pricing_recommendations,
    LOW_MARGIN_THRESHOLD,
    HIGH_MARGIN_THRESHOLD,
    PRICE_INCREASE_RATE,
    PRICE_DECREASE_RATE,
)
from src.advanced.sales_event_generator import DEMAND_ELASTICITY


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_inventory() -> pd.DataFrame:
    """
    Minimal 3-product inventory matching the real schema from inventory.csv.
    Categories match the real data: Beverage, Food, Household.
    """
    return pd.DataFrame([
        {
            "product_id":    "P001",
            "product_name":  "Coke 1L",
            "category":      "Beverage",
            "starting_stock": 24,
            "unit_cost":     60,
            "unit_price":    75,
        },
        {
            "product_id":    "P010",
            "product_name":  "Egg",
            "category":      "Food",
            "starting_stock": 60,
            "unit_cost":     7,
            "unit_price":    10,
        },
        {
            "product_id":    "P014",
            "product_name":  "Laundry Detergent Sachet",
            "category":      "Household",
            "starting_stock": 75,
            "unit_cost":     5,
            "unit_price":    8,
        },
    ])


def make_sales_events() -> pd.DataFrame:
    """
    Two simple sales events for testing: one for Beverage, one for Food.
    """
    return pd.DataFrame([
        {
            "event_id":          "EVT-2022-001",
            "event_name":        "Summer Beverage Sale",
            "start_date":        pd.Timestamp("2022-06-01"),
            "end_date":          pd.Timestamp("2022-06-14"),
            "discount_rate":     0.15,
            "affected_category": "Beverage",
            "demand_multiplier": 1.45,
        },
        {
            "event_id":          "EVT-2022-002",
            "event_name":        "Christmas Food Promo",
            "start_date":        pd.Timestamp("2022-12-15"),
            "end_date":          pd.Timestamp("2022-12-31"),
            "discount_rate":     0.20,
            "affected_category": "Food",
            "demand_multiplier": 1.60,
        },
    ])


def make_transactions(year: int = 2022, month: int = 1) -> pd.DataFrame:
    """
    Minimal synthetic transactions matching the real schema.
    """
    return pd.DataFrame([
        {
            "transaction_id":   f"ADV-{year}{month:02d}-00001",
            "transaction_date": pd.Timestamp(date(year, month, 1)),
            "product_id":       "P001",
            "quantity_sold":    5,
        },
        {
            "transaction_id":   f"ADV-{year}{month:02d}-00002",
            "transaction_date": pd.Timestamp(date(year, month, 1)),
            "product_id":       "P010",
            "quantity_sold":    20,
        },
        {
            "transaction_id":   f"ADV-{year}{month:02d}-00003",
            "transaction_date": pd.Timestamp(date(year, month, 15)),
            "product_id":       "P014",
            "quantity_sold":    10,
        },
    ])


# ===========================================================================
# TEST: five_year_generator.py
# ===========================================================================

class TestGrowthMultiplier(unittest.TestCase):

    def test_base_year_returns_one(self):
        """Growth multiplier for the base year (2022) should be exactly 1.0."""
        self.assertAlmostEqual(_growth_multiplier(START_YEAR), 1.0)

    def test_one_year_growth(self):
        """2023 should be (1 + ANNUAL_GROWTH_RATE)^1."""
        expected = (1 + ANNUAL_GROWTH_RATE) ** 1
        self.assertAlmostEqual(_growth_multiplier(2023), expected)

    def test_five_year_growth(self):
        """2026 should be (1 + ANNUAL_GROWTH_RATE)^4."""
        expected = (1 + ANNUAL_GROWTH_RATE) ** 4
        self.assertAlmostEqual(_growth_multiplier(2026), expected)

    def test_growth_is_always_positive(self):
        """Multiplier must be > 1 for all years after the base year."""
        for year in range(START_YEAR, END_YEAR + 1):
            self.assertGreater(_growth_multiplier(year), 0)


class TestExpectedDailyQty(unittest.TestCase):

    def setUp(self):
        self.sale_date = date(2022, 1, 1)  # January — seasonality = 1.00
        self.no_events = pd.DataFrame(
            columns=["start_date", "end_date",
                     "affected_category", "demand_multiplier"]
        )

    def test_known_product_uses_baseline(self):
        """P001 (Coke 1L) baseline is 5 units/day."""
        qty = _expected_daily_qty("P001", "Beverage", self.sale_date, self.no_events)
        # In Jan 2022: base=5, seasonality=1.0, growth=1.0, DOW varies
        self.assertGreater(qty, 0)

    def test_unknown_product_uses_default(self):
        """Unknown product ID should use DEFAULT_DAILY_UNITS."""
        from src.advanced.five_year_generator import DEFAULT_DAILY_UNITS
        qty = _expected_daily_qty("P999", "Beverage", self.sale_date, self.no_events)
        # Should be > 0 based on default
        self.assertGreater(qty, 0)
        # Should be roughly DEFAULT_DAILY_UNITS (with DOW multiplier)
        self.assertLess(qty, DEFAULT_DAILY_UNITS * 2)

    def test_event_boosts_demand(self):
        """Active event for matching category should increase expected qty."""
        events = pd.DataFrame([{
            "start_date":        date(2022, 1, 1),
            "end_date":          date(2022, 1, 31),
            "affected_category": "Beverage",
            "demand_multiplier": 1.5,
        }])
        qty_with_event    = _expected_daily_qty(
            "P001", "Beverage", self.sale_date, events
        )
        qty_without_event = _expected_daily_qty(
            "P001", "Beverage", self.sale_date, self.no_events
        )
        self.assertGreater(qty_with_event, qty_without_event)

    def test_event_for_different_category_has_no_effect(self):
        """Event for Food should not affect Beverage demand."""
        events = pd.DataFrame([{
            "start_date":        date(2022, 1, 1),
            "end_date":          date(2022, 1, 31),
            "affected_category": "Food",
            "demand_multiplier": 2.0,
        }])
        qty_with_event    = _expected_daily_qty(
            "P001", "Beverage", self.sale_date, events
        )
        qty_without_event = _expected_daily_qty(
            "P001", "Beverage", self.sale_date, self.no_events
        )
        self.assertAlmostEqual(qty_with_event, qty_without_event)


class TestGenerateMonthlyTransactions(unittest.TestCase):

    def setUp(self):
        self.inventory = make_inventory()

    def test_returns_dataframe(self):
        result = generate_monthly_transactions(self.inventory, 2022, 1)
        self.assertIsInstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        result = generate_monthly_transactions(self.inventory, 2022, 1)
        for col in ["transaction_id", "transaction_date",
                    "product_id", "quantity_sold"]:
            self.assertIn(col, result.columns)

    def test_transaction_ids_are_unique(self):
        result = generate_monthly_transactions(self.inventory, 2022, 1)
        self.assertEqual(
            result["transaction_id"].nunique(), len(result),
            "transaction_id values must be unique within a month"
        )

    def test_transaction_id_format(self):
        """IDs should start with ADV- to distinguish from Basic T001 format."""
        result = generate_monthly_transactions(self.inventory, 2022, 1)
        self.assertTrue(
            result["transaction_id"].str.startswith("ADV-").all()
        )

    def test_all_dates_within_target_month(self):
        result = generate_monthly_transactions(self.inventory, 2022, 6)
        dates = pd.to_datetime(result["transaction_date"])
        self.assertTrue((dates.dt.year == 2022).all())
        self.assertTrue((dates.dt.month == 6).all())

    def test_all_products_from_inventory(self):
        """Only product_ids from inventory should appear in transactions."""
        result = generate_monthly_transactions(self.inventory, 2022, 1)
        valid_ids = set(self.inventory["product_id"])
        self.assertTrue(set(result["product_id"]).issubset(valid_ids))

    def test_quantity_sold_always_positive(self):
        result = generate_monthly_transactions(self.inventory, 2022, 1)
        self.assertTrue((result["quantity_sold"] >= 1).all())

    def test_reproducible_with_same_seed(self):
        r1 = generate_monthly_transactions(self.inventory, 2022, 3, random_seed=99)
        r2 = generate_monthly_transactions(self.inventory, 2022, 3, random_seed=99)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seeds_produce_different_results(self):
        r1 = generate_monthly_transactions(self.inventory, 2022, 3, random_seed=1)
        r2 = generate_monthly_transactions(self.inventory, 2022, 3, random_seed=2)
        self.assertFalse(r1.equals(r2))

    def test_different_months_produce_different_volumes(self):
        """December (Christmas) should have higher volume than February."""
        jan = generate_monthly_transactions(self.inventory, 2022, 2)
        dec = generate_monthly_transactions(self.inventory, 2022, 12)
        self.assertGreater(
            dec["quantity_sold"].sum(),
            jan["quantity_sold"].sum(),
            "December should outsell February due to seasonality"
        )

    def test_later_year_has_higher_volume(self):
        """2026 should sell more than 2022 due to annual growth."""
        y2022 = generate_monthly_transactions(self.inventory, 2022, 6)
        y2026 = generate_monthly_transactions(self.inventory, 2026, 6)
        self.assertGreater(
            y2026["quantity_sold"].sum(),
            y2022["quantity_sold"].sum(),
        )

    def test_events_increase_demand(self):
        """Transactions with active event should exceed those without."""
        events = make_sales_events()
        with_events    = generate_monthly_transactions(
            self.inventory, 2022, 6, sales_events=events
        )
        without_events = generate_monthly_transactions(
            self.inventory, 2022, 6, sales_events=None
        )
        self.assertGreater(
            with_events["quantity_sold"].sum(),
            without_events["quantity_sold"].sum(),
        )

    def test_save_monthly_transactions_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txns = generate_monthly_transactions(self.inventory, 2022, 1)
            path = save_monthly_transactions(txns, 2022, 1, tmpdir)
            self.assertTrue(path.exists())
            loaded = pd.read_csv(path)
            self.assertEqual(len(loaded), len(txns))

    def test_save_creates_correct_folder_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txns = generate_monthly_transactions(self.inventory, 2023, 5)
            path = save_monthly_transactions(txns, 2023, 5, tmpdir)
            self.assertIn("year_2023", str(path))
            self.assertIn("month_05", str(path))

    def test_save_five_year_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txns = generate_monthly_transactions(self.inventory, 2022, 1)
            path = save_five_year_transactions_csv(txns, tmpdir)
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "advanced_transactions_5yr.csv")


# ===========================================================================
# TEST: sales_event_generator.py
# ===========================================================================

class TestDemandMultiplier(unittest.TestCase):

    def test_zero_discount_gives_multiplier_one(self):
        self.assertAlmostEqual(_demand_multiplier(0.0), 1.0)

    def test_known_discount(self):
        """15% discount at elasticity 3.0 → 1 + 0.15*3.0 = 1.45."""
        self.assertAlmostEqual(_demand_multiplier(0.15), 1.45)

    def test_multiplier_always_above_one(self):
        for rate in [0.05, 0.10, 0.15, 0.20, 0.25]:
            self.assertGreater(_demand_multiplier(rate), 1.0)

    def test_higher_discount_gives_higher_multiplier(self):
        self.assertGreater(_demand_multiplier(0.20), _demand_multiplier(0.10))


class TestSafeDate(unittest.TestCase):

    def test_normal_date(self):
        self.assertEqual(_safe_date(2022, 6, 15), date(2022, 6, 15))

    def test_clamps_day_to_month_end(self):
        """Day 31 in February should clamp to Feb 28 (non-leap year)."""
        result = _safe_date(2022, 2, 31)
        self.assertEqual(result, date(2022, 2, 28))

    def test_clamps_day_to_month_end_leap(self):
        """Day 31 in February 2024 (leap year) should clamp to Feb 29."""
        result = _safe_date(2024, 2, 31)
        self.assertEqual(result, date(2024, 2, 29))

    def test_december_31(self):
        self.assertEqual(_safe_date(2022, 12, 31), date(2022, 12, 31))


class TestGenerateSalesEvents(unittest.TestCase):

    def setUp(self):
        self.events = generate_sales_events(
            start_year=2022, end_year=2026, random_seed=42
        )

    def test_returns_dataframe(self):
        self.assertIsInstance(self.events, pd.DataFrame)

    def test_required_columns_present(self):
        for col in ["event_id", "event_name", "start_date", "end_date",
                    "discount_rate", "affected_category", "demand_multiplier"]:
            self.assertIn(col, self.events.columns)

    def test_event_count(self):
        """Should have len(EVENT_TEMPLATES) events per year × 5 years."""
        expected = len(EVENT_TEMPLATES) * (2026 - 2022 + 1)
        self.assertEqual(len(self.events), expected)

    def test_all_event_ids_unique(self):
        self.assertEqual(
            self.events["event_id"].nunique(), len(self.events)
        )

    def test_end_date_after_start_date(self):
        self.assertTrue(
            (self.events["end_date"] >= self.events["start_date"]).all()
        )

    def test_demand_multipliers_all_above_one(self):
        self.assertTrue((self.events["demand_multiplier"] > 1.0).all())

    def test_discount_rates_between_zero_and_one(self):
        self.assertTrue((self.events["discount_rate"] > 0).all())
        self.assertTrue((self.events["discount_rate"] < 1).all())

    def test_categories_are_valid(self):
        valid = {"Beverage", "Food", "Snack", "Personal Care", "Household"}
        for cat in self.events["affected_category"].unique():
            self.assertIn(cat, valid)

    def test_events_span_all_five_years(self):
        years_covered = set(
            pd.to_datetime(self.events["start_date"]).dt.year.unique()
        )
        self.assertEqual(years_covered, {2022, 2023, 2024, 2025, 2026})

    def test_reproducible_with_same_seed(self):
        e1 = generate_sales_events(random_seed=7)
        e2 = generate_sales_events(random_seed=7)
        pd.testing.assert_frame_equal(e1, e2)


class TestGetEventsForMonth(unittest.TestCase):

    def setUp(self):
        self.events = make_sales_events()

    def test_returns_events_active_in_month(self):
        """June 2022 should return the Summer Beverage Sale."""
        result = get_events_for_month(self.events, 2022, 6)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["event_id"], "EVT-2022-001")

    def test_returns_empty_for_month_with_no_events(self):
        result = get_events_for_month(self.events, 2022, 3)
        self.assertEqual(len(result), 0)

    def test_partial_overlap_included(self):
        """December 2022 should include the Christmas Food Promo."""
        result = get_events_for_month(self.events, 2022, 12)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["event_id"], "EVT-2022-002")

    def test_different_year_returns_empty(self):
        result = get_events_for_month(self.events, 2023, 6)
        self.assertEqual(len(result), 0)


# ===========================================================================
# TEST: feedback_generator.py
# ===========================================================================

class TestSentimentFromRating(unittest.TestCase):

    def test_high_rating_positive(self):
        self.assertEqual(_sentiment_from_rating(4.0), "positive")
        self.assertEqual(_sentiment_from_rating(5.0), "positive")
        self.assertEqual(_sentiment_from_rating(4.5), "positive")

    def test_mid_rating_neutral(self):
        self.assertEqual(_sentiment_from_rating(3.0), "neutral")
        self.assertEqual(_sentiment_from_rating(3.9), "neutral")

    def test_low_rating_negative(self):
        self.assertEqual(_sentiment_from_rating(1.0), "negative")
        self.assertEqual(_sentiment_from_rating(2.0), "negative")
        self.assertEqual(_sentiment_from_rating(2.9), "negative")


class TestPickComment(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_returns_string(self):
        comment = _pick_comment("positive", "Beverage", self.rng)
        self.assertIsInstance(comment, str)
        self.assertGreater(len(comment), 0)

    def test_works_for_all_sentiments(self):
        for sentiment in ["positive", "neutral", "negative"]:
            comment = _pick_comment(sentiment, "Food", self.rng)
            self.assertIsInstance(comment, str)

    def test_works_for_all_categories(self):
        for category in ["Beverage", "Food", "Snack", "Personal Care", "Household"]:
            comment = _pick_comment("positive", category, self.rng)
            self.assertIsInstance(comment, str)

    def test_unknown_category_returns_default(self):
        """Unknown category should not crash — falls back to default."""
        comment = _pick_comment("positive", "Electronics", self.rng)
        self.assertIsInstance(comment, str)


class TestGenerateMonthlyFeedback(unittest.TestCase):

    def setUp(self):
        self.inventory = make_inventory()

    def test_returns_dataframe(self):
        result = generate_monthly_feedback(self.inventory, 2022, 1)
        self.assertIsInstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        result = generate_monthly_feedback(self.inventory, 2022, 1)
        for col in ["feedback_id", "feedback_date", "product_id",
                    "product_name", "category", "rating",
                    "comment", "sentiment"]:
            self.assertIn(col, result.columns)

    def test_feedback_ids_unique(self):
        result = generate_monthly_feedback(self.inventory, 2022, 1)
        self.assertEqual(
            result["feedback_id"].nunique(), len(result)
        )

    def test_feedback_id_format(self):
        result = generate_monthly_feedback(self.inventory, 2022, 3)
        self.assertTrue(
            result["feedback_id"].str.startswith("FB-202203").all()
        )

    def test_ratings_within_valid_range(self):
        result = generate_monthly_feedback(self.inventory, 2022, 1)
        self.assertTrue((result["rating"] >= 1.0).all())
        self.assertTrue((result["rating"] <= 5.0).all())

    def test_sentiment_values_are_valid(self):
        result = generate_monthly_feedback(self.inventory, 2022, 1)
        valid = {"positive", "neutral", "negative"}
        self.assertTrue(set(result["sentiment"].unique()).issubset(valid))

    def test_sentiment_consistent_with_rating(self):
        """Sentiment must match the rating according to the defined rules."""
        result = generate_monthly_feedback(self.inventory, 2022, 1)
        for _, row in result.iterrows():
            if row["rating"] >= 4.0:
                self.assertEqual(row["sentiment"], "positive")
            elif row["rating"] >= 3.0:
                self.assertEqual(row["sentiment"], "neutral")
            else:
                self.assertEqual(row["sentiment"], "negative")

    def test_all_products_appear_in_feedback(self):
        """Every product in inventory should have at least one feedback entry."""
        result = generate_monthly_feedback(self.inventory, 2022, 1)
        for pid in self.inventory["product_id"]:
            self.assertIn(pid, result["product_id"].values)

    def test_feedback_dates_within_target_month(self):
        result = generate_monthly_feedback(self.inventory, 2022, 6)
        dates = pd.to_datetime(result["feedback_date"])
        self.assertTrue((dates.dt.year == 2022).all())
        self.assertTrue((dates.dt.month == 6).all())

    def test_min_feedback_per_product(self):
        """Each product must receive at least FEEDBACK_PER_PRODUCT_MIN reviews."""
        result = generate_monthly_feedback(self.inventory, 2022, 1)
        for pid in self.inventory["product_id"]:
            count = (result["product_id"] == pid).sum()
            self.assertGreaterEqual(count, FEEDBACK_PER_PRODUCT_MIN)

    def test_weighted_feedback_high_seller_gets_more(self):
        """Product with higher transactions should receive more feedback."""
        txns = pd.DataFrame([
            {"transaction_id": "A", "transaction_date": "2022-01-01",
             "product_id": "P010", "quantity_sold": 100},
            {"transaction_id": "B", "transaction_date": "2022-01-01",
             "product_id": "P001", "quantity_sold": 5},
            {"transaction_id": "C", "transaction_date": "2022-01-01",
             "product_id": "P014", "quantity_sold": 5},
        ])
        result = generate_monthly_feedback(
            self.inventory, 2022, 1, monthly_transactions=txns
        )
        egg_count  = (result["product_id"] == "P010").sum()
        coke_count = (result["product_id"] == "P001").sum()
        self.assertGreaterEqual(egg_count, coke_count)

    def test_reproducible_with_same_seed(self):
        r1 = generate_monthly_feedback(self.inventory, 2022, 1, random_seed=5)
        r2 = generate_monthly_feedback(self.inventory, 2022, 1, random_seed=5)
        pd.testing.assert_frame_equal(r1, r2)


# ===========================================================================
# TEST: inventory_optimizer.py
# ===========================================================================

class TestClassifyStockoutRisk(unittest.TestCase):

    def test_critical_risk(self):
        """Under 7 days of stock should be CRITICAL."""
        self.assertEqual(_classify_stockout_risk(10, 5.0), "CRITICAL")

    def test_high_risk(self):
        """Under 14 days should be HIGH."""
        self.assertEqual(_classify_stockout_risk(50, 5.0), "HIGH")

    def test_medium_risk(self):
        """Under 21 days should be MEDIUM."""
        self.assertEqual(_classify_stockout_risk(80, 5.0), "MEDIUM")

    def test_low_risk(self):
        """More than 21 days of stock should be LOW."""
        self.assertEqual(_classify_stockout_risk(500, 5.0), "LOW")

    def test_zero_daily_sales_is_low_risk(self):
        """If nothing is selling, there is no stockout risk."""
        self.assertEqual(_classify_stockout_risk(0, 0.0), "LOW")

    def test_zero_stock_with_sales_is_critical(self):
        """No stock but active demand is CRITICAL."""
        self.assertEqual(_classify_stockout_risk(0, 5.0), "CRITICAL")


class TestBuildRecommendationReason(unittest.TestCase):

    def test_returns_string(self):
        reason = _build_recommendation_reason(
            risk="HIGH",
            current_stock=20,
            avg_monthly_demand=100,
            restock_qty=130,
            has_upcoming_event=False,
        )
        self.assertIsInstance(reason, str)
        self.assertGreater(len(reason), 0)

    def test_event_flag_adds_event_text(self):
        with_event    = _build_recommendation_reason(
            "HIGH", 20, 100, 130, has_upcoming_event=True
        )
        without_event = _build_recommendation_reason(
            "HIGH", 20, 100, 130, has_upcoming_event=False
        )
        self.assertIn("event", with_event.lower())
        self.assertNotIn("event", without_event.lower())

    def test_zero_restock_mentions_sufficient(self):
        reason = _build_recommendation_reason(
            "LOW", 200, 50, restock_qty=0, has_upcoming_event=False
        )
        self.assertIn("sufficient", reason.lower())


class TestCalculateMonthlyRestock(unittest.TestCase):

    def setUp(self):
        self.inventory = make_inventory()

    def _make_product_summary(self, remaining_stocks: dict) -> pd.DataFrame:
        """Build a product summary with specified remaining stocks."""
        rows = []
        for pid, remaining in remaining_stocks.items():
            rows.append({
                "product_id":          pid,
                "total_quantity_sold": 20,
                "remaining_stock":     remaining,
            })
        return pd.DataFrame(rows)

    def _make_historical_demand(self, units_per_product: dict) -> pd.DataFrame:
        rows = []
        for pid, units in units_per_product.items():
            rows.append({
                "product_id":          pid,
                "total_quantity_sold": units,
            })
        return pd.DataFrame(rows)

    def test_returns_dataframe(self):
        summary  = self._make_product_summary(
            {"P001": 5, "P010": 10, "P014": 15}
        )
        hist     = self._make_historical_demand(
            {"P001": 100, "P010": 500, "P014": 200}
        )
        result = calculate_monthly_restock(
            self.inventory, summary, hist, year=2022, month=6
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        summary = self._make_product_summary(
            {"P001": 5, "P010": 10, "P014": 15}
        )
        hist = self._make_historical_demand(
            {"P001": 100, "P010": 500, "P014": 200}
        )
        result = calculate_monthly_restock(
            self.inventory, summary, hist, year=2022, month=6
        )
        for col in [
            "product_id", "product_name", "category",
            "average_monthly_quantity_sold", "average_daily_sales",
            "stockout_risk", "recommended_restock_quantity",
            "recommendation_reason", "year", "month",
        ]:
            self.assertIn(col, result.columns)

    def test_low_stock_triggers_restock(self):
        """Product with stock well below 50% of avg monthly demand → restock."""
        summary = self._make_product_summary({"P001": 1, "P010": 1, "P014": 1})
        hist    = self._make_historical_demand(
            {"P001": 600, "P010": 600, "P014": 600}
        )
        result = calculate_monthly_restock(
            self.inventory, summary, hist, year=2022, month=6
        )
        # Avg monthly demand = 600/6 months = 100; remaining=1 << 50 (threshold)
        restock_row = result[result["product_id"] == "P001"].iloc[0]
        self.assertGreater(restock_row["recommended_restock_quantity"], 0)

    def test_high_stock_no_restock_needed(self):
        """Product with stock well above reorder threshold → no restock."""
        summary = self._make_product_summary(
            {"P001": 500, "P010": 500, "P014": 500}
        )
        hist = self._make_historical_demand(
            {"P001": 60, "P010": 60, "P014": 60}
        )
        result = calculate_monthly_restock(
            self.inventory, summary, hist, year=2022, month=6
        )
        for _, row in result.iterrows():
            self.assertEqual(
                row["recommended_restock_quantity"], 0,
                f"{row['product_id']} should not need restock"
            )

    def test_restock_qty_never_negative(self):
        """Recommended restock quantity must always be >= 0."""
        summary = self._make_product_summary(
            {"P001": 100, "P010": 100, "P014": 100}
        )
        hist = self._make_historical_demand(
            {"P001": 10, "P010": 10, "P014": 10}
        )
        result = calculate_monthly_restock(
            self.inventory, summary, hist, year=2022, month=1
        )
        self.assertTrue(
            (result["recommended_restock_quantity"] >= 0).all()
        )

    def test_year_and_month_columns_correct(self):
        summary = self._make_product_summary(
            {"P001": 5, "P010": 5, "P014": 5}
        )
        hist = self._make_historical_demand(
            {"P001": 60, "P010": 60, "P014": 60}
        )
        result = calculate_monthly_restock(
            self.inventory, summary, hist, year=2024, month=8
        )
        self.assertTrue((result["year"]  == 2024).all())
        self.assertTrue((result["month"] == 8).all())


# ===========================================================================
# TEST: pricing_strategy.py
# ===========================================================================

class TestComputeGrossMargin(unittest.TestCase):

    def test_known_margin_coke(self):
        """Coke 1L: cost=60, price=75 → margin=(75-60)/75=0.20."""
        self.assertAlmostEqual(_compute_gross_margin(60, 75), 0.20)

    def test_known_margin_coffee_sachet(self):
        """Coffee Sachet: cost=5, price=8 → (8-5)/8=0.375."""
        self.assertAlmostEqual(_compute_gross_margin(5, 8), 0.375)

    def test_known_margin_rice(self):
        """Rice 1kg: cost=42, price=50 → (50-42)/50=0.16."""
        self.assertAlmostEqual(_compute_gross_margin(42, 50), 0.16)

    def test_zero_price_returns_zero(self):
        self.assertEqual(_compute_gross_margin(10, 0), 0.0)

    def test_margin_between_zero_and_one(self):
        for cost, price in [(5, 8), (7, 10), (60, 75), (42, 50)]:
            margin = _compute_gross_margin(cost, price)
            self.assertGreater(margin, 0.0)
            self.assertLess(margin, 1.0)


class TestComputeDemandTrend(unittest.TestCase):

    def test_increasing_trend(self):
        """Clear upward trend should return 'increasing'."""
        units = pd.Series([50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110])
        trend, pct = _compute_demand_trend(units)
        self.assertEqual(trend, "increasing")
        self.assertGreater(pct, 0)

    def test_decreasing_trend(self):
        """Clear downward trend should return 'decreasing'."""
        units = pd.Series([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45])
        trend, pct = _compute_demand_trend(units)
        self.assertEqual(trend, "decreasing")
        self.assertLess(pct, 0)

    def test_stable_trend(self):
        """Flat demand should return 'stable'."""
        units = pd.Series([100] * 12)
        trend, pct = _compute_demand_trend(units)
        self.assertEqual(trend, "stable")
        self.assertAlmostEqual(pct, 0.0)

    def test_single_value_returns_stable(self):
        units = pd.Series([50])
        trend, pct = _compute_demand_trend(units)
        self.assertEqual(trend, "stable")

    def test_empty_series_returns_stable(self):
        units = pd.Series([], dtype=float)
        trend, pct = _compute_demand_trend(units)
        self.assertEqual(trend, "stable")


class TestRecommendPrice(unittest.TestCase):

    def test_increasing_demand_low_margin_raises_price(self):
        """Demand UP + margin below LOW_MARGIN_THRESHOLD → price increase."""
        price, reason = _recommend_price(
            unit_cost=80,
            unit_price=100,
            gross_margin=0.20,      # below LOW_MARGIN_THRESHOLD (0.25)
            demand_trend="increasing",
            current_stock=20,
            avg_monthly_demand=50,
        )
        expected = round(100 * (1 + PRICE_INCREASE_RATE), 2)
        self.assertAlmostEqual(price, expected)
        self.assertIn("increasing", reason.lower())

    def test_decreasing_demand_high_stock_lowers_price(self):
        """Demand DOWN + stock high → price decrease."""
        price, reason = _recommend_price(
            unit_cost=7,
            unit_price=10,
            gross_margin=0.30,
            demand_trend="decreasing",
            current_stock=200,      # >> avg_monthly * HIGH_STOCK_THRESHOLD
            avg_monthly_demand=50,
        )
        expected = round(10 * (1 - PRICE_DECREASE_RATE), 2)
        self.assertAlmostEqual(price, expected)
        self.assertIn("decreasing", reason.lower())

    def test_high_margin_holds_price(self):
        """Margin above HIGH_MARGIN_THRESHOLD → hold price."""
        price, reason = _recommend_price(
            unit_cost=4,
            unit_price=7,
            gross_margin=0.43,      # above HIGH_MARGIN_THRESHOLD (0.35)
            demand_trend="stable",
            current_stock=50,
            avg_monthly_demand=100,
        )
        self.assertAlmostEqual(price, 7.0)
        self.assertIn("holding", reason.lower())

    def test_price_floor_respected(self):
        """Recommended price must never fall below cost * 1.10."""
        price, _ = _recommend_price(
            unit_cost=100,
            unit_price=101,         # tiny margin
            gross_margin=0.01,
            demand_trend="decreasing",
            current_stock=1000,
            avg_monthly_demand=5,
        )
        min_floor = round(100 * 1.10, 2)
        self.assertGreaterEqual(price, min_floor)

    def test_stable_demand_acceptable_margin_holds_price(self):
        price, reason = _recommend_price(
            unit_cost=5,
            unit_price=8,
            gross_margin=0.375,     # above HIGH_MARGIN_THRESHOLD → "holding"
            demand_trend="stable",
            current_stock=30,
            avg_monthly_demand=50,
        )
        self.assertAlmostEqual(price, 8.0)
        self.assertIn("holding", reason.lower())


class TestBuildPricingRecommendations(unittest.TestCase):

    def setUp(self):
        self.inventory = make_inventory()

    def _make_five_year_transactions(self) -> pd.DataFrame:
        """Build minimal 60-month transaction history for 3 products."""
        rows = []
        counter = 1
        for year in range(2022, 2027):
            for month in range(1, 13):
                for pid, qty in [("P001", 5), ("P010", 20), ("P014", 10)]:
                    rows.append({
                        "transaction_id":   f"ADV-{year}{month:02d}-{counter:05d}",
                        "transaction_date": f"{year}-{month:02d}-01",
                        "product_id":       pid,
                        "quantity_sold":    qty,
                    })
                    counter += 1
        return pd.DataFrame(rows)

    def test_returns_dataframe(self):
        txns   = self._make_five_year_transactions()
        result = build_pricing_recommendations(self.inventory, txns)
        self.assertIsInstance(result, pd.DataFrame)

    def test_one_row_per_product(self):
        txns   = self._make_five_year_transactions()
        result = build_pricing_recommendations(self.inventory, txns)
        self.assertEqual(len(result), len(self.inventory))

    def test_required_columns_present(self):
        txns   = self._make_five_year_transactions()
        result = build_pricing_recommendations(self.inventory, txns)
        for col in [
            "product_id", "product_name", "category",
            "current_unit_cost", "current_unit_price",
            "average_monthly_quantity_sold",
            "gross_margin", "demand_trend",
            "recommended_price", "pricing_reason",
        ]:
            self.assertIn(col, result.columns)

    def test_recommended_price_always_above_cost(self):
        txns   = self._make_five_year_transactions()
        result = build_pricing_recommendations(self.inventory, txns)
        merged = result.merge(
            self.inventory[["product_id", "unit_cost"]], on="product_id"
        )
        self.assertTrue(
            (merged["recommended_price"] > merged["unit_cost"]).all()
        )

    def test_demand_trend_valid_values(self):
        txns   = self._make_five_year_transactions()
        result = build_pricing_recommendations(self.inventory, txns)
        valid  = {"increasing", "stable", "decreasing"}
        self.assertTrue(
            set(result["demand_trend"].unique()).issubset(valid)
        )

    def test_gross_margin_between_zero_and_one(self):
        txns   = self._make_five_year_transactions()
        result = build_pricing_recommendations(self.inventory, txns)
        self.assertTrue((result["gross_margin"] > 0).all())
        self.assertTrue((result["gross_margin"] < 1).all())

    def test_avg_monthly_qty_positive(self):
        txns   = self._make_five_year_transactions()
        result = build_pricing_recommendations(self.inventory, txns)
        self.assertTrue((result["average_monthly_quantity_sold"] > 0).all())

    def test_sorted_by_product_id(self):
        txns   = self._make_five_year_transactions()
        result = build_pricing_recommendations(self.inventory, txns)
        ids    = result["product_id"].tolist()
        self.assertEqual(ids, sorted(ids))


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(__import__("__main__"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
