"""
basic_test.py

Comprehensive unit tests and sanity checks for the Basic-level Sari-Sari Store
Simulator.

Covers:
    data_loader.py      - column cleaning, required-column validation,
                          inventory + transactions loading, cross-file linkage
    sales_calculator.py - create_sales_details, create_product_summary,
                          create_ledger_summary, save helpers

Each test uses a small in-memory inventory and transaction frame so execution
is fast and no files are written to disk unless a save function is explicitly
exercised.

Run from project root:
    PYTHONPATH=$PWD python -m pytest test/basic/basic_test.py -v

Or directly:
    PYTHONPATH=$PWD python test/basic/basic_test.py
"""
import unittest
import tempfile
from pathlib import Path

import pandas as pd

from src.basic.data_loader import (
    INVENTORY_REQUIRED_COLUMNS,
    TRANSACTIONS_REQUIRED_COLUMNS,
    clean_column_names,
    load_inventory,
    load_transactions,
    validate_required_columns,
    validate_transactions_match_inventory,
)
from src.basic.sales_calculator import (
    create_ledger_summary,
    create_product_summary,
    create_sales_details,
    save_output_csv_files,
    save_tables_to_sqlite,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_inventory() -> pd.DataFrame:
    """Return a small validated inventory DataFrame (3 products)."""
    return pd.DataFrame({
        "product_id":    ["P001", "P002", "P003"],
        "product_name":  ["Coke 1L", "Piattos Snack", "Coffee Sachet"],
        "category":      ["Beverage", "Snack", "Beverage"],
        "starting_stock": [24, 36, 80],
        "unit_cost":     [60, 14, 5],
        "unit_price":    [75, 18, 8],
    })


def make_transactions(date: str = "2026-01-01") -> pd.DataFrame:
    """Return a small validated transactions DataFrame (4 transactions)."""
    return pd.DataFrame({
        "transaction_id":   ["T001", "T002", "T003", "T004"],
        "transaction_date": pd.to_datetime([date] * 4),
        "product_id":       ["P001", "P002", "P002", "P003"],
        "quantity_sold":    [2, 3, 1, 5],
    })


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


# ===========================================================================
# TEST: clean_column_names
# ===========================================================================

class TestCleanColumnNames(unittest.TestCase):

    def test_lowercases_columns(self):
        df = pd.DataFrame(columns=["Product_ID", "Unit Price"])
        result = clean_column_names(df)
        self.assertIn("product_id", result.columns)
        self.assertIn("unit_price", result.columns)

    def test_replaces_spaces_with_underscore(self):
        df = pd.DataFrame(columns=["Product Name"])
        result = clean_column_names(df)
        self.assertIn("product_name", result.columns)

    def test_replaces_hyphens_with_underscore(self):
        df = pd.DataFrame(columns=["Unit-Cost"])
        result = clean_column_names(df)
        self.assertIn("unit_cost", result.columns)

    def test_strips_whitespace(self):
        df = pd.DataFrame(columns=[" product_id ", "  category  "])
        result = clean_column_names(df)
        self.assertIn("product_id", result.columns)
        self.assertIn("category", result.columns)

    def test_returns_copy_not_original(self):
        df = pd.DataFrame(columns=["X"])
        result = clean_column_names(df)
        self.assertIsNot(df, result)


# ===========================================================================
# TEST: validate_required_columns
# ===========================================================================

class TestValidateRequiredColumns(unittest.TestCase):

    def test_passes_when_all_columns_present(self):
        df = pd.DataFrame(columns=["a", "b", "c"])
        # Should not raise
        validate_required_columns(df, ["a", "b"], "test_file")

    def test_raises_when_column_missing(self):
        df = pd.DataFrame(columns=["a", "b"])
        with self.assertRaises(ValueError) as ctx:
            validate_required_columns(df, ["a", "b", "c"], "test_file")
        self.assertIn("c", str(ctx.exception))
        self.assertIn("test_file", str(ctx.exception))

    def test_raises_lists_all_missing(self):
        df = pd.DataFrame(columns=["a"])
        with self.assertRaises(ValueError) as ctx:
            validate_required_columns(df, ["a", "b", "c"], "f")
        self.assertIn("b", str(ctx.exception))
        self.assertIn("c", str(ctx.exception))


# ===========================================================================
# TEST: load_inventory
# ===========================================================================

class TestLoadInventory(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "inventory.csv"
        write_csv(make_inventory(), self.path)

    def test_returns_dataframe(self):
        result = load_inventory(self.path)
        self.assertIsInstance(result, pd.DataFrame)

    def test_loads_all_required_columns(self):
        result = load_inventory(self.path)
        for col in INVENTORY_REQUIRED_COLUMNS:
            self.assertIn(col, result.columns)

    def test_loads_correct_row_count(self):
        result = load_inventory(self.path)
        self.assertEqual(len(result), 3)

    def test_numeric_columns_are_numeric(self):
        result = load_inventory(self.path)
        for col in ["starting_stock", "unit_cost", "unit_price"]:
            self.assertTrue(pd.api.types.is_numeric_dtype(result[col]))

    def test_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_inventory(self.tmp / "does_not_exist.csv")

    def test_raises_on_missing_column(self):
        bad = make_inventory().drop(columns=["unit_price"])
        bad_path = self.tmp / "bad_inventory.csv"
        write_csv(bad, bad_path)
        with self.assertRaises(ValueError) as ctx:
            load_inventory(bad_path)
        self.assertIn("unit_price", str(ctx.exception))

    def test_raises_on_negative_unit_cost(self):
        bad = make_inventory()
        bad.loc[0, "unit_cost"] = -1
        bad_path = self.tmp / "neg_cost_inventory.csv"
        write_csv(bad, bad_path)
        with self.assertRaises(ValueError) as ctx:
            load_inventory(bad_path)
        self.assertIn("negative", str(ctx.exception).lower())

    def test_raises_on_negative_starting_stock(self):
        bad = make_inventory()
        bad.loc[0, "starting_stock"] = -5
        bad_path = self.tmp / "neg_stock_inventory.csv"
        write_csv(bad, bad_path)
        with self.assertRaises(ValueError):
            load_inventory(bad_path)

    def test_raises_on_duplicate_product_id(self):
        bad = make_inventory()
        bad.loc[2, "product_id"] = "P001"
        bad_path = self.tmp / "dup_inventory.csv"
        write_csv(bad, bad_path)
        with self.assertRaises(ValueError) as ctx:
            load_inventory(bad_path)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_raises_on_blank_product_name(self):
        # A whitespace-only cell becomes "" after .str.strip() and trips the
        # blank-text guard. (An empty cell would be parsed as NaN earlier.)
        bad = make_inventory()
        bad.loc[1, "product_name"] = "   "
        bad_path = self.tmp / "blank_inventory.csv"
        write_csv(bad, bad_path)
        with self.assertRaises(ValueError) as ctx:
            load_inventory(bad_path)
        self.assertIn("blank", str(ctx.exception).lower())

    def test_strips_whitespace_in_text_columns(self):
        bad = make_inventory()
        bad.loc[0, "product_id"] = "  P001  "
        bad_path = self.tmp / "ws_inventory.csv"
        write_csv(bad, bad_path)
        result = load_inventory(bad_path)
        self.assertEqual(result.loc[0, "product_id"], "P001")


# ===========================================================================
# TEST: load_transactions
# ===========================================================================

class TestLoadTransactions(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "transactions.csv"
        write_csv(make_transactions(), self.path)

    def test_returns_dataframe_with_correct_rows(self):
        result = load_transactions(self.path)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)

    def test_loads_all_required_columns(self):
        result = load_transactions(self.path)
        for col in TRANSACTIONS_REQUIRED_COLUMNS:
            self.assertIn(col, result.columns)

    def test_transaction_date_is_datetime(self):
        result = load_transactions(self.path)
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(result["transaction_date"])
        )

    def test_quantity_is_numeric(self):
        result = load_transactions(self.path)
        self.assertTrue(pd.api.types.is_numeric_dtype(result["quantity_sold"]))

    def test_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_transactions(self.tmp / "missing.csv")

    def test_raises_on_invalid_date(self):
        df = pd.DataFrame({
            "transaction_id":   ["T001"],
            "transaction_date": ["not-a-date"],
            "product_id":       ["P001"],
            "quantity_sold":    [1],
        })
        bad_path = self.tmp / "bad_date.csv"
        write_csv(df, bad_path)
        with self.assertRaises(ValueError) as ctx:
            load_transactions(bad_path)
        self.assertIn("transaction_date", str(ctx.exception))

    def test_raises_on_negative_quantity(self):
        df = make_transactions()
        df.loc[0, "quantity_sold"] = -1
        bad_path = self.tmp / "neg_qty.csv"
        write_csv(df, bad_path)
        with self.assertRaises(ValueError):
            load_transactions(bad_path)

    def test_raises_on_duplicate_transaction_id(self):
        df = make_transactions()
        df.loc[3, "transaction_id"] = "T001"
        bad_path = self.tmp / "dup_txn.csv"
        write_csv(df, bad_path)
        with self.assertRaises(ValueError) as ctx:
            load_transactions(bad_path)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_raises_on_non_numeric_quantity(self):
        df = pd.DataFrame({
            "transaction_id":   ["T001"],
            "transaction_date": ["2026-01-01"],
            "product_id":       ["P001"],
            "quantity_sold":    ["abc"],
        })
        bad_path = self.tmp / "bad_qty.csv"
        write_csv(df, bad_path)
        with self.assertRaises(ValueError):
            load_transactions(bad_path)


# ===========================================================================
# TEST: validate_transactions_match_inventory
# ===========================================================================

class TestValidateTransactionsMatchInventory(unittest.TestCase):

    def test_passes_when_all_products_known(self):
        inv = make_inventory()
        txn = make_transactions()
        # Should not raise
        validate_transactions_match_inventory(txn, inv)

    def test_raises_when_unknown_product_id(self):
        inv = make_inventory()
        txn = make_transactions()
        txn.loc[3, "product_id"] = "P999"
        with self.assertRaises(ValueError) as ctx:
            validate_transactions_match_inventory(txn, inv)
        self.assertIn("P999", str(ctx.exception))

    def test_lists_all_unknown_ids(self):
        inv = make_inventory()
        txn = make_transactions()
        txn.loc[2, "product_id"] = "P998"
        txn.loc[3, "product_id"] = "P999"
        with self.assertRaises(ValueError) as ctx:
            validate_transactions_match_inventory(txn, inv)
        self.assertIn("P998", str(ctx.exception))
        self.assertIn("P999", str(ctx.exception))


# ===========================================================================
# TEST: create_sales_details (transaction-level math)
# ===========================================================================

class TestCreateSalesDetails(unittest.TestCase):

    def setUp(self):
        self.inv = make_inventory()
        self.txn = make_transactions()
        self.details = create_sales_details(self.inv, self.txn)

    def test_returns_one_row_per_transaction(self):
        self.assertEqual(len(self.details), len(self.txn))

    def test_revenue_equals_quantity_times_unit_price(self):
        expected = (
            self.details["quantity_sold"] * self.details["unit_price"]
        )
        self.assertTrue(
            (self.details["revenue"].astype(float)
             == expected.astype(float)).all()
        )

    def test_expense_equals_quantity_times_unit_cost(self):
        expected = (
            self.details["quantity_sold"] * self.details["unit_cost"]
        )
        self.assertTrue(
            (self.details["expense"].astype(float)
             == expected.astype(float)).all()
        )

    def test_gross_profit_equals_revenue_minus_expense(self):
        expected = self.details["revenue"] - self.details["expense"]
        self.assertTrue(
            (self.details["gross_profit"].astype(float)
             == expected.astype(float)).all()
        )

    def test_includes_inventory_columns(self):
        for col in ["product_name", "category", "unit_cost", "unit_price"]:
            self.assertIn(col, self.details.columns)

    def test_no_null_revenue_after_merge(self):
        self.assertFalse(self.details["revenue"].isna().any())

    def test_revenue_strictly_positive(self):
        self.assertTrue((self.details["revenue"] > 0).all())


# ===========================================================================
# TEST: create_product_summary
# ===========================================================================

class TestCreateProductSummary(unittest.TestCase):

    def setUp(self):
        inv = make_inventory()
        txn = make_transactions()
        self.details = create_sales_details(inv, txn)
        self.summary = create_product_summary(self.details)

    def test_returns_one_row_per_product(self):
        self.assertEqual(len(self.summary), 3)

    def test_remaining_stock_equals_starting_minus_sold(self):
        for _, row in self.summary.iterrows():
            self.assertEqual(
                row["remaining_stock"],
                row["starting_stock"] - row["total_quantity_sold"],
            )

    def test_stock_status_OK_when_positive(self):
        ok_rows = self.summary[self.summary["remaining_stock"] > 0]
        self.assertTrue((ok_rows["stock_status"] == "OK").all())

    def test_aggregates_revenue_correctly(self):
        # P002 has two transactions (qty 3 and 1) at unit_price 18 -> 72
        p002 = self.summary[self.summary["product_id"] == "P002"].iloc[0]
        self.assertEqual(p002["total_revenue"], 72)

    def test_aggregates_quantity_correctly(self):
        p002 = self.summary[self.summary["product_id"] == "P002"].iloc[0]
        self.assertEqual(p002["total_quantity_sold"], 4)

    def test_total_gross_profit_consistent_with_per_product(self):
        row_profit = (
            self.summary["total_revenue"] - self.summary["total_expense"]
        )
        self.assertTrue(
            (self.summary["total_gross_profit"].astype(float).reset_index(drop=True)
             == row_profit.astype(float).reset_index(drop=True)).all()
        )


# ===========================================================================
# TEST: create_ledger_summary
# ===========================================================================

class TestCreateLedgerSummary(unittest.TestCase):

    def setUp(self):
        inv = make_inventory()
        txn = make_transactions()
        self.details = create_sales_details(inv, txn)
        self.summary = create_product_summary(self.details)
        self.ledger = create_ledger_summary(self.details, self.summary)

    def test_returns_one_row(self):
        self.assertEqual(len(self.ledger), 1)

    def test_total_revenue_matches_details(self):
        self.assertEqual(
            self.ledger.iloc[0]["total_revenue"],
            self.details["revenue"].sum(),
        )

    def test_total_expense_matches_details(self):
        self.assertEqual(
            self.ledger.iloc[0]["total_expense"],
            self.details["expense"].sum(),
        )

    def test_gross_profit_equals_revenue_minus_expense(self):
        row = self.ledger.iloc[0]
        self.assertEqual(
            row["gross_profit"],
            row["total_revenue"] - row["total_expense"],
        )

    def test_transaction_count_matches_details(self):
        self.assertEqual(
            self.ledger.iloc[0]["number_of_transactions"],
            len(self.details),
        )

    def test_unique_products_count_correct(self):
        self.assertEqual(
            self.ledger.iloc[0]["unique_products_sold"],
            self.details["product_id"].nunique(),
        )

    def test_ledger_status_BALANCED_when_clean(self):
        self.assertEqual(self.ledger.iloc[0]["ledger_status"], "BALANCED")

    def test_ledger_status_CHECK_when_multiple_dates(self):
        details = self.details.copy()
        details.loc[3, "transaction_date"] = pd.Timestamp("2026-01-02")
        summary = create_product_summary(details)
        ledger = create_ledger_summary(details, summary)
        self.assertEqual(ledger.iloc[0]["ledger_status"], "CHECK_LEDGER")
        self.assertEqual(
            ledger.iloc[0]["transaction_date"], "MULTIPLE_DATES_FOUND"
        )

    def test_ledger_flags_insufficient_stock(self):
        summary = self.summary.copy()
        summary.loc[0, "remaining_stock"] = -3
        ledger = create_ledger_summary(self.details, summary)
        self.assertEqual(ledger.iloc[0]["insufficient_stock_products"], 1)
        self.assertEqual(ledger.iloc[0]["ledger_status"], "CHECK_LEDGER")

    def test_transaction_date_string_format(self):
        # Single-date input should serialise to ISO YYYY-MM-DD.
        self.assertEqual(
            self.ledger.iloc[0]["transaction_date"], "2026-01-01"
        )


# ===========================================================================
# TEST: end-to-end load -> calculate -> reconcile
# ===========================================================================

class TestEndToEndReconciliation(unittest.TestCase):
    """High-confidence sanity checks the rubric asks for in the project PDF:
    full ledger check + balance trace, all stocks tally, revenue and
    expenses add up."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        write_csv(make_inventory(),    self.tmp / "inventory.csv")
        write_csv(make_transactions(), self.tmp / "transactions.csv")

        self.inv = load_inventory(self.tmp / "inventory.csv")
        self.txn = load_transactions(self.tmp / "transactions.csv")
        self.details = create_sales_details(self.inv, self.txn)
        self.summary = create_product_summary(self.details)
        self.ledger = create_ledger_summary(self.details, self.summary)

    def test_full_ledger_balances(self):
        """Revenue - Expense computed two independent ways must match."""
        from_details = (
            self.details["revenue"].sum() - self.details["expense"].sum()
        )
        from_ledger = self.ledger.iloc[0]["gross_profit"]
        self.assertEqual(from_details, from_ledger)

    def test_stocks_tally(self):
        """Sum of (starting - remaining) per product equals total qty sold."""
        per_product_sold = (
            self.summary["starting_stock"] - self.summary["remaining_stock"]
        ).sum()
        total_sold = self.ledger.iloc[0]["total_quantity_sold"]
        self.assertEqual(per_product_sold, total_sold)

    def test_product_summary_revenue_sums_to_ledger(self):
        self.assertEqual(
            self.summary["total_revenue"].sum(),
            self.ledger.iloc[0]["total_revenue"],
        )

    def test_product_summary_expense_sums_to_ledger(self):
        self.assertEqual(
            self.summary["total_expense"].sum(),
            self.ledger.iloc[0]["total_expense"],
        )


# ===========================================================================
# TEST: save helpers (smoke test - produce files / sqlite tables)
# ===========================================================================

class TestSaveHelpers(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.inv = make_inventory()
        self.txn = make_transactions()
        self.details = create_sales_details(self.inv, self.txn)
        self.summary = create_product_summary(self.details)
        self.ledger = create_ledger_summary(self.details, self.summary)

    def test_save_output_csv_files_writes_three_files(self):
        save_output_csv_files(
            sales_details=self.details,
            product_summary=self.summary,
            ledger_summary=self.ledger,
            output_folder=self.tmp,
        )
        for fname in (
            "daily_transaction_details.csv",
            "daily_product_summary.csv",
            "daily_ledger_summary.csv",
        ):
            self.assertTrue((self.tmp / fname).exists(), fname)

    def test_save_tables_to_sqlite_creates_all_tables(self):
        db_path = self.tmp / "test.db"
        save_tables_to_sqlite(
            inventory=self.inv,
            transactions=self.txn,
            sales_details=self.details,
            product_summary=self.summary,
            ledger_summary=self.ledger,
            sqlite_db_path=db_path,
        )
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            tables = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "ORDER BY name"
                )
            ]
        for tbl in (
            "daily_ledger_summary",
            "daily_product_summary",
            "daily_transaction_details",
            "inventory",
            "transactions",
        ):
            self.assertIn(tbl, tables)


if __name__ == "__main__":
    unittest.main()
