"""
advanced_runner.py

Master orchestrator for the Advanced-level Sari-Sari Store Simulator.

Runs all seven advanced modules in the correct dependency order:

    1. sales_event_generator  → generate five years of promo events
    2. five_year_generator    → generate 60 months of transactions (event-boosted)
    3. monthly_outputs        → transaction_details, product_summary, ledger_summary,
                                inventory_before_monthly_sales per month folder
    4. feedback_generator     → generate monthly customer reviews
    5. inventory_optimizer    → generate monthly restock recommendations
    6. pricing_strategy       → generate final pricing recommendations
    7. advanced_dashboard     → five-year dashboard with event peak flags

Each module saves its own CSVs and SQLite tables. This runner coordinates
the pipeline and passes DataFrames between modules so nothing needs to be
re-read from disk between steps.

Each monthly folder now contains all seven required files:
    transactions.csv
    transaction_details.csv
    product_summary.csv
    ledger_summary.csv
    restock_recommendations.csv
    customer_feedback.csv
    sales_events.csv

Plus one additional file:
    inventory_before_monthly_sales.csv

Usage from project root:

    python -m src.advanced.advanced_runner

Or from a Jupyter notebook:

    from src.advanced.advanced_runner import run_advanced_pipeline
    results = run_advanced_pipeline()

Returns a dictionary with all major DataFrames for notebook inspection:
    results["sales_events"]
    results["all_transactions"]
    results["all_transaction_details"]
    results["all_product_summaries"]
    results["all_ledger_summaries"]
    results["all_feedback"]
    results["all_restock"]
    results["pricing_recommendations"]
    results["dashboard"]
"""

from pathlib import Path
import pandas as pd

try:
    from src.advanced.sales_event_generator import run_sales_event_generator
    from src.advanced.five_year_generator   import run_five_year_generator
    from src.advanced.monthly_outputs       import run_monthly_outputs
    from src.advanced.feedback_generator    import run_feedback_generator
    from src.advanced.inventory_optimizer   import run_inventory_optimizer
    from src.advanced.pricing_strategy      import run_pricing_strategy
    from src.advanced.advanced_dashboard    import run_advanced_dashboard
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.advanced.sales_event_generator import run_sales_event_generator
    from src.advanced.five_year_generator   import run_five_year_generator
    from src.advanced.monthly_outputs       import run_monthly_outputs
    from src.advanced.feedback_generator    import run_feedback_generator
    from src.advanced.inventory_optimizer   import run_inventory_optimizer
    from src.advanced.pricing_strategy      import run_pricing_strategy
    from src.advanced.advanced_dashboard    import run_advanced_dashboard


def run_advanced_pipeline(
    inventory_csv_path: str | Path = "data/raw/inventory.csv",
    output_base_folder: str | Path = "data/processed/advanced",
    sqlite_db_path: str | Path = "src/database/sari_sari_store.db",
    random_seed: int = 42,
    save_csv: bool = True,
    save_sqlite: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run the complete Advanced-level pipeline end to end.

    Executes all seven modules in dependency order, passing DataFrames
    between steps so no intermediate disk reads are needed.

    Parameters
    ----------
    inventory_csv_path : str or Path
        Path to data/raw/inventory.csv (never modified).
    output_base_folder : str or Path
        Root output folder for all advanced-level files.
        Monthly files go to year_YYYY/month_MM subfolders.
        Aggregated files go directly into this folder.
    sqlite_db_path : str or Path
        Path to the shared SQLite database.
        Advanced tables are prefixed with advanced_.
    random_seed : int
        Base seed for all random generation.
    save_csv : bool
        Whether to save all CSV output files.
    save_sqlite : bool
        Whether to save all SQLite tables.
    verbose : bool
        Whether to print step-by-step progress to the console.

    Returns
    -------
    dict
        Dictionary with keys:
        - "sales_events"             : pd.DataFrame
        - "all_transactions"         : pd.DataFrame
        - "all_transaction_details"  : pd.DataFrame
        - "all_product_summaries"    : pd.DataFrame
        - "all_ledger_summaries"     : pd.DataFrame
        - "all_feedback"             : pd.DataFrame
        - "all_restock"              : pd.DataFrame
        - "pricing_recommendations"  : pd.DataFrame
        - "dashboard"                : pd.DataFrame
    """
    print("\n" + "=" * 72)
    print("ADVANCED GOAL: FULL FIVE-YEAR PIPELINE")
    print("=" * 72)
    print(f"Inventory source  : {inventory_csv_path}")
    print(f"Output folder     : {output_base_folder}")
    print(f"SQLite database   : {sqlite_db_path}")
    print(f"Random seed       : {random_seed}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Generate sales events
    # Must run before five_year_generator so event boosts are applied
    # ------------------------------------------------------------------
    print("\n[Step 1 / 7] Generating sales events...")
    sales_events = run_sales_event_generator(
        output_base_folder=output_base_folder,
        sqlite_db_path=sqlite_db_path,
        random_seed=random_seed,
        save_csv=save_csv,
        save_sqlite=save_sqlite,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Step 2: Generate five years of transactions
    # Passes sales_events so demand is boosted during promos
    # ------------------------------------------------------------------
    print("\n[Step 2 / 7] Generating five-year transactions...")
    all_transactions = run_five_year_generator(
        inventory_csv_path=inventory_csv_path,
        output_base_folder=output_base_folder,
        sqlite_db_path=sqlite_db_path,
        sales_events=sales_events,
        random_seed=random_seed,
        save_csv=save_csv,
        save_sqlite=save_sqlite,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Step 3: Generate monthly outputs
    # Produces transaction_details, product_summary, ledger_summary,
    # and inventory_before_monthly_sales for every month folder
    # ------------------------------------------------------------------
    print("\n[Step 3 / 7] Generating monthly outputs (details, summaries, inventory)...")
    monthly_results = run_monthly_outputs(
        inventory_csv_path=inventory_csv_path,
        output_base_folder=output_base_folder,
        sqlite_db_path=sqlite_db_path,
        all_transactions=all_transactions,
        save_csv=save_csv,
        save_sqlite=save_sqlite,
        verbose=verbose,
    )
    all_transaction_details = monthly_results["all_transaction_details"]
    all_product_summaries   = monthly_results["all_product_summaries"]
    all_ledger_summaries    = monthly_results["all_ledger_summaries"]

    # ------------------------------------------------------------------
    # Step 4: Generate customer feedback
    # Weighted by transaction volume per product per month
    # ------------------------------------------------------------------
    print("\n[Step 4 / 7] Generating customer feedback...")
    all_feedback = run_feedback_generator(
        inventory_csv_path=inventory_csv_path,
        output_base_folder=output_base_folder,
        sqlite_db_path=sqlite_db_path,
        all_transactions=all_transactions,
        random_seed=random_seed,
        save_csv=save_csv,
        save_sqlite=save_sqlite,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Step 5: Save monthly sales_events.csv per month folder
    # Each year/month folder needs its own sales_events.csv slice
    # ------------------------------------------------------------------
    if save_csv:
        print("\n[Step 5 / 7] Saving monthly sales events per folder...")
        from src.advanced.sales_event_generator import save_monthly_sales_events_csv
        from src.advanced.five_year_generator import START_YEAR, END_YEAR
        count = 0
        for year in range(START_YEAR, END_YEAR + 1):
            for month in range(1, 13):
                save_monthly_sales_events_csv(
                    sales_events=sales_events,
                    year=year,
                    month=month,
                    output_base_folder=output_base_folder,
                )
                count += 1
        if verbose:
            print(f"  Monthly sales_events.csv saved for {count} month folders.")
    else:
        print("\n[Step 5 / 7] Skipping monthly sales events (save_csv=False).")

    # ------------------------------------------------------------------
    # Step 6: Run inventory optimizer
    # Uses five-year transaction history to compute restock needs.
    # Pass all_product_summaries so remaining_stock is accurate.
    # ------------------------------------------------------------------
    print("\n[Step 6 / 7] Running inventory optimizer...")
    all_restock = run_inventory_optimizer(
        inventory_csv_path=inventory_csv_path,
        output_base_folder=output_base_folder,
        sqlite_db_path=sqlite_db_path,
        all_transactions=all_transactions,
        all_product_summaries=all_product_summaries,
        sales_events=sales_events,
        save_csv=save_csv,
        save_sqlite=save_sqlite,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Step 7: Run pricing strategy
    # Uses five-year demand history to recommend prices
    # ------------------------------------------------------------------
    print("\n[Step 7 / 7] Running pricing and dashboard engines...")
    pricing_recommendations = run_pricing_strategy(
        inventory_csv_path=inventory_csv_path,
        output_base_folder=output_base_folder,
        sqlite_db_path=sqlite_db_path,
        all_transactions=all_transactions,
        all_product_summaries=all_product_summaries,
        save_csv=save_csv,
        save_sqlite=save_sqlite,
        verbose=verbose,
    )

    # Build advanced dashboard (event peaks visible in monthly trend)
    dashboard = run_advanced_dashboard(
        output_base_folder=output_base_folder,
        sqlite_db_path=sqlite_db_path,
        all_ledger_summaries=all_ledger_summaries,
        all_product_summaries=all_product_summaries,
        all_transaction_details=all_transaction_details,
        sales_events=sales_events,
        save_csv=save_csv,
        save_sqlite=save_sqlite,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Pipeline complete
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ADVANCED PIPELINE COMPLETE")
    print("=" * 72)
    print(f"Sales events             : {len(sales_events):,} records")
    print(f"Transactions (5yr)       : {len(all_transactions):,} records")
    print(f"Transaction details (5yr): {len(all_transaction_details):,} records")
    print(f"Product summaries (5yr)  : {len(all_product_summaries):,} records")
    print(f"Ledger summaries (5yr)   : {len(all_ledger_summaries):,} records")
    print(f"Customer feedback        : {len(all_feedback):,} records")
    print(f"Restock recommendations  : {len(all_restock):,} records")
    print(f"Pricing recommendations  : {len(pricing_recommendations):,} records")
    print(f"Dashboard rows           : {len(dashboard):,} records")
    print(f"\nSQLite database          : {sqlite_db_path}")
    print(f"Output folder            : {output_base_folder}")

    print("\nMonthly folder contents (each year_YYYY/month_MM/):")
    for f in [
        "transactions.csv",
        "transaction_details.csv",
        "product_summary.csv",
        "ledger_summary.csv",
        "restock_recommendations.csv",
        "customer_feedback.csv",
        "sales_events.csv",
        "inventory_before_monthly_sales.csv",
    ]:
        print(f"  ✓ {f}")
    print("=" * 72)

    return {
        "sales_events":            sales_events,
        "all_transactions":        all_transactions,
        "all_transaction_details": all_transaction_details,
        "all_product_summaries":   all_product_summaries,
        "all_ledger_summaries":    all_ledger_summaries,
        "all_feedback":            all_feedback,
        "all_restock":             all_restock,
        "pricing_recommendations": pricing_recommendations,
        "dashboard":               dashboard,
    }


if __name__ == "__main__":
    run_advanced_pipeline()
