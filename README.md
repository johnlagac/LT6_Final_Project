# Sari-Sari Store Simulator

**A Python-Based Retail Business Simulation and Analytics Tool.**
DSC 512 final project — MAIDA 2027, Asian Institute of Management.
**Learning Team 6:** Jaco Abat · Lean Jerios · John Lagac · Justin Rotairo.

This is a standalone three-level Python program that simulates the operation
of a small Philippine sari-sari store. It starts with a basic one-day sales
calculator, evolves into a month-long simulator with analytics, and ends as
an inventory optimization + pricing strategy engine over five years of data.

---

## Project levels

| Level | What it does | Key entrypoint |
|---|---|---|
| **Basic** — one-day sales calculator | Loads `inventory.csv` + `transactions.csv`, validates, computes revenue / expense / gross profit, tracks remaining stock, writes a daily ledger with `BALANCED` / `CHECK_LEDGER` status. | `src/basic/sales_calculator.py` → `run_basic_sales_calculator()` |
| **Intermediate** — one-month simulator with analytics | Synthetic month generator parameterised by PH benchmarks (payday + weekend + per-category multipliers), inventory-before snapshot, restock plan for next month, normalised monthly dashboard table + charts. | `src/intermediate/monthly_simulator.py` → `run_intermediate_monthly_simulator()` |
| **Advanced** — Inventory Optimization Engine + Pricing Strategy Analysis | Generates 5 years of synthetic transactions (one folder per year × month), monthly grocery-wide sales events, monthly customer feedback, inventory optimizer with CRITICAL/HIGH/MEDIUM/LOW risk + restock quantities, pricing recommendations from 60-month demand trend + margin, dashboard with event-peak flags. | `src/advanced/advanced_runner.py` → `run_advanced_pipeline()` |

---

## Repo layout

```
LT6_Final_Project/
├── data/
│   ├── raw/                       # inventory.csv, transactions.csv (committed)
│   └── processed/                 # all generated CSVs (gitignored)
├── notebooks/
│   ├── basic_demo.ipynb
│   ├── intermediate_demo.ipynb
│   └── advanced_demo.ipynb
├── reports/
│   └── figures/                   # PNG chart artifacts for the deck
│       ├── basic/
│       ├── intermediate/
│       └── advanced/
├── src/
│   ├── basic/
│   ├── intermediate/
│   ├── advanced/
│   ├── database/
│   └── reports/
│       └── generate_report_charts.py
└── test/
    ├── basic/basic_test.py
    ├── intermediate/test_intermediate.py
    ├── intermediate/test_intermediate_sanity_checks.py
    └── advanced/
        ├── test_advanced.py
        └── test_advanced_anchor.py
```

---

## Install

```bash
# Python 3.12 or 3.13 (anaconda/miniconda recommended for jupyter-sql magic)
pip install pandas numpy matplotlib sqlalchemy ipython-sql jupysql "prettytable>=3.12.0"
```

A `requirements.txt`-style minimum: `pandas`, `numpy`, `matplotlib`,
`sqlalchemy`. The notebooks additionally use `jupysql` + `prettytable>=3.12.0`
for SQL Magic — the lower bound matches `jupysql`'s requirement; older
`prettytable` versions break `%%sql` cell rendering. The first cell in each
notebook installs the right combination.

---

## Run

### As a Python module (terminal)

```bash
PYTHONPATH="$PWD" python -m src.basic.sales_calculator
PYTHONPATH="$PWD" python -m src.intermediate.monthly_simulator
PYTHONPATH="$PWD" python -m src.advanced.advanced_runner
```

### As Jupyter notebooks

```bash
jupyter notebook notebooks/
```

Open and run-all on:

* `basic_demo.ipynb` — one-day ledger demo + SQL Magic checks
* `intermediate_demo.ipynb` — synthetic month + dashboard + SQL Magic
* `advanced_demo.ipynb` — five-year pipeline + event-peak chart + SQL Magic

### Regenerate the chart artifacts

```bash
PYTHONPATH="$PWD" python -m src.reports.generate_report_charts
```

Writes 15 PNGs under `reports/figures/{basic,intermediate,advanced}/` —
suitable for direct use in the final-deliverable deck.

---

## Run the tests

```bash
PYTHONPATH="$PWD" python -m pytest test/ -v
```

Expected: **247 passing.**

* `test/basic/basic_test.py` — **60 unit tests** covering loaders,
  validation, ledger math, stock reconciliation, save helpers
* `test/intermediate/test_intermediate_sanity_checks.py` — script-style
  end-to-end check (run directly with
  `python test/intermediate/test_intermediate_sanity_checks.py`)
* `test/intermediate/test_intermediate.py` — **12 pytest anchors** for
  "How to Test": revenue vs PH benchmark window, dashboard/inventory
  accuracy, restock invariants
* `test/advanced/test_advanced.py` — **161 unit tests** covering all
  seven advanced modules
* `test/advanced/test_advanced_anchor.py` — **14 pytest anchors** for
  "How to Test": recommended prices vs PH competitor window, event-day
  vs non-event-day sales peaks

---

## Outputs

* `data/processed/basic/` — one-day CSV outputs
* `data/processed/intermediate/` — month CSV outputs + dashboard PNG charts
* `data/processed/advanced/year_YYYY/month_MM/` — eight CSVs per month
  (transactions, transaction_details, product_summary, ledger_summary,
  restock_recommendations, customer_feedback, sales_events,
  inventory_before_monthly_sales)
* `data/processed/advanced/advanced_*.csv` — aggregated 5-year tables
* `src/database/sari_sari_store.db` — SQLite database with three sets of
  tables: `daily_*` (basic), `intermediate_*`, `advanced_*`
* `reports/figures/` — committed presentation-ready PNG charts

---

## How it maps to the project rubric

| Rubric category | Weight | Where to look |
|---|---:|---|
| Basic goal + tests | 40% | `src/basic/`, `test/basic/basic_test.py`, `notebooks/basic_demo.ipynb` |
| Intermediate goal + tests | 20% | `src/intermediate/`, `test/intermediate/`, `notebooks/intermediate_demo.ipynb` |
| Advanced goal + tests | 20% | `src/advanced/`, `test/advanced/`, `notebooks/advanced_demo.ipynb` |
| Visualizations + PEP 8 | 20% | `reports/figures/` + every notebook's chart section. `pycodestyle --max-line-length=99` is clean on `E501/E402/W292/E712/W391`. |

---

## CI pipeline (`.github/workflows/ci.yml`)

Runs on every push to `main` and every pull request. Three jobs, each
blocking the next:

1. **`test`** — matrix on Python 3.12 + 3.13
   * `pytest test/ -v`
   * Intermediate script-style sanity script
   * `pycodestyle --max-line-length=99 --select=E501,E402,W292,E712,W391` —
     `E221` (vertical-alignment style) is intentionally excluded
   * Notebook JSON validity check
2. **`pipeline-smoke`** — asserts each level's end-to-end runner produces
   the expected shape: `BALANCED` basic ledger; 1-row intermediate ledger
   + 15-product summary; 60-month advanced ledger + 15 pricing
   recommendations.
3. **`report-artifacts`** — runs `src.reports.generate_report_charts` on a
   clean checkout and uploads two downloadable bundles to the run's
   Summary page:
   * `lt6-report-charts-<sha>` — 15 presentation-ready PNGs
     (basic 3 / intermediate 5 / advanced 7), retained 30 days
   * `lt6-pipeline-csvs-<sha>` — full `data/processed/` tree including
     60 month folders × 8 files, retained 14 days

To download from a PR: **Checks** tab → click `CI` run → **Artifacts**
section at the bottom → click the artifact name for a `.zip`.

---

## Reporting requirements

DSC 512 grades the repo itself plus the demo notebooks (rubric in
`Project_guidelines.pdf`). **No formal paper or PowerPoint is required**
for this course. The PNG artifacts produced by CI are optional aids for
any team-presentation deck and reinforce the 20% Visualizations bucket as
concrete evidence even if a grader doesn't execute every notebook.
