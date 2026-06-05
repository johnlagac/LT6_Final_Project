SARI-SARI STORE SIMULATOR — LEARNING TEAM 6
DSC 512 final project · MAIDA 2027 · Asian Institute of Management
Members: Jaco Abat · Lean Jerios · John Lagac · Justin Rotairo

CONTEXT
-----------
Three-level Python program:
- Basic        : one-day sales calculator (ledger + stock tally)
- Intermediate : one-month simulator with analytics dashboard
- Advanced     : 5-year inventory-optimization + pricing-strategy engine
Binding scope: [LT6 Final Project] Sari-Sari Store Simulator.pdf
Rubric        : Project_guidelines.pdf
                40% Basic + 20% Intermediate + 20% Advanced + 20% Viz/PEP8

DIRECTORY
-----------
LT6_Final_Project/
├── data/
│   ├── raw/                  inventory.csv, transactions.csv (committed)
│   └── processed/            generated CSVs (gitignored, regenerable)
├── notebooks/                basic_demo, intermediate_demo, advanced_demo
├── reports/figures/          PNG chart artifacts (CI-produced, not committed)
├── src/
│   ├── basic/
│   ├── intermediate/
│   ├── advanced/
│   ├── database/
│   └── reports/              chart artifact generator
├── test/                     pytest suites + script sanity checks
└── .github/workflows/ci.yml  CI pipeline (see below)

RUN LOCALLY
-----------
PYTHONPATH="$PWD" python -m src.basic.sales_calculator
PYTHONPATH="$PWD" python -m src.intermediate.monthly_simulator
PYTHONPATH="$PWD" python -m src.advanced.advanced_runner

TESTS
-----------
PYTHONPATH="$PWD" python -m pytest test/ -v
PYTHONPATH="$PWD" python test/intermediate/test_intermediate_sanity_checks.py

REGENERATE CHART ARTIFACTS (LOCALLY)
-----------
PYTHONPATH="$PWD" python -m src.reports.generate_report_charts
# outputs to reports/figures/{basic,intermediate,advanced}/

CI PIPELINE (.github/workflows/ci.yml)
-----------
Runs on every push to main and every pull request.
Jobs run in order, each blocking the next:

  1. test           — matrix on Python 3.12 + 3.13
                      · pytest test/ -v
                      · intermediate script-style sanity script
                      · pycodestyle --max-line-length=99 against
                        E501/E402/W292/E712/W391 (E221 alignment style allowed)
                      · notebook JSON validity check

  2. pipeline-smoke — asserts each level's end-to-end runner produces the
                      expected shape (BALANCED basic ledger; 1-row
                      intermediate ledger + 15 product summary;
                      60-month advanced ledger + 15 pricing recs).

  3. report-artifacts — runs src.reports.generate_report_charts on a
                        clean checkout and uploads two downloadable bundles
                        to the run's Summary page:
                          · lt6-report-charts-<sha>/ — 15 presentation
                            PNGs (basic 3, intermediate 5, advanced 7)
                          · lt6-pipeline-csvs-<sha>/ — full data/processed/
                            tree including 60 month folders × 8 files

How to download from a PR:
  GitHub PR → Checks tab → click "CI" → run summary → "Artifacts" section
  → click the artifact name to download a .zip.

REPORTING REQUIREMENTS
-----------
DSC 512 grades the repo itself plus the demo notebooks (rubric is in
Project_guidelines.pdf). No formal paper or PowerPoint is required for
this course. The PNG artifacts produced by CI are optional aids for any
team-presentation deck.
