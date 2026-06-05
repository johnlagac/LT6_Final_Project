"""
feedback_generator.py

Advanced-level customer feedback generator for the Sari-Sari Store Simulator.

Generates synthetic monthly customer reviews for products sold during the month.
Feedback is grounded in the actual product list from inventory.csv and reflects
realistic Philippine sari-sari store customer sentiment patterns.

Each feedback record contains:
    feedback_id   : Unique identifier (e.g. FB-202201-00001)
    feedback_date : Date the feedback was submitted
    product_id    : Product being reviewed
    product_name  : Product name (joined from inventory)
    category      : Product category
    rating        : Numeric rating 1.0–5.0
    comment       : Short text comment
    sentiment     : "positive", "neutral", or "negative"

Sentiment rules:
    rating >= 4.0 → positive
    rating == 3.0 → neutral
    rating <= 2.0 → negative

Feedback volume: approximately 2–5 feedback entries per product per month,
weighted by transaction volume (higher-selling products get more reviews).

Output files:
    data/processed/advanced/year_YYYY/month_MM/customer_feedback.csv
    data/processed/advanced/advanced_customer_feedback.csv (aggregated)
    SQLite table: advanced_customer_feedback
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date
import calendar

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
# Comment banks by sentiment and category
# Realistic Philippine sari-sari store customer voice
# ---------------------------------------------------------------------------

POSITIVE_COMMENTS = {
    "Beverage": [
        "Masarap at fresh, sulit ang presyo.",
        "Lagi akong bumabalik dito para dito.",
        "Malamig at masustansya, paborito ng pamilya.",
        "Good value for money, lagi may stock.",
        "Laging fresh, hindi mahal.",
        "Perfect para sa mainit na panahon.",
        "Masarap, okay ang presyo.",
    ],
    "Food": [
        "Masarap at mabilis lutuin, sulit.",
        "Laging may stock, maganda ang kalidad.",
        "Perfect ulam, hindi mahal.",
        "Paborito ng mga bata sa bahay.",
        "Fresh lagi, magandang presyo.",
        "Okay ang lasa, abot-kaya.",
        "Sulit na pagkain para sa pamilya.",
    ],
    "Snack": [
        "Masarap, hindi mahal.",
        "Paborito ng mga bata.",
        "Laging gusto ng pamilya.",
        "Magandang meryenda, sulit.",
        "Okay ang lasa at presyo.",
        "Masustansya at masarap.",
    ],
    "Personal Care": [
        "Okay ang gamit, sulit ang sachet.",
        "Maganda ang kalidad, hindi mahal.",
        "Laging may stock, maganda ang presyo.",
        "Sulit, okay ang bango.",
        "Maganda ang linis, abot-kaya.",
    ],
    "Household": [
        "Epektibo at hindi mahal.",
        "Okay ang gamit, sulit ang sachet.",
        "Maganda ang linis, abot-kaya ang presyo.",
        "Laging may stock, sulit.",
        "Okay ang kalidad para sa presyo.",
    ],
}

NEUTRAL_COMMENTS = {
    "Beverage": [
        "Okay naman, pwede na.",
        "Hindi masama, normal lang.",
        "Katamtaman ang lasa.",
        "Pwede na para sa araw-araw.",
        "Okay lang, walang espesyal.",
    ],
    "Food": [
        "Okay naman, karaniwang ulam.",
        "Pwede na, subok na produkto.",
        "Normal lang, walang reklamo.",
        "Katamtaman, okay para sa presyo.",
        "Hindi masama, okay na rin.",
    ],
    "Snack": [
        "Okay naman, pwede na.",
        "Normal lang, walang special.",
        "Katamtaman ang lasa.",
        "Okay rin, puwede na.",
        "Hindi masama, normal lang.",
    ],
    "Personal Care": [
        "Okay naman, gumagana.",
        "Normal lang, walang reklamo.",
        "Katamtaman, okay para sa presyo.",
        "Pwede na, ginagamit namin.",
    ],
    "Household": [
        "Okay naman, gumagana.",
        "Katamtaman, pwede na.",
        "Normal lang, okay na.",
        "Gumagana naman, okay.",
    ],
}

NEGATIVE_COMMENTS = {
    "Beverage": [
        "Medyo mahal na para sa laki.",
        "Hindi masyadong masarap ngayon.",
        "Medyo mabilis maubusan dito.",
        "Sana magbago ng presyo.",
        "Medyo mahal na ngayon.",
    ],
    "Food": [
        "Medyo mahal na para sa dami.",
        "Hindi masyadong masarap ngayon.",
        "Dapat mas malaki para sa presyo.",
        "Medyo maalat para sa akin.",
        "Sana mas mura pa.",
    ],
    "Snack": [
        "Medyo mahal na para sa laki.",
        "Hindi masyadong masarap ngayon.",
        "Mabilis maubusan dito.",
        "Sana mas malaki ang pack.",
        "Medyo maalat, sana baguhin.",
    ],
    "Personal Care": [
        "Medyo mahal na ang sachet.",
        "Hindi masyadong epektibo.",
        "Sana mas malaki ang sachet.",
        "Hindi ako satisfied sa kalidad.",
    ],
    "Household": [
        "Medyo mahal na ang sachet.",
        "Hindi masyadong mabisa.",
        "Sana mas maraming laman.",
        "Hindi ko gusto ang bagong amoy.",
    ],
}

# Default fallbacks for any unknown category
DEFAULT_POSITIVE = ["Maganda ang produkto, sulit.", "Okay ang kalidad, lagi akong bumabalik."]
DEFAULT_NEUTRAL  = ["Okay naman, pwede na.", "Normal lang, walang reklamo."]
DEFAULT_NEGATIVE = ["Medyo mahal na.", "Hindi ako satisfied."]

# Rating distributions by sentiment
# (mean, std) for each sentiment label
RATING_PARAMS = {
    "positive": (4.5, 0.4),
    "neutral":  (3.0, 0.3),
    "negative": (1.8, 0.5),
}

# Probability of each sentiment being assigned per feedback entry
SENTIMENT_WEIGHTS = {
    "positive": 0.65,
    "neutral":  0.20,
    "negative": 0.15,
}

# Approximate number of feedback entries per product per month
FEEDBACK_PER_PRODUCT_MIN = 2
FEEDBACK_PER_PRODUCT_MAX = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentiment_from_rating(rating: float) -> str:
    """
    Derive a sentiment label from a numeric rating.

    Rules:
        rating >= 4.0 → positive
        rating >= 3.0 → neutral
        rating <  3.0 → negative
    """
    if rating >= 4.0:
        return "positive"
    elif rating >= 3.0:
        return "neutral"
    else:
        return "negative"


def _pick_comment(
    sentiment: str,
    category: str,
    rng: np.random.Generator,
) -> str:
    """
    Pick a random comment string matching the given sentiment and category.

    Parameters
    ----------
    sentiment : str
        "positive", "neutral", or "negative".
    category : str
        Product category string.
    rng : np.random.Generator
        Random generator for reproducibility.

    Returns
    -------
    str
        A comment string.
    """
    banks = {
        "positive": POSITIVE_COMMENTS,
        "neutral":  NEUTRAL_COMMENTS,
        "negative": NEGATIVE_COMMENTS,
    }
    bank = banks.get(sentiment, POSITIVE_COMMENTS)
    options = bank.get(category, DEFAULT_POSITIVE)
    idx = int(rng.integers(0, len(options)))
    return options[idx]


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_monthly_feedback(
    inventory: pd.DataFrame,
    year: int,
    month: int,
    monthly_transactions: pd.DataFrame = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate customer feedback for one month.

    Products with higher transaction volume in the month receive more
    feedback entries. If monthly_transactions is not provided, all products
    receive a flat number of feedback entries.

    Parameters
    ----------
    inventory : pd.DataFrame
        Validated inventory DataFrame from load_inventory().
    year : int
        Year of the feedback period.
    month : int
        Month of the feedback period (1-12).
    monthly_transactions : pd.DataFrame, optional
        Transactions for this month (used to weight feedback volume).
        If None, all products get equal feedback volume.
    random_seed : int
        Base seed offset by year and month.

    Returns
    -------
    pd.DataFrame
        Monthly feedback with columns:
        feedback_id, feedback_date, product_id, product_name,
        category, rating, comment, sentiment
    """
    rng = np.random.default_rng(random_seed + year * 100 + month)
    days_in_month = calendar.monthrange(year, month)[1]

    # Compute feedback count per product based on transaction volume
    if monthly_transactions is not None and not monthly_transactions.empty:
        units_by_product = (
            monthly_transactions.groupby("product_id")["quantity_sold"]
            .sum()
            .reset_index(name="units_sold")
        )
        max_units = units_by_product["units_sold"].max()
        units_by_product["feedback_count"] = (
            units_by_product["units_sold"] / max_units
            * (FEEDBACK_PER_PRODUCT_MAX - FEEDBACK_PER_PRODUCT_MIN)
            + FEEDBACK_PER_PRODUCT_MIN
        ).apply(lambda x: max(FEEDBACK_PER_PRODUCT_MIN, int(round(x))))
    else:
        units_by_product = pd.DataFrame({
            "product_id":     inventory["product_id"],
            "units_sold":     FEEDBACK_PER_PRODUCT_MIN,
            "feedback_count": FEEDBACK_PER_PRODUCT_MIN,
        })

    product_lookup = inventory.set_index("product_id")[
        ["product_name", "category"]
    ].to_dict("index")

    sentiment_labels = list(SENTIMENT_WEIGHTS.keys())
    sentiment_probs  = list(SENTIMENT_WEIGHTS.values())

    rows = []
    counter = 1

    for _, row in units_by_product.iterrows():
        product_id = row["product_id"]
        feedback_count = int(row["feedback_count"])

        if product_id not in product_lookup:
            continue

        product_name = product_lookup[product_id]["product_name"]
        category     = product_lookup[product_id]["category"]

        for _ in range(feedback_count):
            # Random date within the month
            feedback_day = int(rng.integers(1, days_in_month + 1))
            feedback_date = date(year, month, feedback_day)

            # Pick sentiment
            sentiment_idx = rng.choice(
                len(sentiment_labels), p=sentiment_probs
            )
            sentiment = sentiment_labels[sentiment_idx]

            # Generate rating from distribution
            mean, std = RATING_PARAMS[sentiment]
            raw_rating = rng.normal(mean, std)
            rating = round(float(np.clip(raw_rating, 1.0, 5.0)), 1)

            # Recalculate sentiment from actual rating (keeps data consistent)
            sentiment = _sentiment_from_rating(rating)

            comment = _pick_comment(sentiment, category, rng)

            rows.append({
                "feedback_id":   f"FB-{year}{month:02d}-{counter:05d}",
                "feedback_date": feedback_date,
                "product_id":    product_id,
                "product_name":  product_name,
                "category":      category,
                "rating":        rating,
                "comment":       comment,
                "sentiment":     sentiment,
            })
            counter += 1

    feedback = pd.DataFrame(rows, columns=[
        "feedback_id",
        "feedback_date",
        "product_id",
        "product_name",
        "category",
        "rating",
        "comment",
        "sentiment",
    ])

    feedback["feedback_date"] = pd.to_datetime(feedback["feedback_date"])

    return feedback


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_monthly_feedback_csv(
    feedback: pd.DataFrame,
    year: int,
    month: int,
    output_base_folder: str | Path,
) -> Path:
    """
    Save monthly feedback to its year/month folder.

    Parameters
    ----------
    feedback : pd.DataFrame
        Monthly feedback DataFrame.
    year : int
        Year of the feedback period.
    month : int
        Month of the feedback period (1-12).
    output_base_folder : str or Path
        Root advanced output folder.

    Returns
    -------
    Path
        Path of the saved CSV.
    """
    folder = (
        Path(output_base_folder) / f"year_{year}" / f"month_{month:02d}"
    )
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / "customer_feedback.csv"
    feedback.to_csv(path, index=False)
    return path


def save_five_year_feedback_csv(
    all_feedback: pd.DataFrame,
    output_base_folder: str | Path,
) -> Path:
    """
    Save aggregated five-year feedback to the advanced root folder.

    Parameters
    ----------
    all_feedback : pd.DataFrame
        Combined feedback for all 60 months.
    output_base_folder : str or Path
        Root advanced output folder.

    Returns
    -------
    Path
        Path of the saved CSV.
    """
    folder = Path(output_base_folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / "advanced_customer_feedback.csv"
    all_feedback.to_csv(path, index=False)
    return path


def save_feedback_to_sqlite(
    all_feedback: pd.DataFrame,
    sqlite_db_path: str | Path,
) -> None:
    """
    Save the aggregated feedback DataFrame to SQLite.

    Table name: advanced_customer_feedback

    Parameters
    ----------
    all_feedback : pd.DataFrame
        Combined feedback for all 60 months.
    sqlite_db_path : str or Path
        Path to the SQLite database.
    """
    sqlite_db_path = Path(sqlite_db_path)
    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{sqlite_db_path}")

    all_feedback.to_sql(
        name="advanced_customer_feedback",
        con=engine,
        if_exists="replace",
        index=False,
    )

    print(f"SQLite table saved    : advanced_customer_feedback → {sqlite_db_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_feedback_generator(
    inventory_csv_path: str | Path = "data/raw/inventory.csv",
    output_base_folder: str | Path = "data/processed/advanced",
    sqlite_db_path: str | Path = "src/database/sari_sari_store.db",
    all_transactions: pd.DataFrame = None,
    random_seed: int = 42,
    save_csv: bool = True,
    save_sqlite: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full customer feedback generation workflow.

    Generates monthly feedback for every year and month in the simulation
    period, saves monthly CSVs to year/month folders, saves an aggregated
    CSV to the advanced root, and saves to SQLite.

    Parameters
    ----------
    inventory_csv_path : str or Path
        Path to data/raw/inventory.csv.
    output_base_folder : str or Path
        Root advanced output folder.
    sqlite_db_path : str or Path
        Path to the SQLite database.
    all_transactions : pd.DataFrame, optional
        Aggregated five-year transactions from run_five_year_generator().
        Used to weight feedback volume per product per month.
        If None, all products receive equal feedback volume.
    random_seed : int
        Base seed for reproducibility.
    save_csv : bool
        Whether to save CSV files.
    save_sqlite : bool
        Whether to save to SQLite.
    verbose : bool
        Whether to print progress to the console.

    Returns
    -------
    pd.DataFrame
        Aggregated five-year customer feedback DataFrame.
    """
    inventory = load_inventory(inventory_csv_path)

    if verbose:
        print("\n" + "=" * 72)
        print("ADVANCED GOAL: CUSTOMER FEEDBACK GENERATOR")
        print("=" * 72)
        print(f"Products        : {len(inventory)}")
        print(f"Period          : {START_YEAR}–{END_YEAR} (60 months)")
        print("-" * 72)

    all_feedback_frames = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):

            # Filter transactions to this month if provided
            if all_transactions is not None and not all_transactions.empty:
                mask = (
                    pd.to_datetime(all_transactions["transaction_date"]).dt.year == year
                ) & (
                    pd.to_datetime(all_transactions["transaction_date"]).dt.month == month
                )
                monthly_txns = all_transactions[mask]
            else:
                monthly_txns = None

            feedback = generate_monthly_feedback(
                inventory=inventory,
                year=year,
                month=month,
                monthly_transactions=monthly_txns,
                random_seed=random_seed,
            )

            if save_csv:
                save_monthly_feedback_csv(
                    feedback=feedback,
                    year=year,
                    month=month,
                    output_base_folder=output_base_folder,
                )

            all_feedback_frames.append(feedback)

            if verbose:
                pos = (feedback["sentiment"] == "positive").sum()
                neu = (feedback["sentiment"] == "neutral").sum()
                neg = (feedback["sentiment"] == "negative").sum()
                avg = feedback["rating"].mean()
                print(
                    f"  {year}-{month:02d}  |  "
                    f"{len(feedback):>4} reviews  |  "
                    f"avg {avg:.2f}★  |  "
                    f"+{pos} ={neu} -{neg}"
                )

    all_feedback = pd.concat(all_feedback_frames, ignore_index=True)

    if save_csv:
        path = save_five_year_feedback_csv(all_feedback, output_base_folder)
        print(f"\nAggregated CSV saved  : {path}")

    if save_sqlite:
        save_feedback_to_sqlite(all_feedback, sqlite_db_path)

    if verbose:
        print("-" * 72)
        print(f"Total feedback records : {len(all_feedback):,}")
        overall_avg = all_feedback["rating"].mean()
        print(f"Overall avg rating     : {overall_avg:.2f}★")
        print("Feedback generation complete.")
        print("=" * 72)

    return all_feedback


if __name__ == "__main__":
    run_feedback_generator()
