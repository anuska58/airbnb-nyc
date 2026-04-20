import csv
from pathlib import Path
import numpy as np
import pandas as pd


INPUT_FILE = Path("Airbnb_Open_Data.csv")
OUTPUT_FILE = Path("airbnb_cleaned.csv")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase snake_case."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_", regex=False)
    )
    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalize empty-like strings to NaN."""
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(
                {
                    "nan": np.nan,
                    "None": np.nan,
                    "": np.nan,
                }
            )

    return df


def clean_money_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert currency-like columns from strings to numeric."""
    df = df.copy()
    money_cols = ["price", "service_fee"]

    for col in money_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[$,]", "", regex=True)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert selected columns to numeric types."""
    df = df.copy()
    numeric_cols = [
        "lat",
        "long",
        "construction_year",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "review_rate_number",
        "calculated_host_listings_count",
        "availability_365",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date columns to datetime."""
    df = df.copy()

    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    return df


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Fix obvious category inconsistencies."""
    df = df.copy()

    if "neighbourhood_group" in df.columns:
        df["neighbourhood_group"] = df["neighbourhood_group"].replace(
            {
                "manhatan": "Manhattan",
                "brookln": "Brooklyn",
            }
        )

    if "instant_bookable" in df.columns:
        df["instant_bookable"] = df["instant_bookable"].replace(
            {
                "true": "true",
                "false": "false",
                "True": "true",
                "False": "false",
            }
        )

    if "host_identity_verified" in df.columns:
        df["host_identity_verified"] = df["host_identity_verified"].replace(
            {
                "verified": "verified",
                "unconfirmed": "unconfirmed",
            }
        )

    return df


def fix_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible values with NaN."""
    df = df.copy()

    if "minimum_nights" in df.columns:
        df.loc[df["minimum_nights"] < 1, "minimum_nights"] = np.nan

    if "availability_365" in df.columns:
        invalid_availability = (
            (df["availability_365"] < 0) | (df["availability_365"] > 365)
        )
        df.loc[invalid_availability, "availability_365"] = np.nan

    if "review_rate_number" in df.columns:
        invalid_review_rate = (
            (df["review_rate_number"] < 1) | (df["review_rate_number"] > 5)
        )
        df.loc[invalid_review_rate, "review_rate_number"] = np.nan

    if "construction_year" in df.columns:
        invalid_year = (
            (df["construction_year"] < 1900) | (df["construction_year"] > 2026)
        )
        df.loc[invalid_year, "construction_year"] = np.nan

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing targets and fill selected fields."""
    df = df.copy()

    if "price" in df.columns:
        df = df.dropna(subset=["price"])

    categorical_fill_cols = [
        "neighbourhood_group",
        "room_type",
        "cancellation_policy",
        "host_identity_verified",
        "instant_bookable",
    ]

    for col in categorical_fill_cols:
        if col in df.columns and df[col].isna().sum() > 0:
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])

    numeric_fill_cols = [
        "service_fee",
        "minimum_nights",
        "reviews_per_month",
        "review_rate_number",
        "calculated_host_listings_count",
        "availability_365",
    ]

    for col in numeric_fill_cols:
        if col in df.columns and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create helpful derived features."""
    df = df.copy()

    if "last_review" in df.columns:
        df["review_year"] = df["last_review"].dt.year
        df["review_month"] = df["last_review"].dt.month

    if "calculated_host_listings_count" in df.columns:
        df["host_type"] = np.where(
            df["calculated_host_listings_count"] >= 3,
            "professional",
            "casual",
        )

    if "price" in df.columns:
        try:
            df["price_category"] = pd.qcut(
                df["price"],
                q=3,
                labels=["Low", "Medium", "High"],
            )
        except ValueError:
            pass

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and duplicate IDs."""
    df = df.copy()
    df = df.drop_duplicates()

    if "id" in df.columns:
        df = df.drop_duplicates(subset="id", keep="first")

    return df


def clean_airbnb_data(input_file: Path, output_file: Path) -> pd.DataFrame:
    """Load, clean, and save the Airbnb dataset."""
    df = pd.read_csv(input_file, engine="python")

    print(f"Raw shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    df = standardize_column_names(df)
    df = remove_duplicates(df)
    df = clean_text_columns(df)
    df = clean_money_columns(df)
    df = convert_numeric_columns(df)
    df = convert_date_columns(df)
    df = standardize_categories(df)
    df = fix_impossible_values(df)
    df = handle_missing_values(df)
    df = add_engineered_features(df)

    if "minimum_nights" in df.columns:
        df.loc[df["minimum_nights"] > 365, "minimum_nights"] = np.nan
        df["minimum_nights"] = df["minimum_nights"].fillna(df["minimum_nights"].median())

    if "license" in df.columns:
        df = df.drop(columns=["license"])
    
    if "house_rules" in df.columns:
        df = df.drop(columns=["house_rules"])

    df.to_csv(
        output_file,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    print(f"Cleaned shape: {df.shape}")
    print(f"Saved cleaned file as: {output_file}")
    print("\nTop missing values:")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    return df


def validate_cleaned_data(file_path: Path) -> None:
    """Run simple validation checks on the cleaned dataset."""
    df = pd.read_csv(file_path)

    print("\nValidation checks")
    print("-" * 40)
    print(f"Shape: {df.shape}")

    if "id" in df.columns:
        print(f"Duplicate IDs: {df['id'].duplicated().sum()}")

    if "price" in df.columns:
        print(f"Null prices: {df['price'].isna().sum()}")

    summary_cols = [
        "price",
        "service_fee",
        "minimum_nights",
        "availability_365",
    ]
    existing_cols = [col for col in summary_cols if col in df.columns]

    if existing_cols:
        print("\nNumeric summary:")
        print(df[existing_cols].describe())


if __name__ == "__main__":
    clean_airbnb_data(INPUT_FILE, OUTPUT_FILE)
    validate_cleaned_data(OUTPUT_FILE)