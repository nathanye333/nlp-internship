import pandas as pd

from text_cleaning import TextCleaner


def main():
    cleaner = TextCleaner()

    # Load week 1 sample
    df = pd.read_csv("data/processed/listing_sample.csv")

    # Profiling before cleaning
    before_profile = cleaner.profile_column(df, "remarks")

    # Apply cleaning
    df["remarks_clean"] = df["remarks"].apply(cleaner.clean_text)

    # Profiling after cleaning
    after_profile = cleaner.profile_column(df, "remarks_clean")

    # Save cleaned dataset
    df.to_csv("data/processed/listing_sample_cleaned.csv", index=False)

    # Save a small before/after sample for inspection
    sample = df[["remarks", "remarks_clean"]].head(50)
    sample.to_csv("data/processed/listing_sample_before_after.csv", index=False)

    # Simple text report to stdout
    print("=== Profiling BEFORE cleaning (remarks) ===")
    print(before_profile)
    print("\n=== Profiling AFTER cleaning (remarks_clean) ===")
    print(after_profile)


if __name__ == "__main__":
    main()

