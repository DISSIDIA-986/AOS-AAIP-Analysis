"""
Extract ATP Data
----------------
Extract Accelerated Tech Pathway data from raw source,
preserving the full stream name before processing.
"""

from pathlib import Path
import pandas as pd
import numpy as np

RAW_DATA_PATH = Path("data/raw/aaip_draws_raw.csv")
OUTPUT_PATH = Path("data/processed/atp_draws_2025.csv")


def extract_atp_data():
    """Extract ATP-specific draws from raw data."""
    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # Rename columns
    df = df.rename(columns={
        "Draw date": "draw_date",
        "Worker stream, pathway, initiative or other focus and selection parameters": "stream_raw",
        "Minimum score of invited candidates": "min_score",
        "Number of invitations": "invitations",
    })

    # Parse dates
    df["draw_date"] = pd.to_datetime(df["draw_date"], format="%B %d, %Y")

    # Filter for 2025 only
    df = df[df["draw_date"].dt.year == 2025].copy()

    # Filter for ATP specifically
    atp_df = df[df["stream_raw"].str.contains("Accelerated Tech Pathway", case=False, na=False)].copy()

    # Clean invitations
    atp_df["invitations"] = pd.to_numeric(atp_df["invitations"], errors="coerce")

    # Clean min_score
    atp_df["min_score"] = atp_df["min_score"].astype(str).str.extract(r"(\d+)")[0]
    atp_df["min_score"] = pd.to_numeric(atp_df["min_score"], errors="coerce")

    # Add stream label
    atp_df["stream"] = "Accelerated Tech Pathway"

    # Sort by date
    atp_df = atp_df.sort_values("draw_date")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atp_df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Extracted {len(atp_df)} ATP draws")
    print(f"📁 Saved to: {OUTPUT_PATH}")
    print(f"\nATP Draws:")
    print(atp_df[["draw_date", "invitations", "min_score"]].to_string(index=False))

    return atp_df


if __name__ == "__main__":
    extract_atp_data()
