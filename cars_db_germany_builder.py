#!/usr/bin/env python3
# cars_db_germany_builder.py

import pandas as pd
from pathlib import Path
import numpy as np
import re

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/cars_db_germany.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Canonical columns we will output:
# id, make, model, year, price_eur, mileage_km, fuel_type, power_ps,
# segment, transmission, vehicle_type, listing_url, source, raw_file

def try_read_csv(p: Path):
    """Try reading CSV with utf-8, then latin1 fallback."""
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(p, encoding="latin1", low_memory=False)
        except Exception as e:
            print(f"❌ Failed to read {p}: {e}")
            return None

def pick_column(df, candidates):
    """Return the first column in df that matches any name in candidates."""
    cols_lower = {col.lower(): col for col in df.columns}
    for name in candidates:
        name_lower = name.lower()
        if name_lower in cols_lower:
            return df[cols_lower[name_lower]]
    # fallback: substring-based fuzzy search
    for name in candidates:
        for col in df.columns:
            if name.lower() in col.lower():
                return df[col]
    return pd.Series([None] * len(df))

def normalize_one(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize one raw dataset into the canonical schema."""
    df_cols_lower = {c.lower(): c for c in df.columns}

    def col(*names):
        return pick_column(df, names)

    canon = pd.DataFrame()
    canon["make"]         = col("make", "manufacturer", "marke", "hersteller", "brand")
    canon["model"]        = col("model", "modell", "title", "car_model", "bezeichnung")
    canon["year"]         = col("year", "baujahr", "erstzulassung", "manufacture_year")
    canon["price_eur"]    = col("price", "preis", "price_eur", "eur", "priceEuro")
    canon["mileage_km"]   = col("mileage", "km", "kilometer", "laufleistung", "odometer")
    canon["fuel_type"]    = col("fuel", "kraftstoff", "fuel_type", "antrieb")
    canon["power_ps"]     = col("powerps", "ps", "leistung", "horsepower", "power")
    canon["transmission"] = col("transmission", "getriebe", "gearbox")
    canon["vehicle_type"] = col("vehicle_type", "body_type", "typ", "category", "kategorie")
    canon["listing_url"]  = col("url", "link", "listing", "listing_url")

    # Track source file name
    canon["source"] = source_name

    # --------------------------------------------------------------
    # Infer segment (German categories)
    # --------------------------------------------------------------
    def infer_segment(row):
        vt = str(row.get("vehicle_type") or "").lower()
        model = str(row.get("model") or "").lower()

        if any(x in vt for x in ["suv", "geländewagen", "offroad"]):
            return "suv"
        if any(x in vt for x in ["kombi", "estate", "wagon"]):
            return "kombi"
        if any(x in vt for x in ["limousine", "sedan"]):
            return "limousine"
        if any(x in model for x in ["golf", "polo", "astra", "fiesta", "clio", "corsa"]):
            return "kleinwagen"
        if any(x in model for x in ["transporter", "sprinter", "vivaro"]):
            return "transporter"
        return "other"

    # --------------------------------------------------------------
    # Clean/conversion numeric columns
    # --------------------------------------------------------------
    for c in ["year", "price_eur", "mileage_km", "power_ps"]:
        if c in canon.columns:
            canon[c] = (
                canon[c]
                .astype(str)
                .str.replace(r"[^\d.-]", "", regex=True)
            )
            canon[c] = pd.to_numeric(canon[c], errors="coerce")

    canon["segment"] = canon.apply(infer_segment, axis=1)

    # Drop rows with missing make/model
    canon = canon.dropna(subset=["make", "model"]).reset_index(drop=True)

    # Clean strings
    for colname in ["make", "model", "fuel_type", "transmission", "vehicle_type", "listing_url"]:
        if colname in canon.columns:
            canon[colname] = (
                canon[colname]
                .astype(str)
                .str.strip()
                .replace({"nan": None})
            )

    return canon[
        [
            "make", "model", "year", "price_eur", "mileage_km",
            "fuel_type", "power_ps", "segment",
            "transmission", "vehicle_type", "listing_url",
            "source"
        ]
    ]

def build_db():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        print("⚠ No CSV files found in data/raw/. Add German datasets and re-run.")
        return

    frames = []
    for f in files:
        print("📥 Reading:", f)
        df = try_read_csv(f)
        if df is None:
            continue

        normalized = normalize_one(df, f.name)
        normalized["raw_file"] = f.name
        frames.append(normalized)

    if not frames:
        print("❌ No usable data found.")
        return

    big = pd.concat(frames, ignore_index=True)

    # --------------------------------------------------------------
    # Remove unrealistic data
    # --------------------------------------------------------------
    big = big[
        (big["price_eur"].isna()) |
        ((big["price_eur"] > 200) & (big["price_eur"] < 500000))
    ]

    big = big[
        (big["mileage_km"].isna()) |
        ((big["mileage_km"] >= 0) & (big["mileage_km"] < 1_000_000))
    ]

    # --------------------------------------------------------------
    # Deduplicate approx
    # --------------------------------------------------------------
    big["dup_key"] = (
        big.make.str.lower().fillna("") + "|" +
        big.model.str.lower().fillna("") + "|" +
        big.year.fillna(0).astype(int).astype(str) + "|" +
        big.price_eur.fillna(0).astype(int).astype(str) + "|" +
        big.mileage_km.fillna(0).astype(int).astype(str)
    )

    before = len(big)
    big = big.drop_duplicates(subset=["dup_key"]).drop(columns=["dup_key"])
    after = len(big)

    print(f"🧹 Deduplication: before={before:,} → after={after:,}")

    # --------------------------------------------------------------
    # Add ID and reorder columns
    # --------------------------------------------------------------
    big = big.reset_index(drop=True)
    big["id"] = big.index + 1

    cols = [
        "id", "make", "model", "year", "price_eur", "mileage_km",
        "fuel_type", "power_ps", "segment", "transmission",
        "vehicle_type", "listing_url", "source", "raw_file"
    ]
    big = big[cols]

    # --------------------------------------------------------------
    # Save final cleaned dataset
    # --------------------------------------------------------------
    big.to_csv(OUT_FILE, index=False)

    print(f"✅ Saved cleaned dataset → {OUT_FILE} ({len(big):,} rows)")
    print("Top 15 makes:")
    print(big["make"].value_counts().head(15))

    return big


if __name__ == "__main__":
    build_db()
