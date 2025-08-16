from pathlib import Path
import pandas as pd
from .config import RAW_CSV, FILTERED_CSV, PROCESSED_DIR, ALLOWED_PRODUCTS
from .utils import clean_text

def run_preprocess():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV)
    # Standardize product names a bit
    df['product'] = df['product'].str.strip()
    df['product'] = df['product'].replace({'Money transfers': 'Money transfer'})
    # Keep only allowed products
    df = df[df['product'].isin({p if p != 'Money transfers' else 'Money transfer' for p in ALLOWED_PRODUCTS})].copy()
    # Rename columns if needed
    if 'narrative' not in df.columns:
        # Try common alternatives
        for alt in ['Consumer complaint narrative', 'complaint_what_happened', 'complaint_text']:
            if alt in df.columns:
                df['narrative'] = df[alt]
                break
    # Drop empty narratives
    df['narrative'] = df['narrative'].fillna("").astype(str)
    df = df[df['narrative'].str.strip() != ""]
    # Clean text
    df['narrative_clean'] = df['narrative'].apply(clean_text)
    # Word count
    df['word_count'] = df['narrative_clean'].str.split().apply(len)
    # Save
    df.to_csv(FILTERED_CSV, index=False)
    return df

if __name__ == "__main__":
    run_preprocess()
