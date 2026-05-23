"""
Dataset preparation script for the Smart Agriculture Decision Support System.

Generates a realistic synthetic agricultural dataset where crop labels are
assigned based on agronomic rules (not randomly), producing meaningful
decision boundaries for the ML models.

Usage:
    python prepare_dataset.py
"""
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Agronomic crop profiles
# Each entry: (crop_name, N_mean, P_mean, K_mean, temp_mean, humidity_mean,
#              ph_mean, rainfall_mean, yield_base)
# ---------------------------------------------------------------------------
CROP_PROFILES = [
    # name,          N,   P,   K,   T,    H,   pH,  Rain, Yield_base
    ('rice',         80,  40,  40,  23,   82,  6.5,  236,  4200),
    ('maize',        78,  48,  20,  22,   65,  6.2,  103,  3800),
    ('chickpea',     40,  67,  79,  18,   16,  7.2,   80,  1800),
    ('kidneybeans',  20,  67,  20,  20,   21,  5.7,  105,  2000),
    ('pigeonpeas',   20,  67,  20,  27,   48,  5.8,  149,  1900),
    ('mothbeans',    21,  48,  20,  28,   53,  6.9,   50,  1600),
    ('mungbeans',    20,  47,  20,  28,   85,  6.7,   48,  1700),
    ('blackgram',    40,  67,  19,  30,   65,  7.0,   68,  1800),
    ('lentil',       18,  68,  19,  24,   64,  6.9,   46,  1500),
    ('pomegranate',  18,  18,  40,  21,   90,  6.0,  107,  5000),
    ('banana',       100, 82,  50,  27,   80,  6.0,  105,  6000),
    ('mango',        20,  27,  30,  31,   50,  5.8,   95,  4500),
    ('grapes',       23,  132, 200, 24,   81,  6.0,   70,  5500),
    ('watermelon',   99,  17,  50,  25,   85,  6.5,   50,  7000),
    ('muskmelon',    100, 17,  50,  28,   92,  6.5,   25,  6500),
    ('apple',        21,  134, 199, 22,   92,  5.8,  113,  4800),
    ('orange',       20,  16,  10,  23,   92,  7.0,  110,  4200),
    ('papaya',       50,  59,  50,  34,   92,  6.7,  143,  5800),
    ('coconut',      22,  16,  30,  27,   94,  5.9,  176,  5200),
    ('cotton',       118, 46,  20,  24,   80,  6.8,   80,  2500),
    ('jute',         78,  46,  40,  25,   80,  6.7,  175,  2200),
    ('sugarcane',    20,  20,  20,  27,   82,  6.5,  185,  7500),
]

# Standard deviations (noise) for each feature — kept tight so crops are separable
NOISE = dict(N=8, P=8, K=8, temperature=2, humidity=8, ph=0.3, rainfall=20)


def create_realistic_dataset(n_samples: int = 2200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset with agronomically realistic crop profiles.

    Each crop gets an equal share of samples. Feature values are drawn from
    a normal distribution centred on the crop's agronomic profile.
    """
    rng = np.random.default_rng(seed)
    n_crops = len(CROP_PROFILES)
    per_crop = n_samples // n_crops

    rows = []
    for (name, N, P, K, T, H, pH, Rain, yield_base) in CROP_PROFILES:
        n = per_crop
        N_vals    = rng.normal(N,    NOISE['N'],           n).clip(0, 140)
        P_vals    = rng.normal(P,    NOISE['P'],           n).clip(5, 145)
        K_vals    = rng.normal(K,    NOISE['K'],           n).clip(5, 205)
        T_vals    = rng.normal(T,    NOISE['temperature'], n).clip(8.8, 43.7)
        H_vals    = rng.normal(H,    NOISE['humidity'],    n).clip(14, 99)
        pH_vals   = rng.normal(pH,   NOISE['ph'],          n).clip(3.5, 9.9)
        Rain_vals = rng.normal(Rain, NOISE['rainfall'],    n).clip(20, 298)

        # Yield: crop-specific base + feature deviations from crop mean + noise
        # The yield correlates with how close the soil/climate is to the crop's optimum
        # This gives the regression a learnable signal from the 7 input features.
        feature_deviation = (
            (N_vals - N) ** 2 / (NOISE['N'] ** 2)
            + (P_vals - P) ** 2 / (NOISE['P'] ** 2)
            + (K_vals - K) ** 2 / (NOISE['K'] ** 2)
            + (T_vals - T) ** 2 / (NOISE['temperature'] ** 2)
            + (pH_vals - pH) ** 2 / (NOISE['ph'] ** 2)
        )
        # Yield decreases as features deviate from the crop's optimum
        yield_vals = (
            yield_base * (1.0 - 0.04 * feature_deviation)
            + rng.normal(0, yield_base * 0.03, n)
        ).clip(500)

        for i in range(n):
            rows.append({
                'N':           round(float(N_vals[i]),    1),
                'P':           round(float(P_vals[i]),    1),
                'K':           round(float(K_vals[i]),    1),
                'temperature': round(float(T_vals[i]),    2),
                'humidity':    round(float(H_vals[i]),    1),
                'ph':          round(float(pH_vals[i]),   2),
                'rainfall':    round(float(Rain_vals[i]), 1),
                'label':       name,
                'yield':       round(float(yield_vals[i]), 2),
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def prepare_dataset():
    """Create the data/ directory and write agricultural_data.csv if absent."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / 'agricultural_data.csv'

    if not csv_path.exists():
        print("Creating realistic sample dataset…")
        df = create_realistic_dataset()
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}  (shape: {df.shape})")
        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))
        print("\nClass distribution:")
        print(df['label'].value_counts().to_string())
    else:
        print(f"Dataset already exists at {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset shape: {df.shape}")

    return csv_path


def create_data_dictionary(df: pd.DataFrame) -> str:
    """Generate a Markdown data dictionary for the dataset."""
    doc = """# Data Dictionary — Agricultural Dataset

## Overview
| Property | Value |
|---|---|
| Total Records | {n} |
| Input Features | 7 |
| Target Variables | 2 (label, yield) |
| Crop Classes | {n_crops} |

## Feature Specifications

| # | Feature | Unit | Type | Min | Max | Mean | Std | Description |
|---|---|---|---|---|---|---|---|---|
| 1 | N (Nitrogen) | mg/kg | Float | 0 | 140 | {N_mean:.1f} | {N_std:.1f} | Soil nitrogen — primary macronutrient |
| 2 | P (Phosphorus) | mg/kg | Float | 5 | 145 | {P_mean:.1f} | {P_std:.1f} | Soil phosphorus — energy transfer |
| 3 | K (Potassium) | mg/kg | Float | 5 | 205 | {K_mean:.1f} | {K_std:.1f} | Soil potassium — water regulation |
| 4 | temperature | °C | Float | 8.8 | 43.7 | {T_mean:.1f} | {T_std:.1f} | Growing-season average temperature |
| 5 | humidity | % | Float | 14 | 99 | {H_mean:.1f} | {H_std:.1f} | Relative humidity |
| 6 | ph | — | Float | 3.5 | 9.9 | {pH_mean:.2f} | {pH_std:.2f} | Soil pH (acidity/alkalinity) |
| 7 | rainfall | mm | Float | 20 | 298 | {R_mean:.1f} | {R_std:.1f} | Total seasonal rainfall |
| 8 | label | — | Categorical | — | — | {n_crops} classes | — | Recommended crop (classification target) |
| 9 | yield | kg/ha | Float | 500 | — | {Y_mean:.0f} | {Y_std:.0f} | Crop yield (regression target) |

## Crop Classes ({n_crops} total)
{crop_list}

## Preprocessing Pipeline
```
Raw Data ({n} records)
    ↓ Handle missing values (median / mode imputation)
    ↓ Remove outliers (IQR ×1.5)
    ↓ Encode categorical features (LabelEncoder on 'label')
    ↓ Standardise features (StandardScaler — features only, not targets)
    ↓ Train/test split 80/20 (stratified on label)
```

## Citation
Synthetic dataset generated from agronomic profiles derived from:
Ingle, A. (2018). Crop Recommendation Dataset. Kaggle.
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

---
*Generated: {date}*
""".format(
        n=len(df),
        n_crops=df['label'].nunique(),
        N_mean=df['N'].mean(), N_std=df['N'].std(),
        P_mean=df['P'].mean(), P_std=df['P'].std(),
        K_mean=df['K'].mean(), K_std=df['K'].std(),
        T_mean=df['temperature'].mean(), T_std=df['temperature'].std(),
        H_mean=df['humidity'].mean(), H_std=df['humidity'].std(),
        pH_mean=df['ph'].mean(), pH_std=df['ph'].std(),
        R_mean=df['rainfall'].mean(), R_std=df['rainfall'].std(),
        Y_mean=df['yield'].mean(), Y_std=df['yield'].std(),
        crop_list='\n'.join(
            f'{i+1}. {c}' for i, c in enumerate(sorted(df['label'].unique()))
        ),
        date=pd.Timestamp.now().strftime('%Y-%m-%d'),
    )
    return doc


if __name__ == "__main__":
    csv_path = prepare_dataset()

    df = pd.read_csv(csv_path)
    data_dict = create_data_dictionary(df)

    dict_path = Path('data') / 'data_dictionary.md'
    dict_path.write_text(data_dict, encoding='utf-8')
    print(f"\nData dictionary saved to {dict_path}")

    print("\n✓ Dataset preparation complete!")
    print("Next steps:")
    print("  1. python train_models.py")
    print("  2. python src/gui.py")
