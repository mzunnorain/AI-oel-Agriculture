# Data Dictionary — Agricultural Dataset

## Overview
| Property | Value |
|---|---|
| Total Records | 2200 |
| Input Features | 7 |
| Target Variables | 2 (label, yield) |
| Crop Classes | 22 |

## Feature Specifications

| # | Feature | Unit | Type | Min | Max | Mean | Std | Description |
|---|---|---|---|---|---|---|---|---|
| 1 | N (Nitrogen) | mg/kg | Float | 0 | 140 | 46.7 | 34.5 | Soil nitrogen — primary macronutrient |
| 2 | P (Phosphorus) | mg/kg | Float | 5 | 145 | 52.2 | 33.5 | Soil phosphorus — energy transfer |
| 3 | K (Potassium) | mg/kg | Float | 5 | 205 | 47.6 | 50.5 | Soil potassium — water regulation |
| 4 | temperature | °C | Float | 8.8 | 43.7 | 25.4 | 4.3 | Growing-season average temperature |
| 5 | humidity | % | Float | 14 | 99 | 71.9 | 22.6 | Relative humidity |
| 6 | ph | — | Float | 3.5 | 9.9 | 6.41 | 0.56 | Soil pH (acidity/alkalinity) |
| 7 | rainfall | mm | Float | 20 | 298 | 105.7 | 54.6 | Total seasonal rainfall |
| 8 | label | — | Categorical | — | — | 22 classes | — | Recommended crop (classification target) |
| 9 | yield | kg/ha | Float | 500 | — | 3173 | 1645 | Crop yield (regression target) |

## Crop Classes (22 total)
1. apple
2. banana
3. blackgram
4. chickpea
5. coconut
6. cotton
7. grapes
8. jute
9. kidneybeans
10. lentil
11. maize
12. mango
13. mothbeans
14. mungbeans
15. muskmelon
16. orange
17. papaya
18. pigeonpeas
19. pomegranate
20. rice
21. sugarcane
22. watermelon

## Preprocessing Pipeline
```
Raw Data (2200 records)
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
*Generated: 2026-05-23*
