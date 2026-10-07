# Automated Insight Generation

A lightweight Streamlit Auto-Analytics Engine for district-level healthcare performance data.

## Features

- CSV upload and required-column validation
- Missing-value report
- Live district/month/indicator filters
- Configurable trend detection
- IQR or Z-score outlier detection
- Pearson correlation detection
- Dynamic human-readable insights
- Low / Medium / High severity
- Severity bar chart
- Correlation heatmap
- Per-district line chart
- Downloadable insights CSV and correlation matrix

## Project structure

```text
automated-insight-generation/
├── app.py
├── insight_engine.py
├── data.csv
├── requirements.txt
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset schema

Required columns:

- month
- district
- anc_coverage
- institutional_delivery
- immunization
- high_risk_cases

## Important limitation

The provided sample dataset contains only 2 months across 6 districts. Pearson correlations on such a small sample are statistically fragile and should be interpreted cautiously. More months/districts are recommended for reliable analysis.

## Assignment mapping

The application implements:
- Data loading and validation
- Trend detection using configurable percentage-change threshold
- IQR/Z-score outlier detection
- Pearson correlation with configurable threshold
- Structured automated insights
- Severity classification
- Streamlit filters and visualizations
