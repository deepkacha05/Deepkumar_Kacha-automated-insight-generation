import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "month",
    "district",
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
]

INDICATORS = [
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
]


def _empty_trends():
    return pd.DataFrame(
        columns=[
            "type", "indicator", "entity", "period",
            "metric", "change", "prev_value", "explanation"
        ]
    )


def detect_trends(df, indicators, threshold=10.0):
    """Detect significant month-over-month changes for each district/indicator."""
    records = []

    if not indicators:
        return _empty_trends()

    work = df.copy()
    work["_month_sort"] = pd.to_datetime(work["month"], errors="coerce")
    work["_month_sort"] = work["_month_sort"].fillna(
        pd.to_datetime(work["month"].astype(str), errors="coerce")
    )

    for indicator in indicators:
        if indicator not in work.columns:
            continue

        temp = work[["district", "month", "_month_sort", indicator]].copy()
        temp[indicator] = pd.to_numeric(temp[indicator], errors="coerce")
        temp = temp.dropna(subset=[indicator]).sort_values(
            ["district", "_month_sort", "month"]
        )

        for district, group in temp.groupby("district", sort=False):
            group = group.reset_index(drop=True)

            for i in range(1, len(group)):
                previous = float(group.loc[i - 1, indicator])
                current = float(group.loc[i, indicator])

                if previous == 0:
                    continue

                change_pct = (current - previous) / abs(previous) * 100.0

                if abs(change_pct) >= float(threshold):
                    direction = "increased" if change_pct > 0 else "dropped"
                    records.append(
                        {
                            "type": "trend",
                            "indicator": indicator,
                            "entity": str(district),
                            "period": str(group.loc[i, "month"]),
                            "metric": current,
                            "change": change_pct,
                            "prev_value": previous,
                            "explanation": (
                                f"{district} {indicator.replace('_', ' ').title()} "
                                f"{direction} by {abs(change_pct):.1f}% compared "
                                f"with the previous month, exceeding the "
                                f"{threshold:.0f}% significant-change threshold."
                            ),
                        }
                    )

    if not records:
        return _empty_trends()

    return pd.DataFrame(records)


def _empty_outliers():
    return pd.DataFrame(
        columns=[
            "type", "indicator", "entity", "period",
            "metric", "change", "outlier_score", "explanation"
        ]
    )


def detect_outliers(df, indicators, method="IQR", threshold=1.5):
    """Detect indicator values that are unusually far from the distribution."""
    records = []

    if not indicators:
        return _empty_outliers()

    for indicator in indicators:
        if indicator not in df.columns:
            continue

        values = pd.to_numeric(df[indicator], errors="coerce")
        valid = values.dropna()

        if len(valid) < 3:
            continue

        if method == "IQR":
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr

            for idx, value in values.items():
                if pd.isna(value):
                    continue

                if value < lower or value > upper:
                    if value > upper:
                        distance = (value - upper) / iqr
                        direction = "above"
                    else:
                        distance = (lower - value) / iqr
                        direction = "below"

                    records.append(
                        {
                            "type": "outlier",
                            "indicator": indicator,
                            "entity": str(df.loc[idx, "district"]),
                            "period": str(df.loc[idx, "month"]),
                            "metric": float(value),
                            "change": np.nan,
                            "outlier_score": float(distance),
                            "explanation": (
                                f"{df.loc[idx, 'district']} "
                                f"{indicator.replace('_', ' ').title()} value "
                                f"of {value:g} is an outlier, lying "
                                f"{distance:.1f} IQR beyond the {direction} "
                                f"outlier boundary."
                            ),
                        }
                    )

        else:
            mean = valid.mean()
            std = valid.std(ddof=0)

            if std == 0:
                continue

            z_scores = (values - mean) / std

            for idx, z in z_scores.items():
                if pd.isna(z):
                    continue

                if abs(z) >= threshold:
                    value = float(values.loc[idx])
                    records.append(
                        {
                            "type": "outlier",
                            "indicator": indicator,
                            "entity": str(df.loc[idx, "district"]),
                            "period": str(df.loc[idx, "month"]),
                            "metric": value,
                            "change": np.nan,
                            "outlier_score": float(abs(z)),
                            "explanation": (
                                f"{df.loc[idx, 'district']} "
                                f"{indicator.replace('_', ' ').title()} value "
                                f"of {value:g} has a Z-score of {z:.2f}, "
                                f"exceeding the configured threshold of "
                                f"{threshold:.1f}."
                            ),
                        }
                    )

    if not records:
        return _empty_outliers()

    return pd.DataFrame(records)


def _empty_correlations():
    return pd.DataFrame(
        columns=[
            "type", "indicator", "entity", "period",
            "metric", "change", "correlation", "explanation"
        ]
    )


def detect_correlations(df, indicators, threshold=0.70):
    """Return correlation matrix and unique pairs above the threshold."""
    available = [
        c for c in indicators
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 2
    ]

    if len(available) < 2:
        return _empty_correlations(), pd.DataFrame()

    numeric = df[available].apply(pd.to_numeric, errors="coerce")
    corr_matrix = numeric.corr(method="pearson")

    records = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            a = available[i]
            b = available[j]
            r = corr_matrix.loc[a, b]

            if pd.isna(r):
                continue

            if abs(r) >= threshold:
                direction = "positive" if r > 0 else "negative"
                records.append(
                    {
                        "type": "correlation",
                        "indicator": f"{a} ↔ {b}",
                        "entity": "All selected districts",
                        "period": "All selected periods",
                        "metric": float(r),
                        "change": float(r),
                        "correlation": float(r),
                        "explanation": (
                            f"{a.replace('_', ' ').title()} and "
                            f"{b.replace('_', ' ').title()} show a strong "
                            f"{direction} correlation (r = {r:.2f})."
                        ),
                    }
                )

    if not records:
        return _empty_correlations(), corr_matrix

    return pd.DataFrame(records), corr_matrix


def _severity_for_trend(change, threshold):
    magnitude = abs(float(change))
    if magnitude >= 2 * threshold:
        return "High"
    if magnitude >= threshold:
        return "Medium"
    return "Low"


def _severity_for_outlier(score):
    if pd.isna(score):
        return "Medium"
    return "High" if float(score) >= 1.0 else "Medium"


def _severity_for_correlation(r, threshold):
    magnitude = abs(float(r))
    if magnitude >= 0.90:
        return "High"
    if magnitude >= threshold:
        return "Medium"
    return "Low"


def generate_insights(
    trends,
    outliers,
    correlations,
    trend_threshold=10.0,
    correlation_threshold=0.70,
    outlier_method="IQR",
):
    """Combine all detection results into the assignment's insight schema."""
    records = []

    for _, row in trends.iterrows():
        records.append(
            {
                "type": "trend",
                "indicator": row["indicator"],
                "entity": row["entity"],
                "period": row["period"],
                "metric": float(row["metric"]),
                "change": float(row["change"]),
                "severity": _severity_for_trend(row["change"], trend_threshold),
                "explanation": row["explanation"],
            }
        )

    for _, row in outliers.iterrows():
        records.append(
            {
                "type": "outlier",
                "indicator": row["indicator"],
                "entity": row["entity"],
                "period": row["period"],
                "metric": float(row["metric"]),
                "change": np.nan,
                "severity": _severity_for_outlier(row["outlier_score"]),
                "explanation": row["explanation"],
            }
        )

    for _, row in correlations.iterrows():
        records.append(
            {
                "type": "correlation",
                "indicator": row["indicator"],
                "entity": row["entity"],
                "period": row["period"],
                "metric": float(row["metric"]),
                "change": float(row["change"]),
                "severity": _severity_for_correlation(
                    row["correlation"], correlation_threshold
                ),
                "explanation": row["explanation"],
            }
        )

    columns = [
        "insight_id",
        "type",
        "indicator",
        "entity",
        "period",
        "metric",
        "change",
        "severity",
        "explanation",
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    result = pd.DataFrame(records)
    result["_severity_order"] = result["severity"].map(severity_order)
    result = result.sort_values(
        ["_severity_order", "type", "entity", "period"],
        na_position="last",
    ).drop(columns="_severity_order").reset_index(drop=True)

    result.insert(
        0,
        "insight_id",
        [f"INS-{i:04d}" for i in range(1, len(result) + 1)],
    )

    return result[columns]
