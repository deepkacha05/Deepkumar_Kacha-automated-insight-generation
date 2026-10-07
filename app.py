import io
import pandas as pd
import streamlit as st
import plotly.express as px

from insight_engine import (
    REQUIRED_COLUMNS,
    INDICATORS,
    detect_trends,
    detect_outliers,
    detect_correlations,
    generate_insights,
)

st.set_page_config(
    page_title="Automated Healthcare Insight Engine",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Automated Healthcare Insight Engine")
st.caption(
    "Upload district-level healthcare data and automatically discover "
    "trends, outliers, correlations, and human-readable insights."
)

# -----------------------------
# Data loading
# -----------------------------
uploaded = st.file_uploader("Upload healthcare CSV", type=["csv"])

if uploaded is not None:
    try:
        # ``utf-8-sig`` also handles CSVs exported by Excel with a BOM in
        # the first column name.
        df = pd.read_csv(uploaded, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        st.error("Could not read the CSV: the uploaded file is empty.")
        st.stop()
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read the CSV. Check its encoding and delimiter: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
else:
    try:
        df = pd.read_csv("data.csv")
        st.info("Using the included sample data.csv. Upload another CSV to analyze it.")
    except FileNotFoundError:
        st.error("data.csv was not found. Upload a CSV file to continue.")
        st.stop()

# Normalize month and validate columns
df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.error(f"Invalid dataset. Missing required columns: {', '.join(missing)}")
    st.stop()

# Keep filter values consistent even when a CSV stores district IDs as
# numbers. ``string`` preserves missing values instead of turning them into
# the literal text "nan".
df["month"] = df["month"].astype("string").str.strip()
df["district"] = df["district"].astype("string").str.strip()
for col in INDICATORS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Validation summary
# -----------------------------
with st.expander("🔎 Data validation", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Districts", df["district"].nunique())
    c3.metric("Missing cells", int(df.isna().sum().sum()))

    missing_table = df.isna().sum().rename("missing_values").to_frame()
    st.dataframe(missing_table, use_container_width=True)

    if df.isna().sum().sum() == 0:
        st.success("✅ Dataset validation passed: no missing values detected.")
    else:
        st.warning("⚠️ Missing values detected. Detection functions ignore invalid numeric values.")

# -----------------------------
# Sidebar filters/configuration
# -----------------------------
st.sidebar.header("⚙️ Filters & Configuration")

district_options = sorted(df["district"].dropna().astype(str).unique())
month_options = sorted(df["month"].dropna().astype(str).unique())

selected_districts = st.sidebar.multiselect(
    "District",
    district_options,
    default=district_options,
)

selected_months = st.sidebar.multiselect(
    "Month",
    month_options,
    default=month_options,
)

selected_indicators = st.sidebar.multiselect(
    "Indicator",
    INDICATORS,
    default=INDICATORS,
)

trend_threshold = st.sidebar.slider(
    "Trend threshold (%)",
    min_value=1,
    max_value=50,
    value=10,
    step=1,
)

outlier_method = st.sidebar.selectbox(
    "Outlier method",
    ["IQR", "Z-score"],
)

if outlier_method == "IQR":
    outlier_threshold = st.sidebar.slider(
        "IQR multiplier",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
    )
else:
    outlier_threshold = st.sidebar.slider(
        "Z-score threshold",
        min_value=1.5,
        max_value=5.0,
        value=3.0,
        step=0.1,
    )

correlation_threshold = st.sidebar.slider(
    "Correlation threshold |r|",
    min_value=0.50,
    max_value=0.99,
    value=0.70,
    step=0.01,
)

# Apply live filters
filtered = df.copy()

if selected_districts:
    filtered = filtered[filtered["district"].isin(selected_districts)]

if selected_months:
    filtered = filtered[filtered["month"].isin(selected_months)]

if selected_indicators:
    analysis_indicators = selected_indicators
else:
    analysis_indicators = []

if filtered.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

# -----------------------------
# Run analytics
# -----------------------------
trends = detect_trends(filtered, analysis_indicators, trend_threshold)
outliers = detect_outliers(
    filtered,
    analysis_indicators,
    method=outlier_method,
    threshold=outlier_threshold,
)
correlations, corr_matrix = detect_correlations(
    filtered,
    analysis_indicators,
    correlation_threshold,
)

insights = generate_insights(
    trends,
    outliers,
    correlations,
    trend_threshold=trend_threshold,
    correlation_threshold=correlation_threshold,
    outlier_method=outlier_method,
)

# -----------------------------
# KPI cards
# -----------------------------
st.subheader("📊 Overview")

total_insights = len(insights)
high_count = int((insights["severity"] == "High").sum()) if not insights.empty else 0
district_count = filtered["district"].nunique()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Insights", total_insights)
k2.metric("High Severity", high_count)
k3.metric("Districts", district_count)
k4.metric("Rows Analyzed", len(filtered))

# -----------------------------
# Insights
# -----------------------------
st.subheader("💡 Automated Insights")

if insights.empty:
    st.info(
        "No insights were flagged with the current filters and thresholds. "
        "Try a lower trend/correlation threshold or select more months/districts."
    )
else:
    display_cols = [
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
    st.dataframe(
        insights[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = insights.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download insights.csv",
        data=csv_bytes,
        file_name="insights.csv",
        mime="text/csv",
    )

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("📈 Visualizations")

left, right = st.columns(2)

with left:
    severity_counts = (
        insights["severity"].value_counts()
        .reindex(["High", "Medium", "Low"], fill_value=0)
        .rename_axis("severity")
        .reset_index(name="count")
    )
    fig_severity = px.bar(
        severity_counts,
        x="severity",
        y="count",
        title="Insights by Severity",
        labels={"severity": "Severity", "count": "Insights"},
    )
    st.plotly_chart(fig_severity, use_container_width=True)

with right:
    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        st.info("Select at least two indicators to display the correlation heatmap.")
    else:
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            title="Pearson Correlation Heatmap",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# District line chart
st.markdown("### 📈 Per-District Indicator Trend")

if analysis_indicators:
    chart_indicator = st.selectbox(
        "Indicator for line chart",
        analysis_indicators,
    )
    chart_districts = sorted(filtered["district"].dropna().unique())

    if chart_districts:
        chart_district = st.selectbox(
            "District for line chart",
            chart_districts,
        )
        chart_df = filtered[filtered["district"] == chart_district].copy()
        chart_df = chart_df.sort_values("month")

        fig_line = px.line(
            chart_df,
            x="month",
            y=chart_indicator,
            markers=True,
            title=f"{chart_indicator} — {chart_district}",
            labels={chart_indicator: chart_indicator.replace("_", " ").title()},
        )
        st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------
# Correlation download
# -----------------------------
st.subheader("🔗 Correlation Matrix")

if corr_matrix.empty:
    st.info("Not enough numeric indicators for a correlation matrix.")
else:
    st.dataframe(corr_matrix, use_container_width=True)
    corr_csv = corr_matrix.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download correlation_matrix.csv",
        data=corr_csv,
        file_name="correlation_matrix.csv",
        mime="text/csv",
    )

# -----------------------------
# Raw filtered data
# -----------------------------
with st.expander("📋 Filtered dataset"):
    st.dataframe(filtered, use_container_width=True, hide_index=True)

st.caption(
    "Note: The provided sample is small (2 months × 6 districts), so Pearson "
    "correlations should be interpreted cautiously."
)
