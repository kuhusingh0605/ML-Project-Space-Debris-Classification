"""
dashboard/app.py — Space Debris Collision Risk Predictor Streamlit Dashboard

Dark aerospace-inspired interactive dashboard for exploring conjunction risk predictions.
Displays KPI metrics, risk-ranked conjunction table, and interactive visualizations.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ─── Page config must be the first Streamlit call ─────────────────────────
st.set_page_config(
    page_title="Space Debris Risk Predictor",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark Aerospace CSS ────────────────────────────────────────────────────
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #030712 0%, #0A0F1E 50%, #020617 100%);
    color: #E5E7EB;
}
.stSidebar > div:first-child {
    background: linear-gradient(180deg, #0F172A 0%, #0A0F1E 100%);
    border-right: 1px solid #1E3A5F;
}
/* KPI Card */
.kpi-card {
    background: linear-gradient(135deg, #0F172A 0%, #111827 100%);
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,212,255,0.08);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,212,255,0.16);
}
.kpi-value {
    font-size: 2.6rem;
    font-weight: 900;
    line-height: 1;
    background: linear-gradient(135deg, #00D4FF, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.kpi-label {
    font-size: 0.85rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
    font-weight: 600;
}
.kpi-value-red { background: linear-gradient(135deg, #FF4D4D, #FF6B35); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.kpi-value-amber { background: linear-gradient(135deg, #F59E0B, #FBBF24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.kpi-value-green { background: linear-gradient(135deg, #10B981, #34D399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
/* Section titles */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #00D4FF;
    border-bottom: 1px solid #1E3A5F;
    padding-bottom: 8px;
    margin-bottom: 16px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
/* Risk badges */
.badge-high   { background:#FF4D4D22; color:#FF4D4D; border:1px solid #FF4D4D55; border-radius:6px; padding:2px 10px; font-weight:700; font-size:0.78rem; }
.badge-medium { background:#F59E0B22; color:#F59E0B; border:1px solid #F59E0B55; border-radius:6px; padding:2px 10px; font-weight:700; font-size:0.78rem; }
.badge-low    { background:#10B98122; color:#10B981; border:1px solid #10B98155; border-radius:6px; padding:2px 10px; font-weight:700; font-size:0.78rem; }
/* Disclaimer box */
.disclaimer {
    background: #0F172A;
    border-left: 3px solid #F59E0B;
    border-radius: 6px;
    padding: 12px 16px;
    color: #D1D5DB;
    font-size: 0.82rem;
    line-height: 1.5;
    margin-bottom: 16px;
}
/* Benchmark card */
.benchmark-card {
    background: linear-gradient(135deg, #0F172A, #1A2744);
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 16px;
}
/* Mono font for values */
.mono { font-family: 'JetBrains Mono', monospace; }
/* Hide default streamlit footer */
footer { visibility: hidden; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─── Matplotlib dark style ─────────────────────────────────────────────────
PALETTE = {
    "bg": "#030712",
    "panel": "#0F172A",
    "accent": "#00D4FF",
    "accent2": "#7C3AED",
    "high": "#FF4D4D",
    "medium": "#F59E0B",
    "low": "#10B981",
    "muted": "#374151",
    "text": "#E5E7EB",
}

def apply_mpl_dark(ax, title=""):
    ax.set_facecolor(PALETTE["panel"])
    ax.figure.patch.set_facecolor(PALETTE["bg"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"], which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["muted"])
    ax.grid(True, color=PALETTE["muted"], alpha=0.3, linewidth=0.5)


# ─── Data loading helpers ──────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_scored_results():
    path = Path("results/risk_scored_conjunctions.csv")
    if path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_benchmark():
    path = Path("results/benchmark.json")
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=60)
def load_metrics():
    path = Path("results/metrics.json")
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=60)
def load_propagated_states():
    parquet_path = Path("data/processed/propagated_states.parquet")
    csv_path = Path("data/processed/propagated_states.csv")
    if parquet_path.is_file():
        return pd.read_parquet(parquet_path)
    elif csv_path.is_file():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# ─── Sidebar ──────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    st.sidebar.markdown(
        "<div style='text-align:center;padding:12px 0 4px;'>"
        "<span style='font-size:2rem;'>🛸</span>"
        "<p style='font-size:1rem;font-weight:700;color:#00D4FF;margin:4px 0 0;'>Space Debris</p>"
        "<p style='font-size:0.7rem;color:#6B7280;margin:0;'>Collision Risk Predictor</p>"
        "</div>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔭 Filters")

    if df.empty:
        return df

    # Risk category filter
    risk_options = ["ALL"] + sorted(df["risk_category"].dropna().unique().tolist())
    selected_risk = st.sidebar.multiselect(
        "Risk Category", options=risk_options[1:], default=risk_options[1:]
    )

    # Max distance slider
    max_dist = float(df["minimum_distance_km"].max()) if "minimum_distance_km" in df.columns else 500.0
    min_dist = float(df["minimum_distance_km"].min()) if "minimum_distance_km" in df.columns else 0.0
    dist_range = st.sidebar.slider(
        "Minimum Distance (km)", min_value=0.0,
        max_value=max(max_dist, 1.0),
        value=(min_dist, max_dist), step=1.0
    )

    # Object name search
    sat_search = st.sidebar.text_input("Search Satellite Name", placeholder="e.g. STARLINK")
    deb_search = st.sidebar.text_input("Search Debris Name", placeholder="e.g. COSMOS")

    # Apply filters
    filtered = df.copy()
    if selected_risk:
        filtered = filtered[filtered["risk_category"].isin(selected_risk)]
    filtered = filtered[
        (filtered["minimum_distance_km"] >= dist_range[0]) &
        (filtered["minimum_distance_km"] <= dist_range[1])
    ]
    if sat_search:
        filtered = filtered[filtered["sat_name"].str.contains(sat_search, case=False, na=False)]
    if deb_search:
        filtered = filtered[filtered["debris_name"].str.contains(deb_search, case=False, na=False)]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Showing {len(filtered)} of {len(df)} pairs**")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:0.72rem;color:#4B5563;'>"
        "🎓 B.Tech ML Feasibility Study<br>"
        "Kuhu Singh · Janhavi Sharma<br>"
        "Inspired by Peri (Dec 2025) GNN paper<br>"
        "Academic Use Only"
        "</div>",
        unsafe_allow_html=True
    )
    return filtered


# ─── KPI Cards ────────────────────────────────────────────────────────────
def render_kpi_row(df: pd.DataFrame, df_full: pd.DataFrame, benchmark: dict):
    df_states = load_propagated_states()
    n_objects = df_states["norad_id"].nunique() if not df_states.empty else 0
    n_total_pairs = benchmark.get("total_possible_pairs", len(df_full))
    n_candidates = benchmark.get("candidate_pairs_retained", len(df_full))
    n_high = len(df_full[df_full["risk_category"] == "HIGH"]) if not df_full.empty else 0
    n_medium = len(df_full[df_full["risk_category"] == "MEDIUM"]) if not df_full.empty else 0
    reduction = benchmark.get("reduction_percentage", 0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value'>{n_objects:,}</div>
            <div class='kpi-label'>Objects Propagated</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value'>{n_total_pairs:,}</div>
            <div class='kpi-label'>Possible Pairs</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value'>{n_candidates:,}</div>
            <div class='kpi-label'>Candidate Pairs</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value kpi-value-red'>{n_high}</div>
            <div class='kpi-label'>High Priority</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value kpi-value-amber'>{n_medium}</div>
            <div class='kpi-label'>Medium Priority</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value kpi-value-green'>{reduction:.1f}%</div>
            <div class='kpi-label'>Pair Reduction</div>
        </div>""", unsafe_allow_html=True)


# ─── Conjunction Table ─────────────────────────────────────────────────────
def render_conjunction_table(df: pd.DataFrame):
    st.markdown("<div class='section-header'>🎯 Ranked Conjunction Candidates</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='disclaimer'>⚠️ <strong>Academic Disclaimer:</strong> "
        "Minimum separation values are <em>sample-based predicted estimates</em> from simulated propagation. "
        "Risk categories are experimental model-priority labels, NOT official collision probabilities. "
        "This system is NOT suitable for operational spacecraft collision avoidance decisions.</div>",
        unsafe_allow_html=True
    )

    if df.empty:
        st.info("No conjunction pairs match the current filters.")
        return

    display_cols = {
        "rank": "Rank",
        "sat_name": "Satellite",
        "debris_name": "Debris Object",
        "minimum_distance_km": "Min Dist (km)",
        "relative_velocity_km_s": "Rel. Velocity (km/s)",
        "altitude_difference_km": "Alt. Diff (km)",
        "time_to_tca_min": "TCA (min)",
        "risk_score": "Risk Score",
        "risk_category": "Risk Tier",
    }

    available_cols = {k: v for k, v in display_cols.items() if k in df.columns}
    df_display = df[list(available_cols.keys())].copy()
    df_display = df_display.rename(columns=available_cols)

    # Format numeric columns
    for col in ["Min Dist (km)", "Rel. Velocity (km/s)", "Alt. Diff (km)", "TCA (min)", "Risk Score"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )

    # Color risk tiers
    def color_risk(val):
        colors_map = {
            "HIGH": "color:#FF4D4D; font-weight:700;",
            "MEDIUM": "color:#F59E0B; font-weight:700;",
            "LOW": "color:#10B981; font-weight:600;",
        }
        return colors_map.get(str(val), "")

    st.dataframe(
        df_display.head(50),
        use_container_width=True,
        height=420,
    )


# ─── Visualizations ────────────────────────────────────────────────────────
def render_visualizations(df_filtered: pd.DataFrame, df_full: pd.DataFrame, benchmark: dict):
    st.markdown("<div class='section-header'>📊 Analysis & Visualizations</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📏 Distance Distribution",
        "🏷️ Risk Categories",
        "⚡ Velocity Analysis",
        "🖥️ Computational Benchmark"
    ])

    # ── Tab 1: Distance Distribution ──────────────────────────────────────
    with tab1:
        if df_full.empty or "minimum_distance_km" not in df_full.columns:
            st.info("No data available.")
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor(PALETTE["bg"])

            # Histogram
            data = df_full["minimum_distance_km"].dropna()
            ax1.hist(data, bins=25, color=PALETTE["accent"], alpha=0.75, edgecolor=PALETTE["muted"])
            ax1.axvline(data.mean(), color=PALETTE["high"], linestyle="--", linewidth=1.5, label=f"Mean: {data.mean():.1f} km")
            ax1.axvline(data.median(), color=PALETTE["medium"], linestyle="--", linewidth=1.5, label=f"Median: {data.median():.1f} km")
            apply_mpl_dark(ax1, "Distribution of Min. Predicted Separation")
            ax1.set_xlabel("Minimum Distance (km)")
            ax1.set_ylabel("Count")
            legend = ax1.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"], fontsize=9)

            # CDF
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax2.plot(sorted_data, cdf, color=PALETTE["accent"], linewidth=2)
            ax2.fill_between(sorted_data, cdf, alpha=0.15, color=PALETTE["accent"])
            apply_mpl_dark(ax2, "Cumulative Distribution of Separations")
            ax2.set_xlabel("Minimum Distance (km)")
            ax2.set_ylabel("Cumulative Probability")
            ax2.set_ylim([0, 1.05])

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Tab 2: Risk Categories ─────────────────────────────────────────────
    with tab2:
        if df_full.empty or "risk_category" not in df_full.columns:
            st.info("No risk category data available.")
        else:
            cat_counts = df_full["risk_category"].value_counts()
            cat_order = ["HIGH", "MEDIUM", "LOW"]
            cat_colors = [PALETTE["high"], PALETTE["medium"], PALETTE["low"]]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor(PALETTE["bg"])

            # Donut chart
            labels, values, colors = [], [], []
            for cat, color in zip(cat_order, cat_colors):
                if cat in cat_counts.index:
                    labels.append(cat)
                    values.append(cat_counts[cat])
                    colors.append(color)

            if values:
                wedges, texts, autotexts = ax1.pie(
                    values, labels=labels, colors=colors, autopct="%1.1f%%",
                    startangle=90, pctdistance=0.75,
                    wedgeprops={"linewidth": 2, "edgecolor": PALETTE["bg"]},
                )
                for t in texts:
                    t.set_color(PALETTE["text"])
                for at in autotexts:
                    at.set_color(PALETTE["bg"])
                    at.set_fontweight("bold")
                centre = plt.Circle((0, 0), 0.52, color=PALETTE["panel"])
                ax1.add_patch(centre)
                ax1.text(0, 0, f"{sum(values)}\nPairs", ha="center", va="center",
                         color=PALETTE["text"], fontsize=11, fontweight="bold")
            apply_mpl_dark(ax1, "Risk Category Distribution")

            # Bar chart showing distances per category
            for i, (cat, color) in enumerate(zip(cat_order, cat_colors)):
                sub = df_full[df_full["risk_category"] == cat]["minimum_distance_km"]
                if not sub.empty:
                    ax2.bar([i], [sub.mean()], color=color, alpha=0.8, width=0.5, label=cat)
                    ax2.errorbar([i], [sub.mean()], yerr=[[sub.mean() - sub.min()], [sub.max() - sub.mean()]],
                                 fmt="none", color=PALETTE["text"], capsize=5, linewidth=1.5)
            ax2.set_xticks(range(len(cat_order)))
            ax2.set_xticklabels([c for c in cat_order])
            apply_mpl_dark(ax2, "Mean Distance by Risk Category (km)")
            ax2.set_ylabel("Distance (km)")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Tab 3: Velocity Analysis ───────────────────────────────────────────
    with tab3:
        if df_full.empty or "relative_velocity_km_s" not in df_full.columns:
            st.info("No velocity data available.")
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor(PALETTE["bg"])

            # Scatter plot: distance vs velocity
            cat_color_map = {"HIGH": PALETTE["high"], "MEDIUM": PALETTE["medium"], "LOW": PALETTE["low"]}
            for cat, color in cat_color_map.items():
                sub = df_full[df_full["risk_category"] == cat] if "risk_category" in df_full.columns else df_full
                if not sub.empty:
                    ax1.scatter(
                        sub["relative_velocity_km_s"],
                        sub["minimum_distance_km"],
                        c=color, alpha=0.7, s=50, label=cat,
                        edgecolors=PALETTE["muted"], linewidth=0.5
                    )
            apply_mpl_dark(ax1, "Velocity vs. Min Separation")
            ax1.set_xlabel("Relative Velocity (km/s)")
            ax1.set_ylabel("Min Separation (km)")
            legend = ax1.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])

            # Velocity histogram
            vel_data = df_full["relative_velocity_km_s"].dropna()
            ax2.hist(vel_data, bins=20, color=PALETTE["accent2"], alpha=0.75, edgecolor=PALETTE["muted"])
            apply_mpl_dark(ax2, "Relative Velocity Distribution")
            ax2.set_xlabel("Relative Velocity (km/s)")
            ax2.set_ylabel("Count")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Tab 4: Computational Benchmark ────────────────────────────────────
    with tab4:
        if not benchmark:
            st.info("Benchmark data not available. Run `python main.py` first.")
        else:
            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown(
                    f"""<div class='benchmark-card'>
                    <h4 style='color:#00D4FF;margin:0 0 12px'>📊 Computational Reduction</h4>
                    <table style='width:100%;color:#E5E7EB;font-size:0.9rem;'>
                    <tr><td>Total Possible Pairs</td><td style='text-align:right;font-family:monospace;color:#00D4FF;font-weight:700;'>{benchmark.get('total_possible_pairs', 'N/A'):,}</td></tr>
                    <tr><td>After Locality Filter</td><td style='text-align:right;font-family:monospace;color:#10B981;font-weight:700;'>{benchmark.get('candidate_pairs_retained', 'N/A'):,}</td></tr>
                    <tr><td>Pairs Eliminated</td><td style='text-align:right;font-family:monospace;color:#FF4D4D;font-weight:700;'>{benchmark.get('pairs_removed', 'N/A'):,}</td></tr>
                    <tr style='border-top:1px solid #374151;'><td><strong>Reduction %</strong></td><td style='text-align:right;font-family:monospace;color:#F59E0B;font-weight:700;font-size:1.1rem;'>{benchmark.get('reduction_percentage', 0):.2f}%</td></tr>
                    </table>
                    </div>""",
                    unsafe_allow_html=True
                )
            with bc2:
                st.markdown(
                    f"""<div class='benchmark-card'>
                    <h4 style='color:#00D4FF;margin:0 0 12px'>⏱️ Runtime Comparison</h4>
                    <table style='width:100%;color:#E5E7EB;font-size:0.9rem;'>
                    <tr><td>Naive Pairwise Runtime</td><td style='text-align:right;font-family:monospace;color:#FF4D4D;font-weight:700;'>{benchmark.get('baseline_runtime_sec', 0):.4f} s</td></tr>
                    <tr><td>Locality-Filtered Runtime</td><td style='text-align:right;font-family:monospace;color:#10B981;font-weight:700;'>{benchmark.get('filtered_runtime_sec', 0):.4f} s</td></tr>
                    <tr style='border-top:1px solid #374151;'><td><strong>Measured Speedup</strong></td><td style='text-align:right;font-family:monospace;color:#7C3AED;font-weight:700;font-size:1.1rem;'>{benchmark.get('speedup_factor', 1.0):.2f}×</td></tr>
                    </table>
                    <p style='color:#6B7280;font-size:0.75rem;margin-top:12px;'>
                    Runtimes measured empirically. Results vary with dataset size and hardware.
                    </p>
                    </div>""",
                    unsafe_allow_html=True
                )

            # Visual bar comparison
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor(PALETTE["bg"])
            methods = ["Naive Pairwise", "Locality Filtered"]
            times = [benchmark.get("baseline_runtime_sec", 1), benchmark.get("filtered_runtime_sec", 1)]
            colors = [PALETTE["high"], PALETTE["low"]]
            bars = ax.barh(methods, times, color=colors, alpha=0.85, height=0.45)
            for bar, t in zip(bars, times):
                ax.text(t + 0.001, bar.get_y() + bar.get_height()/2,
                        f"{t:.4f} s", va="center", color=PALETTE["text"], fontsize=10, fontweight="bold")
            apply_mpl_dark(ax, "Runtime Comparison (Naive vs. Locality-Filtered)")
            ax.set_xlabel("Execution Time (seconds)")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ─── ML Metrics Section ────────────────────────────────────────────────────
def render_ml_metrics(metrics: dict):
    st.markdown("<div class='section-header'>🤖 Model Performance Metrics</div>", unsafe_allow_html=True)
    if not metrics:
        st.info("No metrics found. Run `python main.py` first to train models.")
        return

    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    cols = st.columns(len(metrics))
    for i, (model_name, m) in enumerate(metrics.items()):
        with cols[i]:
            st.markdown(f"**{model_name}**")
            for k, label in zip(metric_names, metric_labels):
                val = m.get(k)
                if val is not None:
                    color = "#00D4FF" if val >= 0.75 else ("#F59E0B" if val >= 0.55 else "#FF4D4D")
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1E3A5F;'>"
                        f"<span style='color:#9CA3AF;font-size:0.85rem;'>{label}</span>"
                        f"<span style='color:{color};font-weight:700;font-family:monospace;'>{val:.4f}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

    # Display saved plots
    results_dir = Path("results")
    plot_cols = st.columns(3)
    plot_idx = 0
    for plot_file in sorted(results_dir.glob("*.png")):
        with plot_cols[plot_idx % 3]:
            st.image(str(plot_file), caption=plot_file.stem.replace("_", " ").title(), use_container_width=True)
        plot_idx += 1


# ─── Main App ──────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown(
        "<div style='text-align:center;padding:24px 0 8px;'>"
        "<h1 style='font-size:2.4rem;font-weight:900;background:linear-gradient(135deg,#00D4FF,#7C3AED);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0;'>"
        "🛸 Space Debris Collision Risk Predictor</h1>"
        "<p style='color:#6B7280;font-size:0.95rem;margin:6px 0 0;'>"
        "B.Tech ML Feasibility Study &nbsp;·&nbsp; SGP4 + Locality Filtering + Classical ML &nbsp;·&nbsp; "
        "Kuhu Singh &amp; Janhavi Sharma</p>"
        "</div>",
        unsafe_allow_html=True
    )

    # Check if results exist
    results_ready = Path("results/risk_scored_conjunctions.csv").is_file()
    if not results_ready:
        st.warning(
            "⚠️ **Results not yet generated.** Please run the pipeline first:\n"
            "```bash\npython main.py\n```\n"
            "Then refresh this page."
        )
        st.stop()

    # Load data
    df_full = load_scored_results()
    benchmark = load_benchmark()
    metrics = load_metrics()

    # Sidebar with filters
    df_filtered = render_sidebar(df_full)

    # KPI cards
    st.markdown("<br>", unsafe_allow_html=True)
    render_kpi_row(df_filtered, df_full, benchmark)
    st.markdown("<br>", unsafe_allow_html=True)

    # Main content
    tab_main, tab_models = st.tabs(["🎯 Conjunction Analysis", "🤖 ML Models"])

    with tab_main:
        render_conjunction_table(df_filtered)
        st.markdown("<br>", unsafe_allow_html=True)
        render_visualizations(df_filtered, df_full, benchmark)

    with tab_models:
        render_ml_metrics(metrics)


if __name__ == "__main__":
    main()
