import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import uuid

from main import (
    run_analysis, RATIO_NAMES, BENCHMARK_IDX, ALL_MFs, PORTFOLIO_PATH, DEFAULT_CUTOFF
)

CATEGORY_OPTIONS = ["large_3", "mid_2", "small_1", "flexi_4"]

# ── AMFI Fund Search ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching AMFI fund list...")
def fetch_amfi_funds() -> pd.DataFrame:
    import requests
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    records = []
    for line in resp.text.splitlines():
        parts = line.strip().split(";")
        if len(parts) >= 4 and parts[0].strip().isdigit():
            records.append({
                "code": parts[0].strip(),
                "name": parts[3].strip(),
                "nav":  parts[4].strip() if len(parts) > 4 else "",
                "date": parts[5].strip() if len(parts) > 5 else "",
            })
    return pd.DataFrame(records)



def get_portfolio_path() -> str:
    """Return a session-specific portfolio CSV path."""
    session_dir = os.path.join(os.path.dirname(__file__), "../data/sessions")
    os.makedirs(session_dir, exist_ok=True)
    return os.path.join(session_dir, f"portfolio_{st.session_state.session_id}.csv")


def save_portfolio_csv():
    if not st.session_state.get("portfolio_funds"):
        return None
    path = get_portfolio_path()
    df = pd.DataFrame(st.session_state.portfolio_funds)[["name", "id", "code"]]
    df.to_csv(path, index=False)
    return path


def load_portfolio_from_csv():
    path = get_portfolio_path()
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.session_state.portfolio_funds = df.to_dict(orient="records")


# ── Cleanup old session files (optional) ─────────────────────────────────────
def cleanup_old_sessions(max_age_hours: int = 24):
    """Delete session CSV files older than max_age_hours."""
    import time
    session_dir = os.path.join(os.path.dirname(__file__), "../data/sessions")
    if not os.path.exists(session_dir):
        return
    now = time.time()
    for fname in os.listdir(session_dir):
        fpath = os.path.join(session_dir, fname)
        if os.path.isfile(fpath):
            age_hours = (now - os.path.getmtime(fpath)) / 3600
            if age_hours > max_age_hours:
                os.remove(fpath)


cleanup_old_sessions()

# ── Init session state ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]   # unique per browser tab

if "portfolio_funds" not in st.session_state:
    st.session_state.portfolio_funds = []
    load_portfolio_from_csv()

    
# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MF Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Mutual Fund Analysis Dashboard")
st.markdown("Risk ratios, rolling performance & ATH change analysis.")

# ── Run Analysis ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Fetching data & computing ratios... this may take a few minutes.")
def load_analysis(csv_path: str, cutoff: str):        # ← cutoff added
    return run_analysis(csv_path, cutoff=cutoff)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    portfolio_ready = bool(st.session_state.portfolio_funds)
    run_btn = st.button(
        "🔄 Run Analysis on Portfolio",
        type="primary",
        use_container_width=True,
        disabled=not portfolio_ready,
        help="Add funds to portfolio first, then run analysis.",
    )
    if not portfolio_ready:
        st.caption("⚠️ Add funds to portfolio to enable analysis.")

    st.caption("Data: AMFI India + Yahoo Finance")
    
    st.markdown("---")

    # ── Two options via tabs ──────────────────────────────────────────────────
    st.subheader("➕ Add Funds to Portfolio")
    add_tab1, add_tab2 = st.tabs(["🔍 Search", "📂 Upload CSV"])

    amfi_df = fetch_amfi_funds()

    # ── TAB: Search ───────────────────────────────────────────────────────────
    with add_tab1:
        search_query = st.text_input(
            "Type fund name",
            placeholder="e.g. Mirae, HDFC, Nippon...",
            key="fund_search",
        )

        if search_query and len(search_query) >= 2:
            mask    = amfi_df["name"].str.contains(search_query, case=False, na=False)
            results = amfi_df[mask][["code", "name", "nav", "date"]].head(20)

            if results.empty:
                st.info("No funds found.")
            else:
                st.caption(f"{len(results)} result(s) found")
                selected_row = st.selectbox(
                    "Select a fund",
                    options=results["name"].tolist(),
                    key="fund_select",
                )

                if selected_row:
                    fund_info = results[results["name"] == selected_row].iloc[0]
                    st.code(
                        f"Code : {fund_info['code']}\n"
                        f"NAV  : ₹{fund_info['nav']}\n"
                        f"Date : {fund_info['date']}"
                    )

                    selected_cat = st.selectbox(
                        "Assign Category",
                        options=CATEGORY_OPTIONS,
                        help="Used to match the right benchmark for capture ratios.",
                        key="fund_cat",
                    )

                    if st.button("➕ Add to Portfolio", use_container_width=True):
                        codes = [f["code"] for f in st.session_state.portfolio_funds]
                        if fund_info["code"] not in codes:
                            st.session_state.portfolio_funds.append({
                                "name": fund_info["name"],
                                "code": fund_info["code"],
                                "id":   selected_cat,
                            })
                            save_portfolio_csv()
                            st.success(f"✅ Added: {fund_info['name'][:40]}")
                            st.rerun()
                        else:
                            st.warning("Already in portfolio.")
        else:
            st.caption("Type at least 2 characters to search.")

    # ── TAB: Upload CSV ───────────────────────────────────────────────────────
    with add_tab2:
        st.caption("CSV must have columns: `name`, `code`, `id`")
        uploaded_file = st.file_uploader(
            "Upload portfolio CSV",
            type=["csv"],
            key="portfolio_upload",
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                uploaded_df.columns = uploaded_df.columns.str.strip().str.lower()

                required_cols = {"name", "code", "id"}
                if not required_cols.issubset(set(uploaded_df.columns)):
                    st.error(f"CSV must contain columns: {required_cols}")
                else:
                    uploaded_df["code"] = uploaded_df["code"].astype(str).str.strip()

                    # Preview
                    st.dataframe(
                        uploaded_df[["name", "code", "id"]],
                        use_container_width=True,
                        height=180,
                    )

                    replace_col, merge_col = st.columns(2)

                    # Replace portfolio
                    if replace_col.button("♻️ Replace Portfolio", use_container_width=True):
                        st.session_state.portfolio_funds = uploaded_df[["name", "code", "id"]].to_dict(orient="records")
                        save_portfolio_csv()
                        st.success(f"✅ Replaced with {len(st.session_state.portfolio_funds)} fund(s).")
                        st.rerun()

                    # Merge into existing
                    if merge_col.button("➕ Merge into Portfolio", use_container_width=True):
                        existing_codes = {f["code"] for f in st.session_state.portfolio_funds}
                        new_funds = [
                            row.to_dict()
                            for _, row in uploaded_df[["name", "code", "id"]].iterrows()
                            if row["code"] not in existing_codes
                        ]
                        if new_funds:
                            st.session_state.portfolio_funds.extend(new_funds)
                            save_portfolio_csv()
                            st.success(f"✅ Added {len(new_funds)} new fund(s).")
                            st.rerun()
                        else:
                            st.info("All funds already in portfolio.")

            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    # ── Portfolio List ────────────────────────────────────────────────────────
    if st.session_state.portfolio_funds:
        st.markdown("---")
        st.subheader(f"📁 Portfolio ({len(st.session_state.portfolio_funds)} funds)")

        for i, f in enumerate(st.session_state.portfolio_funds):
            col_a, col_b = st.columns([4, 1])
            col_a.markdown(
                f"<small><b>{f['name'][:35]}...</b><br/>"
                f"<code>{f['code']}</code> · <i>{f['id']}</i></small>",
                unsafe_allow_html=True,
            )
            if col_b.button("🗑", key=f"del_{i}"):
                st.session_state.portfolio_funds.pop(i)
                save_portfolio_csv()
                st.rerun()

        col_clear, col_dl = st.columns(2)
        if col_clear.button("🗑 Clear All", use_container_width=True):
            st.session_state.portfolio_funds = []
            if os.path.exists(PORTFOLIO_PATH):
                os.remove(PORTFOLIO_PATH)
            st.rerun()

        portfolio_df = pd.DataFrame(st.session_state.portfolio_funds)
        col_dl.download_button(
            "⬇️ CSV",
            data=portfolio_df.to_csv(index=False).encode(),
            file_name="my_portfolio.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    

    # ── ATH Cutoff Date ───────────────────────────────────────────────────────
    st.subheader("📅 ATH Cutoff Date")
    default_cutoff_date = pd.Timestamp(DEFAULT_CUTOFF).date()
    cutoff_date = st.date_input(
        "Select cutoff date",
        value=default_cutoff_date,
        max_value=pd.Timestamp.today().date(),
        help=f"Default is {DEFAULT_CUTOFF}. NAV % change is calculated from this date to today.",
        key="cutoff_date",
    )
    if cutoff_date != default_cutoff_date:
        st.caption(f"⚠️ Default is {DEFAULT_CUTOFF}")
    else:
        st.caption(f"✅ Using default: {DEFAULT_CUTOFF}")

    st.markdown("---")

# ── Trigger Analysis ──────────────────────────────────────────────────────────
if run_btn:
    st.cache_data.clear()

csv_path = save_portfolio_csv()

if not csv_path or not os.path.exists(csv_path):
    st.info("👈 Search and add funds to your portfolio, then click **Run Analysis**.")
    st.stop()

cutoff_str = cutoff_date.strftime("%Y-%m-%d")        # pass to analysis
mf_df, mf_fall, my_mf_names = load_analysis(csv_path, cutoff_str)

# Drop rolling columns for display
display_cols = ['name', 'id', 'nav', 'returns_%'] + RATIO_NAMES
summary_df   = mf_df[display_cols].copy()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Summary",
    "📉 ATH Change",
    "📊 Ratios",
    "🔄 Rolling Ratios",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – Summary Table
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Fund Summary")
    st.caption("'mid', 'small', 'large', 'flexi' --> market category benchmark ratios")
    cats = ["All"] + sorted(mf_df["id"].unique().tolist())
    selected_cat = st.selectbox("Filter by Category", cats)
    filtered = summary_df if selected_cat == "All" else summary_df[summary_df["id"] == selected_cat]

    st.dataframe(filtered.round(2).reset_index(drop=True), use_container_width=True, height=500)

    st.markdown("### 🏆 Top Performers")
    col1, col2, col3, col4 = st.columns(4)
    best_sharpe = filtered.loc[filtered["sharpe"].idxmax()]
    best_alpha  = filtered.loc[filtered["alpha"].idxmax()]
    best_return = filtered.loc[filtered["returns_%"].idxmax()]
    low_beta    = filtered.loc[filtered["beta"].idxmin()]

    # ── Reduce metric font size ───────────────────────────────────────────
    st.markdown("""
        <style>
        [data-testid="stMetricLabel"]  { font-size: 12px !important; }
        [data-testid="stMetricValue"]  { font-size: 15px !important; }
        [data-testid="stMetricDelta"]  { font-size: 12px !important; }
        </style>
    """, unsafe_allow_html=True)
    col1.metric("***Best CAGR***",   best_return["name"], f"{best_return['returns_%']}%")
    col2.metric("***Best Sharpe***", best_sharpe["name"], f"{best_sharpe['sharpe']}")
    col3.metric("***Best Alpha***",  best_alpha["name"],  f"{best_alpha['alpha']}%")
    col4.metric("***Lowest Beta***", low_beta["name"],    f"{low_beta['beta']}")

    st.markdown("")   # spacer

    # ── Row 2 ─────────────────────────────────────────────────────────────
    best_info        = filtered.loc[filtered["info"].idxmax()]
    best_up_capture  = filtered.loc[filtered["up_capture"].idxmax()]
    best_down_capture = filtered.loc[filtered["down_capture"].idxmin()]   # lowest = best

    col5, col6, col7, _ = st.columns(4)
    col5.metric("***Best Info Ratio***",    best_info["name"],          f"{best_info['info']}")
    col6.metric("***Highest Up Capture***", best_up_capture["name"],    f"{best_up_capture['up_capture']}%")
    col7.metric("***Lowest Down Capture***", best_down_capture["name"], f"{best_down_capture['down_capture']}%")



# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – ATH Change  (update subheader to reflect chosen date)
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"% Change from ATH ({cutoff_date.strftime('%d-%b-%Y')})")   # ← dynamic
    st.caption("Percentage change from the all-time high (ATH) value. MFs having returns >0 shows good recovery compared to Market and other peers. MFs are color coded in Plot for segregation.")

    change_list, names = [], []
    for name, vals in mf_fall.items():
        if len(vals) < 2:
            continue
        change_list.append(round((vals[-1] - vals[0]) / vals[0] * 100, 2))
        names.append(name)

    ath_df = pd.DataFrame({"name": names, "change": change_list})
    ath_df = ath_df.sort_values("change").reset_index(drop=True)

    def ath_color(v):
        if v > 5:    return "green"
        elif v > 0:  return "magenta"
        elif v > -5: return "orange"
        else:        return "red"

    ath_df["color"] = ath_df["change"].apply(ath_color)

    fig_ath = go.Figure()
    fig_ath.add_trace(go.Scatter(
        x=ath_df["change"], y=ath_df["name"],
        mode="markers+lines",
        marker=dict(color=ath_df["color"], size=8),
        line=dict(color="lightgrey", width=1),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig_ath.add_vline(x=0, line_dash="dash", line_color="white", line_width=1)
    fig_ath.add_vline(x=-5, line_dash="dash", line_color="white", line_width=0.5)
    fig_ath.add_vline(x=5, line_dash="dash", line_color="white", line_width=0.5)

    # ── Colored Y-axis labels via annotations ─────────────────────────────
    x_anchor = ath_df["change"].min() - 1   # place labels just left of the plot
    for _, row in ath_df.iterrows():
        fig_ath.add_annotation(
            x=x_anchor,
            y=row["name"],
            text=row["name"],
            showarrow=False,
            xanchor="right",
            xref="x",
            yref="y",
            font=dict(color=row["color"], size=11),
        )

    fig_ath.update_yaxes(showticklabels=False)   # hide default tick labels
    fig_ath.update_layout(
        height=max(500, len(ath_df) * 20),
        xaxis_title="% Change",
        margin=dict(l=250),
    )
    st.plotly_chart(fig_ath, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Ratios
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Performance Ratios by Fund")

    RATIO_CAPTIONS = {
        "beta":         "📐 Beta — Measures fund's sensitivity to market movements. Beta < 1 means less volatile than market, > 1 means more volatile. \n Low value means less fluctuations compared to market and Good for short term durations",
        "sharpe":       "⚖️ Sharpe Ratio — Risk-adjusted return over the risk-free rate. Higher is better; > 1 is considered good.",
        "alpha":        "🎯 Alpha — Excess return generated over the expected market return.",
        "info":         "📡 Information Ratio — Consistency of excess returns over the benchmark. Higher means more consistent outperformance. It quantifies a fund manager's skill and consistency in outperforming the market.",
        "up_capture":   "📈 Up Capture Ratio — How much of the benchmark's gains the fund captured in rising markets. > 100% means outperformed benchmark on the way up.",
        "down_capture": "📉 Down Capture Ratio — How much of the benchmark's losses the fund suffered in falling markets. < 100% means fund fell less than benchmark.",
    }

    selected_ratio = st.selectbox("Select Ratio", RATIO_NAMES)
    # st.caption(RATIO_CAPTIONS[selected_ratio])
    st.markdown(
        f'<p style="font-size:16px; font-weight:bold; font-style:italic; color:white;">'
        f'{RATIO_CAPTIONS[selected_ratio]}'
        f'</p>', 
        unsafe_allow_html=True
    )


    ratio_df = mf_df[["name", "id", selected_ratio]].dropna().copy()

    fig_ratio = go.Figure()
    colors_map = px.colors.qualitative.Set2
    for i, (cat, grp) in enumerate(ratio_df.groupby("id")):
        grp = grp.sort_values(selected_ratio)
        fig_ratio.add_trace(go.Bar(
            y=grp["name"], x=grp[selected_ratio], name=cat,
            orientation="h", marker_color=colors_map[i % len(colors_map)],
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        ))
    fig_ratio.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
    fig_ratio.update_layout(
        height=max(500, len(ratio_df) * 22), xaxis_title=selected_ratio,
        barmode="overlay", margin=dict(l=250), legend_title="Category",
    )
    st.plotly_chart(fig_ratio, use_container_width=True)

    st.markdown("### 🕸 Ratio Radar – Compare Funds")
    fund_options   = mf_df["name"].tolist()
    selected_funds = st.multiselect("Select Funds", fund_options, default=fund_options[:3])
    if selected_funds:
        radar_df = mf_df[mf_df["name"].isin(selected_funds)][["name"] + RATIO_NAMES].dropna()
        fig_radar = go.Figure()
        for _, row in radar_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[r] for r in RATIO_NAMES], theta=RATIO_NAMES,
                fill="toself", name=row["name"].split(" ")[0],
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), height=450)
        st.plotly_chart(fig_radar, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – Rolling Ratios
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("18-Month Rolling Ratios")
    roll_cat   = st.selectbox("Select Category", sorted(mf_df["id"].unique().tolist()), key="roll_cat")
    roll_ratio = st.selectbox("Select Ratio", RATIO_NAMES, key="roll_ratio")

    cat_df   = mf_df[mf_df["id"] == roll_cat]
    fig_roll = go.Figure()
    for _, row in cat_df.iterrows():
        roll_vals = row[f"rolling_{roll_ratio}"]
        if not roll_vals or (isinstance(roll_vals, list) and len(roll_vals) == 0):
            continue
        fig_roll.add_trace(go.Scatter(
            y=roll_vals, mode="lines",
            name=row["name"].split(" ")[0],
            hovertemplate=f"%{{y:.2f}}<extra>{row['name'].split(' ')[0]}</extra>",
        ))
    fig_roll.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig_roll.update_layout(
        height=450, xaxis_title="Rolling Window", yaxis_title=roll_ratio,
        hovermode="x unified", legend=dict(font=dict(size=9)),
    )
    st.plotly_chart(fig_roll, use_container_width=True)