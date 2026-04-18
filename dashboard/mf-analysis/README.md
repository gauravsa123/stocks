# 📊 Mutual Fund Analysis Dashboard

A comprehensive mutual fund analysis tool with financial ratios, MACD signals, rolling performance metrics and an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
mf-analysis
├── src
│   ├── main.py               # Core analysis engine (ratios, rolling, ATH, CAGR)
│   ├── streamlit_app.py      # Interactive Streamlit dashboard
│   ├── macd_functions.py     # MACD calculation & buy/sell signal logic
│   ├── mf_analysis.py        # NAV data loading & fund filtering helpers
│   └── ratios.py             # Financial ratio calculations (Sharpe, Beta, etc.)
├── data
│   ├── mf_ga_amfi.csv        # Default portfolio CSV (name, code, id)
│   └── portfolio.csv         # Auto-saved user portfolio (generated at runtime)
├── output                    # Auto-created; stores charts & result CSV
│   ├── mf_results.csv
│   ├── ath_change.png
│   ├── ratio_<name>.png
│   └── rolling_<category>.png
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mf-analysis
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS / Linux
   venv\Scripts\activate          # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Run analysis from terminal
```bash
python src/main.py

# With a custom ATH cutoff date
python src/main.py --cutoff 2024-06-01
```

### Launch Streamlit dashboard
```bash
streamlit run src/streamlit_app.py
```

---

## 🖥️ Streamlit Dashboard Features

### ➕ Portfolio Management (Sidebar)
| Feature | Details |
|---------|---------|
| 🔍 Search | Live search across all AMFI funds as you type (≥ 2 chars) |
| 📂 Upload CSV | Upload a previously saved portfolio CSV |
| ♻️ Replace | Replace current portfolio with uploaded CSV |
| ➕ Merge | Merge uploaded CSV into existing portfolio (no duplicates) |
| 🗑 Remove | Remove individual funds or clear all |
| ⬇️ Download | Export current portfolio as CSV |

### 📅 ATH Cutoff Date (Sidebar)
- Defaults to **26-Sep-2024**
- User can select any custom date via date picker
- Changing the date auto-invalidates cache and reruns analysis

### 📋 Tab 1 — Summary
- Filterable table of all funds with ratios and CAGR
- KPI cards: Best Sharpe, Best Alpha, Best CAGR, Lowest Beta

### 📉 Tab 2 — ATH Change
- Color-coded dot plot of % NAV change from selected cutoff date
- 🟢 > 5% &nbsp; 🔵 0–5% &nbsp; 🟠 0 to -5% &nbsp; 🔴 < -5%

### 📊 Tab 3 — Ratios
- Horizontal bar chart per ratio, grouped by fund category
- 🕸 Radar chart to compare multiple funds across all ratios

### 🔄 Tab 4 — Rolling Ratios
- 18-month rolling ratio line chart per category
- Selectable category and ratio

---

## 📐 Metrics Computed

| Metric | Description |
|--------|-------------|
| **CAGR** | Compound Annual Growth Rate over the analysis period |
| **Beta** | Sensitivity to market (Nifty 50) movements |
| **Sharpe Ratio** | Risk-adjusted return over risk-free rate |
| **Alpha** | Excess return over expected CAPM return |
| **Information Ratio** | Excess return over benchmark per unit of tracking error |
| **Up Capture** | Performance relative to benchmark in up markets |
| **Down Capture** | Performance relative to benchmark in down markets |

---

## 📄 Portfolio CSV Format

The portfolio CSV used for analysis must contain these columns:

```csv
name,code,id
Mirae Asset Large Cap Fund,118989,large_3
Nippon India Small Cap Fund,118778,small_1
```

| Column | Description |
|--------|-------------|
| `name`  | Full scheme name |
| `code`  | AMFI scheme code |
| `id`    | Category: `large_3`, `mid_2`, `small_1`, `flexi_4` |

---

## 🔗 Data Sources

| Source | Usage |
|--------|-------|
| [AMFI India](https://www.amfiindia.com/spages/NAVAll.txt) | Live fund search & NAV data |
| [Yahoo Finance](https://finance.yahoo.com) | Nifty 50 (market) & risk-free fund data |

---

## 📦 Dependencies

```
pandas
numpy
yfinance
mftool
streamlit
plotly
requests
matplotlib
```

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file