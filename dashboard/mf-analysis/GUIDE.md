# 📊 MF Analysis Dashboard – User Guide

---

## 🚀 Getting Started

Add funds for analysis. You can search and add (these can be saved for later use in CSV). Also you can upload CSV with funds.

### 1. Add Funds to Your Portfolio

Use the **sidebar** to add funds via two methods:

#### 🔍 Search Tab
1. Type at least **Few EXACT characters** of a fund name (e.g., "Motilal Oswal mid", "Nippon India Flexi")
2. Select a fund from the dropdown results (top 20 matches shown)
3. View fund details: Code, Current NAV, and Date
4. Assign a **Category** to the fund:

| Category | Description |
|----------|-------------|
| `small_1` | Small Cap |
| `mid_2` | Mid Cap |
| `large_3` | Large Cap |
| `flexi_4` | Flexi Cap |
| `hybrid_5` | Hybrid |
| `hedge_6` | Hedge (Gold, multi asset, bond funds) |
| `global_7` | Global |

5. Click **➕ Add to Portfolio**

> ⚠️ A fund already in the portfolio cannot be added again.

#### 📂 Upload CSV Tab
- Upload a CSV file with columns: `name`, `code`, `id`
- Preview the uploaded funds before confirming
- Choose an action:
  - **♻️ Replace Portfolio** — overwrites the current portfolio
  - **➕ Merge into Portfolio** — appends only new funds (skips duplicates)

---

## ▶️ Running the Analysis

1. Once funds are added, click **🔄 Run Analysis on Portfolio** in the sidebar
2. A live status log will show data fetching and computation progress
3. Analysis is **cached** per portfolio + cutoff date combination — reruns only when either changes

> ⚠️ The Run button is disabled until at least one fund is added to the portfolio.

---

## 📅 ATH Cutoff Date

- Located at the **bottom of the sidebar**
- Set a cutoff date to define the starting point for NAV % change calculation
- Default date is pre-configured in the app
- All ATH change metrics in **Tab 2** update based on this date

---

## 📋 Tabs Overview

### Tab 1 – 📋 Summary
- View a full table of all funds with: CAGR (`returns_%`), 36-month rolling return (`roll_36`), and all ratios
- **Filter by Category** using the dropdown
- **Top Performers** section highlights:
  - Best CAGR, Best Sharpe, Best Alpha, Lowest Beta
  - Best Info Ratio, Highest Up Capture, Lowest Down Capture

---

### Tab 2 – 📉 ATH Change
- Scatter plot showing **% change from the selected cutoff date** to today
- Funds are **color-coded**:

| Color | Range | Meaning |
|-------|-------|---------|
| 🟢 Green | > +5% | Strong recovery |
| 🟣 Magenta | 0% to +5% | Mild recovery |
| 🟠 Orange | -5% to 0% | Mild decline |
| 🔴 Red | < -5% | Significant fall |

- Funds above 0% have recovered better than market peers

---

### Tab 3 – 📊 Ratios
- Select a ratio from the dropdown to view a **horizontal bar chart** grouped by category
- Ratio descriptions:

| Ratio | Description |
|-------|-------------|
| `beta` | Sensitivity to market. < 1 = less volatile |
| `sharpe` | Risk-adjusted return. > 1 is good |
| `alpha` | Excess return over expected market return |
| `info` | Consistency of outperformance over benchmark |
| `up_capture` | % of benchmark gains captured in rising markets |
| `down_capture` | % of benchmark losses suffered in falling markets |

- **🕸 Radar Chart** — Select multiple funds to compare all ratios side-by-side simultaneously

---

### Tab 4 – 🔄 Rolling Ratios
- View **18-month rolling** ratio trends for a selected category
- Select a **Category** and a **Ratio** to plot
- Useful for identifying consistency vs. one-time outperformance over time

---

## 💾 Funds List Management

| Action | How |
|--------|-----|
| Remove a single fund | Click 🗑 next to the fund name in the sidebar |
| Remove all funds | Click **🗑 Clear All** in the sidebar |
| Export portfolio | Click **⬇️ CSV** to download as `my_portfolio.csv` |
| Restore session | Bookmark the URL — it contains `?sid=` to restore your portfolio |

---

## 💡 Tips

- Your portfolio is **auto-saved** per session via a unique session ID embedded in the URL (`?sid=`)
- Bookmark the URL to **restore your portfolio** in a future visit
- **Categories** determine which benchmark index is used for capture ratios and alpha calculations
- The **Radar Chart** is ideal for a quick multi-metric comparison across selected funds
- The **Rolling Ratios** tab helps assess if a fund's performance is consistent or just a recent spike