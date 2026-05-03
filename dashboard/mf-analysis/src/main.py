import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from mftool import Mftool
from functools import reduce


# ── Config ────────────────────────────────────────────────────────────────────
PERIOD       = '5y'
NUM_YEARS    = int(PERIOD[0])
COUNT        = 252 * NUM_YEARS
ALL_MFs      = False
DATA_PATH      = os.path.join(os.path.dirname(__file__), "../data/mf_ga_amfi.csv")
PORTFOLIO_PATH = os.path.join(os.path.dirname(__file__), "../data/portfolio.csv")

DEFAULT_CUTOFF = '2024-09-26'       # ← default ATH cutoff date

RISK_FREE_MF = '0P0000XV6E.BO'
MARKET       = '^NSEI'

RATIO_NAMES  = ['beta', 'sharpe', 'alpha', 'info', 'up_capture', 'down_capture']

BENCHMARK_IDX = {
    'name': ['large', 'mid', 'small', 'flexi'],
    'id':   ['large_3', 'mid_2', 'small_1', 'flexi_4'],
    'code': [118741, 147622, 148519, 147625],
}

mf = Mftool()


# ── Reference Data ────────────────────────────────────────────────────────────
def load_reference_data():
    print("Fetching risk-free and market data...")
    risk_free_hist = yf.Ticker(RISK_FREE_MF).history(period=PERIOD)
    market_hist    = yf.Ticker(MARKET).history(period=PERIOD)

    risk_free_hist['DATE'] = risk_free_hist.index.strftime('%Y-%m-%d')
    market_hist['DATE']    = market_hist.index.strftime('%Y-%m-%d')

    riskfree_market_df = pd.merge(
        risk_free_hist[['DATE', 'Close']],
        market_hist[['DATE', 'Close']],
        suffixes=('_risk', '_market'),
        on='DATE', how='inner'
    )
    return riskfree_market_df


def load_benchmark_data():
    print("Fetching benchmark data...")
    benchmark_df = []
    for idx, code in zip(BENCHMARK_IDX['id'], BENCHMARK_IDX['code']):
        df = mf.get_scheme_historical_nav(code, as_Dataframe=True) \
               .iloc[::-1].iloc[-COUNT:].astype(float)
        df.rename(columns={'nav': idx + '_Close'}, inplace=True)
        df.index = pd.to_datetime(df.index, format="%d-%m-%Y").strftime('%Y-%m-%d')
        benchmark_df.append(df.reset_index()[['date', idx + '_Close']])
    return reduce(
        lambda left, right: pd.merge(left, right, on='date', how='outer'),
        benchmark_df
    )


# ── Ratios ────────────────────────────────────────────────────────────────────
def get_ratios(hist_df, mf_cat, riskfree_market_df, benchmark_df):
    hist_df = hist_df.copy()
    hist_df['DATE'] = hist_df.index.strftime('%Y-%m-%d')

    merged = pd.merge(riskfree_market_df, hist_df[['DATE', 'Close']], on='DATE', how='inner')
    returns_df = merged.groupby('DATE')[['Close_risk', 'Close_market', 'Close']] \
                       .mean().pct_change().dropna()
    returns_df.rename(columns={
        'Close_risk':   'risk_free_return',
        'Close_market': 'market_return',
        'Close':        'weekly_return'
    }, inplace=True)

    # Beta
    cov  = np.cov(returns_df['weekly_return'], returns_df['market_return'])
    beta = cov[0, 1] / cov[1, 1]

    # Alpha
    expected_return = (returns_df['risk_free_return']
                       + beta * (returns_df['market_return'] - returns_df['risk_free_return']))
    alpha = returns_df['weekly_return'] - expected_return

    # Sharpe
    excess_rf   = returns_df['weekly_return'] - returns_df['risk_free_return']
    sharpe      = np.sqrt(252) * excess_rf.mean() / excess_rf.std()

    # Info
    excess_mkt  = returns_df['weekly_return'] - returns_df['market_return']
    info        = np.sqrt(252) * excess_mkt.mean() / excess_mkt.std()

    # Capture Ratios
    cat = mf_cat if mf_cat in BENCHMARK_IDX['id'] else 'large_3'
    bench_col = cat + '_Close'
    hist_bench = pd.merge(
        benchmark_df[['date', bench_col]].rename(columns={'date': 'DATE'}),
        hist_df, on='DATE', how='inner'
    )
    # Month-on-Month Returns in %
    hist_bench.index = pd.to_datetime(hist_bench['DATE'], format="%Y-%m-%d")
    monthly = hist_bench.resample('ME').last()[[bench_col, 'Close']].pct_change().dropna()
    # Upside and Downside
    up   = monthly[monthly[bench_col] >= 0]
    down = monthly[monthly[bench_col] <  0]
    # Calculate the cumulative product (1 + return_1) * (1 + return_2) * ... - 1
    up_fund_cumu    = (1 + up['Close']).prod()   - 1
    up_bench_cumu   = (1 + up[bench_col]).prod() - 1
    down_fund_cumu  = (1 + down['Close']).prod()   - 1
    down_bench_cumu = (1 + down[bench_col]).prod() - 1

    up_capture   = (up_fund_cumu   / up_bench_cumu)   * 100 if up_bench_cumu   != 0 else np.nan
    down_capture = (down_fund_cumu / down_bench_cumu) * 100 if down_bench_cumu != 0 else np.nan

    return {
        'sharpe':       round(sharpe, 2),
        'beta':         round(beta, 2),
        'alpha':        round(alpha.mean() * 100, 2),
        'info':         round(info, 2),
        'up_capture':   round(up_capture, 2),
        'down_capture': round(down_capture, 2),
    }


# ── Rolling Ratios ────────────────────────────────────────────────────────────
ROLL_LENGTHS = [18]

def rolling_results(hist_df, mf_cat, riskfree_market_df, benchmark_df):
    hist_df = hist_df.copy()
    hist_df['Month'] = hist_df.index.strftime('%Y-%m')
    months = hist_df['Month'].unique()

    roll_result_df = pd.DataFrame(columns=['start_month', 'roll_length'] + RATIO_NAMES)
    for roll_len in ROLL_LENGTHS:
        for i in range(len(months) - roll_len + 1):
            roll_months = months[i:i + roll_len]
            roll_df     = hist_df[hist_df['Month'].isin(roll_months)].copy()
            vals        = get_ratios(roll_df, mf_cat, riskfree_market_df, benchmark_df)
            roll_result_df.loc[len(roll_result_df)] = (
                [roll_df['Month'].iloc[0], roll_len] + list(vals.values())
            )
    return roll_result_df.to_dict(orient='list')


def rolling_cagr_median(df, roll_lengths=[12], years=1):
    """
    Calculate rolling CAGR and return median for each roll length.
    
    Args:
        df: DataFrame with 'Close' column and DatetimeIndex
        roll_lengths: list of rolling window lengths in months
        years: number of years for CAGR calculation
    
    Returns:
        dict: {roll_len: median_cagr} for each roll length
    """
    df = df.copy()
    df['Month'] = df.index.strftime('%Y-%m')
    months = df['Month'].unique()

    medians = {}
    for roll_len in roll_lengths:
        rolling_returns = []
        for i in range(len(months) - roll_len + 1):
            roll_months = months[i:i + roll_len]
            roll_df = df[df['Month'].isin(roll_months)]
            cagr = (roll_df['Close'].iloc[-1] / roll_df['Close'].iloc[0]) ** (1 / years) - 1
            rolling_returns.append(cagr * 100)
        medians[f'roll_{roll_len}'] = round(np.median(rolling_returns), 2)

    return medians


# ── Portfolio / All MFs ───────────────────────────────────────────────────────
def load_mf_df(csv_path: str = None):
    path = csv_path or DATA_PATH
    mf_df = pd.read_csv(path)

    if not ALL_MFs:
        mf_df = pd.concat([mf_df, pd.DataFrame.from_dict(BENCHMARK_IDX)], ignore_index=True)
        return mf_df, None

    # All MFs mode
    names = ['small_1', 'mid_2', 'large_3', 'flexi_4']
    cat_idx = 1
    mf_dict = mf.get_available_schemes(names[cat_idx].split('_')[0])
    mf_dict_ref = {k: v for k, v in mf_dict.items()
                   if all(w in v.lower() for w in ['direct', 'growth'])}
    if names[cat_idx] == 'mid_2':
        mf_dict_ref = {k: v for k, v in mf_dict_ref.items()
                       if 'large' not in v.lower() and 'small' not in v.lower()}
    if names[cat_idx] == 'large_3':
        mf_dict_ref = {k: v for k, v in mf_dict_ref.items()
                       if 'mid' not in v.lower()}

    my_mf_names = (mf_df[mf_df['id'] == names[cat_idx]]['name']
                   .apply(lambda x: ' '.join(x.split(' ')[:2])).str.lower().tolist())

    all_df = pd.DataFrame({'name': list(mf_dict_ref.values()),
                           'id':   names[cat_idx],
                           'code': list(mf_dict_ref.keys())})
    all_df['short_name'] = all_df['name'].apply(lambda x: ' '.join(x.split(' ')[:2])).str.lower()
    my_codes = all_df[all_df['short_name'].isin(my_mf_names)]['code'].tolist()
    all_df['portfolio'] = all_df['code'].apply(lambda x: 1 if x in my_codes else 0)
    return all_df, all_df[all_df['portfolio'] == 1]['name'].values


# ── Main Analysis ─────────────────────────────────────────────────────────────
def run_analysis(csv_path: str = None, cutoff: str = DEFAULT_CUTOFF):
    riskfree_market_df = load_reference_data()
    benchmark_df       = load_benchmark_data()
    mf_df, my_mf_names = load_mf_df(csv_path)

    mf_fall        = {}
    latest         = []
    ratios         = {k: [] for k in RATIO_NAMES}
    rolling_r      = {k: [] for k in RATIO_NAMES}
    returns_report = {}
    rolling_returns = {}

    total = len(mf_df)
    for i, (idx, row) in enumerate(mf_df.iterrows(), 1):
        print(f"[{i}/{total}] Analysing: {row['name']}")
        try:
            df = mf.get_scheme_historical_nav(row['code'], as_Dataframe=True)
            df.rename(columns={'nav': 'Close'}, inplace=True)
            df = df.iloc[::-1].iloc[-COUNT:]
            df['Close'] = df['Close'].astype(float)
            df.index    = pd.to_datetime(df.index, format="%d-%m-%Y")

            latest.append(df['Close'].iloc[-1])

            # ── ATH cutoff slice using passed cutoff date ──────────────────
            cutoff_ts = pd.Timestamp(cutoff)
            before_cutoff = df[df.index <= cutoff_ts]
            if not before_cutoff.empty:
                mf_fall[row['name']] = df['Close'].loc[before_cutoff.index[-1]:].values
            else:
                mf_fall[row['name']] = df['Close'].values

            vals = get_ratios(df.copy(), row['id'], riskfree_market_df, benchmark_df)
            for k in RATIO_NAMES:
                ratios[k].append(vals[k])

            roll_vals = rolling_results(df.copy(), row['id'], riskfree_market_df, benchmark_df)
            for k in RATIO_NAMES:
                rolling_r[k].append(roll_vals[k])

            if df.shape[0] == COUNT:
                cagr = (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (1 / NUM_YEARS) - 1
                returns_report[row['name']] = round(cagr * 100, 2)
            else:
                returns_report[row['name']] = np.nan

            #  1 year Rolling returns
            rolling_returns[row['name']] = rolling_cagr_median(df.copy(), [36], years=3)

        except Exception as e:
            print(f"  ⚠ Skipped {row['name']}: {e}")
            latest.append(np.nan)
            mf_fall[row['name']] = np.array([])
            for k in RATIO_NAMES:
                ratios[k].append(np.nan)
                rolling_r[k].append([])
            returns_report[row['name']] = np.nan

    for k in RATIO_NAMES:
        mf_df[k] = ratios[k]
    mf_df['nav']       = latest
    mf_df['returns_%'] = mf_df['name'].map(returns_report)
    for k in RATIO_NAMES:
        mf_df[f'rolling_{k}'] = rolling_r[k]

    # 1 year Rolling returns
    rolling_keys = list(next(iter(rolling_returns.values())).keys())
    for k in rolling_keys:
        mf_df[k] = mf_df['name'].map(lambda x: rolling_returns.get(x, {}).get(k, np.nan))

    return mf_df, mf_fall, my_mf_names


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_ath_change(mf_fall, my_mf_names):
    change_list, names = [], []
    for name, vals in mf_fall.items():
        if len(vals) < 2:
            continue
        change_list.append(round((vals[-1] - vals[0]) / vals[0] * 100, 2))
        names.append(name)

    names, change_list = zip(*sorted(zip(names, change_list), key=lambda x: x[1]))

    plt.figure(figsize=(8, 8))
    plt.plot(change_list, names, marker='o')
    ax = plt.gca()
    for i, label in enumerate(ax.get_yticklabels()):
        v = change_list[i]
        if   v > 5:            label.set_color('green')
        elif 0 < v <= 5:       label.set_color('blue')
        elif -5 < v <= 0:      label.set_color('magenta')
        elif -10 < v <= -5:    label.set_color('maroon')
        if label.get_text() in BENCHMARK_IDX['name']:
            label.set_fontweight('bold'); label.set_color('black')
        if ALL_MFs and my_mf_names is not None and label.get_text() in my_mf_names:
            label.set_fontweight('bold')
    plt.grid(True)
    plt.title('% Change from ATH (26-Sep-2024)')
    plt.tight_layout()
    plt.savefig('output/ath_change.png', dpi=150)
    plt.show()


def plot_ratios(mf_df, my_mf_names):
    for ratio in RATIO_NAMES:
        plt.figure(figsize=(6, 10))
        for cat, df in mf_df.groupby('id'):
            df = df.sort_values(by=ratio)
            plt.barh(df['name'], df[ratio], label=cat)
            plt.barh('----' + cat + '----', 0)
        plt.title(ratio)
        plt.grid(True, axis='x')
        plt.legend()
        ax = plt.gca()
        for label in ax.get_yticklabels():
            if label.get_text() in BENCHMARK_IDX['name']:
                label.set_fontweight('bold'); label.set_color('maroon')
            if ALL_MFs and my_mf_names is not None and label.get_text() in my_mf_names:
                label.set_fontweight('bold')
        plt.tight_layout()
        plt.savefig(f'output/ratio_{ratio}.png', dpi=150)
        plt.show()


def plot_rolling_ratios(mf_df):
    for cat, df in mf_df.groupby('id'):
        fig, axes = plt.subplots(1, len(RATIO_NAMES), figsize=(18, 4))
        fig.suptitle(cat, fontsize=16)
        for i, ratio in enumerate(RATIO_NAMES):
            for _, row in df.iterrows():
                axes[i].plot(row[f'rolling_{ratio}'],
                             label=row['name'].split(' ')[0])
            axes[i].set_title(ratio)
            axes[i].legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(f'output/rolling_{cat}.png', dpi=150)
        plt.show()


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', type=str, default=DEFAULT_CUTOFF,
                        help='ATH cutoff date in YYYY-MM-DD format')
    args = parser.parse_args()

    os.makedirs(os.path.join(os.path.dirname(__file__), '../output'), exist_ok=True)
    mf_df, mf_fall, my_mf_names = run_analysis(cutoff=args.cutoff)

    print("\n=== Summary ===")
    print(mf_df[['name', 'id', 'returns_%'] + RATIO_NAMES].round(2).to_string())

    # Save results
    out_csv = os.path.join(os.path.dirname(__file__), '../output/mf_results.csv')
    mf_df[['name', 'id', 'code', 'nav', 'returns_%', 'roll_12'] + RATIO_NAMES].to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    plot_ath_change(mf_fall, my_mf_names)
    plot_ratios(mf_df, my_mf_names)
    plot_rolling_ratios(mf_df)