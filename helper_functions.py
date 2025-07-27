import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

def get_lob_snapshots(ticker_dir, freq='1min'):
    dfs = []

    for file in ticker_dir.glob("*.csv"):
        cols = pd.read_csv(file, nrows=0).columns
        required_cols = [
            col for col in cols 
            if col == "ts_event" or col.startswith(("ask_px_", "ask_sz_", "bid_px_", "bid_sz_"))
            ]
        df = pd.read_csv(file, usecols=required_cols)

        df['ts_event'] = pd.to_datetime(df['ts_event'], format='mixed')
        df.set_index('ts_event', inplace=True)
        df = df.resample("1min").first().ffill().reset_index()
        df['ts_event'] = df['ts_event'].dt.tz_localize(None)    # remove timezone

        mid_price = (df['ask_px_00'] + df['bid_px_00']) / 2
        df.insert(1, 'mid_px', mid_price)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True).sort_values("ts_event").reset_index(drop=True)

def compute_empirical_impact(snapshot, side='buy', num_points=100):
    def compute_vwap(x, book):
        remaining = x
        cost = 0
        for price, size in book:
            trade_qty = min(size, remaining)
            cost += trade_qty * price
            remaining -= trade_qty
            if remaining == 0:
                break
        return cost / x if remaining == 0 else None

    if side == 'buy':
        px_cols = [col for col in snapshot.index if col.startswith("ask_px_")]
        sz_cols = [col for col in snapshot.index if col.startswith("ask_sz_")]
    elif side == 'sell':
        px_cols = [col for col in snapshot.index if col.startswith("bid_px_")]
        sz_cols = [col for col in snapshot.index if col.startswith("bid_sz_")]
    else:
        raise ValueError("side must be 'buy' or 'sell'")

    book = [(snapshot[px], snapshot[sz]) for px, sz in zip(px_cols, sz_cols)]
    book = [(p, s) for p, s in book if pd.notnull(p) and pd.notnull(s) and s > 0]

    mid = snapshot['mid_px']

    max_qty = sum(s for _, s in book)
    x_range = np.linspace(max_qty / num_points, max_qty, num_points)
    
    x_vals = []
    g_vals = []
    for x in x_range:
        vwap = compute_vwap(x, book)
        if vwap is not None:
            slippage = vwap - mid if side == 'buy' else mid - vwap
            x_vals.append(x)
            g_vals.append(slippage)

    return np.array(x_vals), np.array(g_vals)

def plot_impact(ax, x_vals, g_vals, label='', linestyle='-', linewidth=1, alpha=1, color='tab:blue', title=''):
    ax.plot(x_vals, g_vals, label=label, linestyle=linestyle, linewidth=linewidth, alpha=alpha, color=color)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Order size (shares)")
    ax.set_ylabel("Slippage ($)")
    ax.grid(True, linestyle='--', alpha=0.5)

def sample_snapshot(df, rows_per_day=390, day_indices=[0, 5, 10, 15, 20], time_offsets = list(range(0, 390, 30)) + [389]):
    row_indices = [d * rows_per_day + t for d in day_indices for t in time_offsets]
    return df.iloc[row_indices].reset_index(drop=True)

def linear_model(x, beta):
    return beta * x

def power_law_model(x, alpha, delta):
        with np.errstate(over='ignore', invalid='ignore'):
            return alpha * x ** delta

def quadratic_model(x, alpha, beta):
        return alpha * x + beta * x ** 2

def fit_impact_model(x_vals, g_vals, model_func, p0, bounds=(-np.inf, np.inf)):
    popt, _ = curve_fit(model_func, x_vals, g_vals, p0=p0, bounds=bounds, maxfev=10000)
    g_fit = model_func(x_vals, *popt)

    mse = mean_squared_error(g_vals, g_fit)
    r2 = r2_score(g_vals, g_fit)

    return popt, mse, r2, g_fit

def shifted_power_law(x, gamma, alpha, delta):
        with np.errstate(over='ignore', invalid='ignore'):
            return gamma + alpha * x ** delta