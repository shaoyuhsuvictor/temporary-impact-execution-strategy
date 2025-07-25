import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def plot_empirical_impact(x_vals, g_vals, title='Empirical Impact Curve', side='buy'):
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, g_vals, label=f'{side.capitalize()} side impact', linewidth=2)
    plt.xlabel("Order size (shares)")
    plt.ylabel("Slippage ($)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()