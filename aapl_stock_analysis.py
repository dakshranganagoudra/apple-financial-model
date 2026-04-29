"""
AAPL 5-Year Stock Performance Dashboard
========================================
Pulls data via yfinance and generates a professional
multi-panel chart saved as a PNG.

Usage:
    pip install yfinance matplotlib pandas
    python aapl_stock_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np

# ── 1. FETCH DATA ─────────────────────────────────────────────────────────────
TICKER = "AAPL"
PERIOD = "5y"
INTERVAL = "1wk"          # weekly candles for a clean 5-year view

print(f"Fetching {TICKER} data ({PERIOD}, {INTERVAL})...")
stock = yf.Ticker(TICKER)
df = stock.history(period=PERIOD, interval=INTERVAL)

# Drop any rows where Close is NaN
df = df.dropna(subset=["Close"])

# Compute derived series
df["MA_50"]  = df["Close"].rolling(window=50 // 5).mean()   # ~50-day via weekly
df["MA_200"] = df["Close"].rolling(window=200 // 5).mean()  # ~200-day via weekly

# Normalised price (rebased to 100)
df["Indexed"] = df["Close"] / df["Close"].iloc[0] * 100

# % return from start
total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
annualized   = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (1 / 5) - 1) * 100
current_price = df["Close"].iloc[-1]
peak_price    = df["Close"].max()
drawdown_pct  = (current_price / peak_price - 1) * 100

# Volume moving average
df["Vol_MA"] = df["Volume"].rolling(window=4).mean()

# RSI (14-period)
delta = df["Close"].diff()
gain  = delta.clip(lower=0)
loss  = -delta.clip(upper=0)
avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()
rs  = avg_gain / avg_loss.replace(0, np.nan)
df["RSI"] = 100 - 100 / (1 + rs)

# Annual returns (calendar year)
df_daily = stock.history(period=PERIOD, interval="1d").dropna(subset=["Close"])
df_daily["Year"] = df_daily.index.year
annual = (
    df_daily.groupby("Year")["Close"]
    .apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
    .reset_index()
)
annual.columns = ["Year", "Return"]

print(f"  Latest price : ${current_price:.2f}")
print(f"  5-Year return: {total_return:.1f}%  |  CAGR: {annualized:.1f}%")
print(f"  Current drawdown from peak: {drawdown_pct:.1f}%")

# ── 2. STYLING ────────────────────────────────────────────────────────────────
APPLE_DARK  = "#1D1D1F"
APPLE_BLUE  = "#0071E3"
APPLE_GREEN = "#30D158"
APPLE_RED   = "#FF453A"
APPLE_GRAY  = "#8E8E93"
BG_COLOR    = "#F5F5F7"
PANEL_COLOR = "#FFFFFF"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.facecolor":  PANEL_COLOR,
    "figure.facecolor": BG_COLOR,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.edgecolor":  "#D1D1D6",
    "axes.linewidth":  0.8,
    "xtick.color":     APPLE_GRAY,
    "ytick.color":     APPLE_GRAY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "grid.color":      "#E5E5EA",
    "grid.linewidth":  0.6,
    "grid.alpha":      0.8,
})

# ── 3. BUILD FIGURE ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor=BG_COLOR)
fig.suptitle(
    f"APPLE INC. (AAPL) — 5-Year Performance Dashboard",
    fontsize=16, fontweight="bold", color=APPLE_DARK, y=0.98
)

gs = GridSpec(
    4, 2,
    figure=fig,
    hspace=0.50,
    wspace=0.30,
    top=0.93, bottom=0.06,
    left=0.07, right=0.97,
    height_ratios=[3, 1.2, 1.2, 1.2],
)

dates = df.index

# ── Panel 1: Price + MAs (spans full width) ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])

ax1.fill_between(dates, df["Close"], df["Close"].min() * 0.97,
                 alpha=0.12, color=APPLE_BLUE, zorder=1)
ax1.plot(dates, df["Close"], color=APPLE_BLUE, linewidth=1.8,
         label=f"Close  ${current_price:.2f}", zorder=3)
ax1.plot(dates, df["MA_50"],  color="#FF9F0A", linewidth=1.2,
         linestyle="--", label="~50-Day MA", zorder=2)
ax1.plot(dates, df["MA_200"], color=APPLE_RED, linewidth=1.2,
         linestyle="--", label="~200-Day MA", zorder=2)

ax1.set_title("Price History with Moving Averages", fontsize=10,
              color=APPLE_GRAY, pad=6)
ax1.set_ylabel("Price (USD)", fontsize=9, color=APPLE_GRAY)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax1.legend(loc="upper left", fontsize=8, framealpha=0.8)
ax1.grid(True, axis="y")

# Annotate peak
peak_idx = df["Close"].idxmax()
ax1.annotate(
    f"  Peak\n  ${peak_price:.2f}",
    xy=(peak_idx, peak_price),
    xytext=(peak_idx, peak_price * 1.06),
    fontsize=7.5, color=APPLE_GRAY,
    arrowprops=dict(arrowstyle="->", color=APPLE_GRAY, lw=0.8),
)

# ── Panel 2: Volume ───────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
colors = [APPLE_GREEN if c >= o else APPLE_RED
          for c, o in zip(df["Close"], df["Open"])]
ax2.bar(dates, df["Volume"] / 1e6, width=5, color=colors, alpha=0.7, zorder=2)
ax2.plot(dates, df["Vol_MA"] / 1e6, color=APPLE_DARK, linewidth=1,
         label="4-Wk Avg")
ax2.set_title("Weekly Trading Volume", fontsize=9, color=APPLE_GRAY, pad=4)
ax2.set_ylabel("Volume (M shares)", fontsize=8, color=APPLE_GRAY)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.legend(fontsize=7, framealpha=0.8)
ax2.grid(True, axis="y")

# ── Panel 3: RSI ──────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
rsi_vals = df["RSI"].dropna()
ax3.plot(rsi_vals.index, rsi_vals, color=APPLE_BLUE, linewidth=1.2, zorder=3)
ax3.fill_between(rsi_vals.index, rsi_vals, 50, where=rsi_vals > 50,
                 alpha=0.15, color=APPLE_GREEN, zorder=2)
ax3.fill_between(rsi_vals.index, rsi_vals, 50, where=rsi_vals < 50,
                 alpha=0.15, color=APPLE_RED, zorder=2)
ax3.axhline(70, color=APPLE_RED,   linewidth=0.8, linestyle="--", alpha=0.7)
ax3.axhline(30, color=APPLE_GREEN, linewidth=0.8, linestyle="--", alpha=0.7)
ax3.axhline(50, color=APPLE_GRAY,  linewidth=0.6, linestyle=":")
ax3.set_ylim(0, 100)
ax3.set_title("Relative Strength Index (RSI-14)", fontsize=9, color=APPLE_GRAY, pad=4)
ax3.set_ylabel("RSI", fontsize=8, color=APPLE_GRAY)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.text(dates[-1], 72, " Overbought (70)", fontsize=7, color=APPLE_RED, va="bottom")
ax3.text(dates[-1], 28, " Oversold (30)",   fontsize=7, color=APPLE_GREEN, va="top")
ax3.grid(True, axis="y")

# ── Panel 4: Annual Returns Bar Chart ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
bar_colors = [APPLE_GREEN if r >= 0 else APPLE_RED for r in annual["Return"]]
bars = ax4.bar(annual["Year"].astype(str), annual["Return"], color=bar_colors,
               alpha=0.85, zorder=2)
ax4.axhline(0, color=APPLE_DARK, linewidth=0.8)
ax4.set_title("Annual Total Return (%)", fontsize=9, color=APPLE_GRAY, pad=4)
ax4.set_ylabel("Return (%)", fontsize=8, color=APPLE_GRAY)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax4.grid(True, axis="y", zorder=1)
for bar, val in zip(bars, annual["Return"]):
    ypos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 3
    ax4.text(bar.get_x() + bar.get_width() / 2, ypos,
             f"{val:.1f}%", ha="center", fontsize=7.5,
             color=APPLE_GREEN if val >= 0 else APPLE_RED, fontweight="bold")

# ── Panel 5: Indexed Return Comparison ───────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])

# Fetch SPY for comparison
spy = yf.Ticker("SPY").history(period=PERIOD, interval=INTERVAL).dropna(subset=["Close"])
spy_idx = spy["Close"] / spy["Close"].iloc[0] * 100
aapl_common = df["Indexed"].reindex(df.index)

ax5.plot(df.index, df["Indexed"], color=APPLE_BLUE, linewidth=1.5, label="AAPL")
ax5.plot(spy.index, spy_idx, color=APPLE_GRAY, linewidth=1.2,
         linestyle="--", label="S&P 500 (SPY)")
ax5.axhline(100, color=APPLE_DARK, linewidth=0.6, linestyle=":")
ax5.set_title("AAPL vs. S&P 500 (Rebased to 100)", fontsize=9, color=APPLE_GRAY, pad=4)
ax5.set_ylabel("Indexed Return", fontsize=8, color=APPLE_GRAY)
ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax5.xaxis.set_major_locator(mdates.YearLocator())
ax5.legend(fontsize=8, framealpha=0.8)
ax5.grid(True, axis="y")

# ── Panel 6: Key Stats Summary Box ───────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, :])
ax6.axis("off")

info = stock.info
market_cap = info.get("marketCap", 0) / 1e12
pe_ratio   = info.get("trailingPE", float("nan"))
div_yield  = (info.get("dividendYield", 0) or 0) * 100
week_high  = info.get("fiftyTwoWeekHigh", float("nan"))
week_low   = info.get("fiftyTwoWeekLow",  float("nan"))

stats = [
    ("Current Price",         f"${current_price:.2f}"),
    ("5-Year Total Return",   f"{total_return:.1f}%"),
    ("5-Year CAGR",           f"{annualized:.1f}%"),
    ("Market Cap",            f"${market_cap:.2f}T"),
    ("P/E Ratio (TTM)",       f"{pe_ratio:.1f}x" if pd.notna(pe_ratio) else "N/A"),
    ("Dividend Yield",        f"{div_yield:.2f}%"),
    ("52-Week High",          f"${week_high:.2f}"),
    ("52-Week Low",           f"${week_low:.2f}"),
    ("Peak (5-Year)",         f"${peak_price:.2f}"),
    ("Drawdown from Peak",    f"{drawdown_pct:.1f}%"),
]

n_cols = 5
for idx, (k, v) in enumerate(stats):
    col_x = (idx % n_cols) / n_cols + 0.01
    row_y = 0.80 if idx < n_cols else 0.20
    ax6.text(col_x, row_y + 0.10, k, fontsize=8, color=APPLE_GRAY,
             transform=ax6.transAxes)
    color = APPLE_GREEN if "Return" in k or "CAGR" in k else (
            APPLE_RED if "Drawdown" in k else APPLE_DARK)
    ax6.text(col_x, row_y - 0.08, v, fontsize=10.5, fontweight="bold",
             color=color, transform=ax6.transAxes)

ax6.set_facecolor(LIGHT_BLUE := "#EEF4FF")

# Footer
fig.text(
    0.5, 0.01,
    f"Data: Yahoo Finance via yfinance  |  Model: github.com/[YourHandle]/aapl-financial-model  |  Not investment advice.",
    ha="center", fontsize=7.5, color=APPLE_GRAY, style="italic"
)

# ── 4. SAVE ───────────────────────────────────────────────────────────────────
out_path = "/home/claude/AAPL_Stock_Dashboard.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Chart saved → {out_path}")
plt.close()
