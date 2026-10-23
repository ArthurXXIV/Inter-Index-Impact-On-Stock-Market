"""Run every experiment and write results/ tables and images/ figures.

    .venv/bin/python run_experiments.py            # uses cached data/
    .venv/bin/python run_experiments.py --refresh  # re-download from Yahoo
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from interindex.data import INDICES, load_prices
from interindex.experiment import run
from interindex.features import build_panel, technical_indicators
from interindex.models import Ridge, RecurrentModel, all_models

ROOT = Path(__file__).resolve().parent
RESULTS, IMAGES = ROOT / "results", ROOT / "images"
MARKETS = list(INDICES)

# Chart styling: recessive axes, fixed categorical order, one-hue sequential ramp
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
SERIES = {"Actual": "#2a78d6", "Ridge": "#eb6834", "LSTM": "#1baf7a", "GRU": "#eda100",
          "Zero": "#52514e", "Mean": "#52514e"}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def correlation_figures(panel):
    rets = panel[[f"{m}_Return" for m in MARKETS]].set_axis(MARKETS, axis=1)
    corr = rets.corr()
    # lead_lag[a][b] = corr(a's return this week, b's return next week)
    lead_lag = pd.DataFrame({b: {a: rets[a].corr(rets[b].shift(-1)) for a in MARKETS} for b in MARKETS})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [(corr, "Same-week correlation of returns", 0.4, 1.0),
              (lead_lag, "This week (row) vs next week (column)", -0.3, 0.3)]
    for ax, (m, title, lo, hi) in zip(axes, panels):
        cmap = BLUES if lo >= 0 else LinearSegmentedColormap.from_list(
            "div", ["#eb6834", "#f0efec", "#2a78d6"])
        im = ax.imshow(m.values, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_xticks(range(4), MARKETS)
        ax.set_yticks(range(4), MARKETS)
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                v = m.values[i, j]
                dark = abs(v - lo) / (hi - lo) > 0.6 if lo >= 0 else abs(v) > 0.2
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if dark else INK, fontsize=11)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(IMAGES / "correlations.png", dpi=150)
    plt.close(fig)
    corr.round(3).to_csv(RESULTS / "correlation_same_week.csv")
    lead_lag.round(3).to_csv(RESULTS / "correlation_lead_lag.csv")


def indicator_figure(prices):
    df = prices["UK"]
    ind = technical_indicators(df)
    mid, sd = df["Close"].rolling(20).mean(), df["Close"].rolling(20).std()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.fill_between(df.index, mid - 2 * sd, mid + 2 * sd, color="#cde2fb", lw=0, label="Bollinger band (20, 2σ)")
    ax1.plot(df.index, df["Close"], color="#2a78d6", lw=2, label="FTSE 100 close")
    ax1.set_title("FTSE 100 weekly close with Bollinger band, and RSI")
    ax1.legend(frameon=False, loc="upper left")
    ax2.plot(ind.index, ind["RSI"], color="#eb6834", lw=2)
    for lvl in (30, 70):
        ax2.axhline(lvl, color=INK_2, lw=1, ls="--")
    ax2.set_ylabel("RSI (14)")
    fig.tight_layout()
    fig.savefig(IMAGES / "ftse_indicators.png", dpi=150)
    plt.close(fig)


def r2_figure(table, path, title, legend_loc="lower right"):
    """Grouped horizontal bars: out-of-sample R² per target, per model."""
    models = ["Ridge", "LSTM", "GRU"]
    t = table[table.Model.isin(models)].pivot(index="Target", columns="Model", values="R2_OOS")
    t = t.loc[MARKETS[::-1], models]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    h = 0.26
    for k, m in enumerate(models):
        y = np.arange(len(t)) + (1 - k) * h
        ax.barh(y, t[m], height=h - 0.03, color=SERIES[m], label=m)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(t)), t.index)
    ax.set_xlabel("Out-of-sample R²  (0 = no better than predicting the average)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc=legend_loc)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def prediction_figure(pred, path, title):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(pred.index, pred["Actual"], color=SERIES["Actual"], lw=2, marker="o", ms=4, label="Actual")
    ax.plot(pred.index, pred["Ridge"], color=SERIES["Ridge"], lw=2, label="Ridge prediction")
    ax.plot(pred.index, pred["LSTM"], color=SERIES["LSTM"], lw=2, label="LSTM prediction")
    ax.set_ylabel("Weekly return (%)")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(refresh=False):
    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    prices = load_prices(refresh=refresh)
    panel = build_panel(prices)
    print(f"Panel: {len(panel)} weeks, {panel.index[0]:%Y-%m-%d} to {panel.index[-1]:%Y-%m-%d}")

    correlation_figures(panel)
    indicator_figure(prices)

    rows = []
    # 1. Co-movement: all other markets -> target, same week
    for target in MARKETS:
        others = [m for m in MARKETS if m != target]
        r, pred = run(panel, target, others, "same_week", all_models())
        rows += r
        if target == "India":
            prediction_figure(pred, IMAGES / "same_week_india.png",
                              "India weekly return vs prediction from US, UK and Japan (test period)")
        print(f"same_week  {target:<6} done")

    # 2. Which foreign markets explain India best?
    foreign = [m for m in MARKETS if m != "India"]
    for k in (1, 2):
        for combo in combinations(foreign, k):
            rows += run(panel, "India", list(combo), "same_week",
                        [Ridge(), RecurrentModel("LSTM"), RecurrentModel("GRU")])[0]
    print("India subsets done")

    # 3. Forecasting next week: own history only vs all four markets
    for target in MARKETS:
        rows += run(panel, target, [target], "next_week",
                    [Ridge(), RecurrentModel("LSTM"), RecurrentModel("GRU")])[0]
        r, pred = run(panel, target, MARKETS, "next_week", all_models())
        rows += r
        print(f"next_week  {target:<6} done")

    results = pd.DataFrame(rows)
    results.round(4).to_csv(RESULTS / "all_results.csv", index=False)

    full = results[results.Inputs.str.count(r"\+") == 2]
    same = full[full.Task == "same_week"]
    nxt = results[(results.Task == "next_week") & (results.Inputs == " + ".join(MARKETS))]
    r2_figure(same, IMAGES / "r2_same_week.png", "Same week: the other markets explain part of each market's moves")
    r2_figure(nxt, IMAGES / "r2_next_week.png", "Next week: no model beats predicting the average",
              legend_loc="upper left")

    pd.set_option("display.width", 160)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download prices")
    main(**vars(parser.parse_args()))
