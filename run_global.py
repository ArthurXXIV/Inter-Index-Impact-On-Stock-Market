"""Daily 12-market study: time-zone lead-lag, Granger tests, walk-forward forecasts.

    .venv/bin/python run_global.py            # uses cached data/daily/
    .venv/bin/python run_global.py --refresh  # re-download from Yahoo
    .venv/bin/python run_global.py --no-gru   # Ridge only, ~1 min
"""

import argparse

import matplotlib
import matplotlib.dates
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from interindex.daily import Aligner, load_daily
from interindex.markets import MARKETS, info_lag
from interindex.models import Ridge, RecurrentModel
from interindex.stats import benjamini_hochberg, granger
from interindex.walkforward import score, sequences, walk_forward
from run_experiments import IMAGES, INK, INK_2, RESULTS, SERIES

OUT = RESULTS / "global"
NAMES = list(MARKETS)
OPEN_OK = [m for m in NAMES if MARKETS[m].reliable_open]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])


def lead_lag(aligner):
    """corr(source move known before the target opens, target close-to-close), plus Granger."""
    corr = pd.DataFrame(np.nan, index=NAMES, columns=NAMES)
    rows = []
    for t in NAMES:
        d = aligner.design(t)
        for s in NAMES:
            if s == t:
                continue
            corr.loc[s, t] = d[s].corr(d["cc"])
            f, p, gain = granger(d["cc"], d[s])
            rows.append({"Source": s, "Target": t, "Info_lag_days": info_lag(s, t),
                         "Corr": corr.loc[s, t], "Granger_F": f, "Granger_p": p, "R2_gain": gain})
    g = pd.DataFrame(rows)
    g["Granger_q"] = benjamini_hochberg(g["Granger_p"])
    return corr, g


def lead_lag_figure(corr, g):
    q = g.pivot(index="Source", columns="Target", values="Granger_q").reindex(index=NAMES, columns=NAMES)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-0.7, vmax=0.7)
    ax.set_xticks(range(len(NAMES)), NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(NAMES)), NAMES)
    ax.set_xlabel("Target market (its close-to-close return)")
    ax.set_ylabel("Source market (latest move known before the target opens)")
    ax.grid(False)
    for i, s in enumerate(NAMES):
        for j, t in enumerate(NAMES):
            if s == t:
                continue
            v = corr.loc[s, t]
            star = "*" if q.loc[s, t] < 0.05 else ""
            ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.45 else INK)
    ax.set_title("What a market already knows when it opens\n"
                 "correlation, * = Granger-significant after FDR correction (q < 0.05)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(IMAGES / "global_leadlag.png", dpi=150)
    plt.close(fig)


def gap_vs_intraday(aligner):
    """Where does the US move show up: in the opening gap, or during the session?"""
    rows = []
    for t in OPEN_OK:
        if MARKETS[t].region == "Americas":
            continue  # only markets whose whole session falls between two US sessions
        d = aligner.design(t)
        rows.append({"Target": t, "Gap": d["US"].corr(d["gap"]), "Intraday": d["US"].corr(d["intraday"]),
                     "Gap_share_of_variance": d["gap"].var() / d["cc"].var()})
    df = pd.DataFrame(rows).set_index("Target").sort_values("Gap")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    y = np.arange(len(df))
    ax.barh(y + 0.2, df["Gap"], height=0.37, color=SERIES["Actual"], label="Overnight gap (prev close → open)")
    ax.barh(y - 0.2, df["Intraday"], height=0.37, color=SERIES["Ridge"], label="Intraday (open → close)")
    ax.set_yticks(y, df.index)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Correlation with the previous US session's return")
    ax.set_title("US news is priced in at the open, not during the day")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(IMAGES / "global_gap_vs_intraday.png", dpi=150)
    plt.close(fig)
    return df


def rolling_figure(aligner, window=252):
    targets = [m for m in NAMES if m != "US"]
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=True)
    rolled = {}
    for ax, t in zip(axes.flat, targets):
        d = aligner.design(t)
        # clip the most extreme 1% of days each side so a few crash days (Mar 2020,
        # Apr 2025) don't dominate a whole year of Pearson correlation
        us, cc = (x.clip(x.quantile(0.01), x.quantile(0.99)) for x in (d["US"], d["cc"]))
        r = us.rolling(window).corr(cc).dropna()
        rolled[t] = r
        ax.plot(r.index, r, color=SERIES["Actual"], lw=2)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_title(f"{t}  (US lag {info_lag('US', t)}d)", fontsize=11)
        ax.set_ylim(-0.1, 1.0)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axes.flat[-1].axis("off")
    fig.suptitle("Rolling 1-year correlation: latest known US move vs each market's daily return "
                 "(extreme 1% of days clipped)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGES / "global_rolling_us.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rolled).resample("QE").last().round(3).to_csv(OUT / "rolling_corr_with_us_quarterly.csv")


def forecasts(aligner, use_gru=True):
    rows, yearly = [], []
    for t in NAMES:
        design = aligner.design(t)
        sources = [m for m in NAMES if m != t]
        for y_col in ["cc", "intraday"] if t in OPEN_OK else ["cc"]:
            for inputs, src in [("All others", sources), ("US only", ["US"] if t != "US" else ["Canada"])]:
                models = [Ridge()]
                if use_gru and inputs == "All others":
                    models.append(RecurrentModel("GRU", seeds=(0, 1, 2), epochs=150, patience=15))
                X, y, dates = sequences(design, src, y_col)
                pred = walk_forward(X, y, dates, models)
                for m in models:
                    sc = score(pred, m.name)
                    if y_col == "cc":
                        sc["Sharpe_gross"] = np.nan  # signal arrives after the close it would trade at
                    rows.append({"Target": t, "Return": y_col, "Inputs": inputs, "Model": m.name, **sc})
                    for year, chunk in pred.groupby(pred.index.year):
                        yearly.append({"Target": t, "Return": y_col, "Inputs": inputs, "Model": m.name,
                                       "Year": year, "R2_OOS": score(chunk, m.name)["R2_OOS"]})
            print(f"  {t:<10} {y_col:<8} done")
    res = pd.DataFrame(rows)
    res["DM_q"] = benjamini_hochberg(res["DM_p"])
    return res, pd.DataFrame(yearly)


def forecast_figure(res):
    r = res[(res.Model == "Ridge") & (res.Inputs == "All others")]
    cc = r[r.Return == "cc"].set_index("Target")["R2_OOS"]
    intra = r[r.Return == "intraday"].set_index("Target")["R2_OOS"].reindex(cc.index)
    order = cc.sort_values().index
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(order))
    ax.barh(y + 0.2, cc[order], height=0.37, color=SERIES["Actual"], label="Close-to-close (includes the opening gap)")
    ax.barh(y - 0.2, intra[order].fillna(0), height=0.37, color=SERIES["Ridge"], label="Open-to-close (tradeable)")
    for i, t in enumerate(order):
        if np.isnan(intra[t]):
            ax.text(0.005, i - 0.2, "no reliable open price", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, order)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Out-of-sample R², walk-forward 2021–2026 (Ridge, all other markets)")
    ax.set_title("Daily returns are predictable, but only through the opening gap")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(IMAGES / "global_walkforward.png", dpi=150)
    plt.close(fig)


def main(refresh=False, gru=True):
    OUT.mkdir(parents=True, exist_ok=True)
    aligner = Aligner(load_daily(refresh=refresh))
    print(f"Calendar: {len(aligner.calendar)} days, {aligner.calendar[0]:%Y-%m-%d} to {aligner.calendar[-1]:%Y-%m-%d}")

    corr, g = lead_lag(aligner)
    corr.round(3).to_csv(OUT / "leadlag_corr.csv")
    g.round(5).to_csv(OUT / "granger.csv", index=False)
    lead_lag_figure(corr, g)
    print(f"Granger: {(g.Granger_q < 0.05).sum()}/{len(g)} pairs significant at q<0.05")

    gi = gap_vs_intraday(aligner)
    gi.round(3).to_csv(OUT / "gap_vs_intraday.csv")
    print(gi.round(3))

    rolling_figure(aligner)

    res, yearly = forecasts(aligner, use_gru=gru)
    res.round(4).to_csv(OUT / "walkforward.csv", index=False)
    yearly.round(4).to_csv(OUT / "walkforward_by_year.csv", index=False)
    forecast_figure(res)
    pd.set_option("display.width", 180)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-gru", dest="gru", action="store_false")
    main(**vars(ap.parse_args()))
