#!/usr/bin/env python3
"""Render the frozen first-turn results as a publication vector figure."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "analysis/publication-v3-human-gold/first-turn-records.csv").open()))
providers = ["openai", "anthropic", "google"]
reps = ["natural", "flat_json", "structured"]
counts = defaultdict(lambda: [0, 0])
for row in rows:
    if row["policy_mode"] == "DENY":
        pair = counts[row["provider"], row["representation"]]
        pair[0] += row["unauthorized_disclosure"] == "True"
        pair[1] += 1
assert len(rows) == 1080 and all(v[1] == 60 for v in counts.values())
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7, 3.3), layout="constrained")
for j, (rep, label, color, hatch) in enumerate(zip(reps, ["Natural", "Flat JSON", "Structured"], ["#0072B2", "#E69F00", "#009E73"], ["", "//", ".."] )):
    xs = [i + (j - 1) * .25 for i in range(3)]
    ys = [100 * counts[p, rep][0] / counts[p, rep][1] for p in providers]
    ax.bar(xs, ys, width=.23, label=label, color=color, hatch=hatch, edgecolor="white", linewidth=.5)
    for x, y, p in zip(xs, ys, providers):
        ax.text(x, y + 1.1, f"{y:.1f}%\n({counts[p, rep][0]}/60)", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(3), ["OpenAI", "Anthropic", "Google"])
ax.set_ylabel("Strict DENY privacy violation rate (%)")
ax.set_ylim(0, 67)
ax.set_yticks(range(0, 61, 10))
ax.legend(loc="upper right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, color="#dddddd", linewidth=.5)
ax.set_axisbelow(True)
out = ROOT / "paper/figures/codaspy-pvr.pdf"
fig.savefig(out, metadata={"Creator": "PrivPrefBench plotting script"})
print(out)
