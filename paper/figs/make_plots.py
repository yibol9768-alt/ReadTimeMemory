"""Data figures (vector PDF). Numbers are copied from docs/PROGRESS.md / the judged files and saved to plot_data.json."""
import json, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--", "lines.linewidth": 1.6, "axes.edgecolor": "#6B7785",
    "axes.labelcolor": "#2F3E4E", "xtick.color": "#2F3E4E", "ytick.color": "#2F3E4E"})
OURS, HS, HSCC, REC, RAW = "#4F7CAC", "#C46A5E", "#E3A48F", "#8C9AA8", "#C9D0D8"
D = {}
# --- Fig 3: budget curve (BEAM-1M dev, 100 q, tuned prompt, no CC, one run)
D["budget"] = {"x": [8, 20, 60, 124], "records": [68.6, 69.7, 70.7, 69.2], "hindsight": [47.3, 57.5, 69.1, 72.3]}
fig, ax = plt.subplots(figsize=(2.7, 2.1))
b = D["budget"]
ax.plot(b["x"], b["records"], "o-", color=REC, label="Atomic records")
ax.plot(b["x"], b["hindsight"], "s-", color=HS, label="Hindsight")
ax.set_xscale("log"); ax.set_xticks(b["x"]); ax.set_xticklabels(["8K", "20K", "60K", "124K"]); ax.minorticks_off()
ax.set_xlabel("Context budget (characters)"); ax.set_ylabel("BEAM-1M score"); ax.set_ylim(44, 76)
for x, y in zip(b["x"], b["hindsight"]): ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(5, -9), fontsize=7.5, color=HS)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("fig3_budget.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# --- Fig 4: where lost points go (BEAM-1M, minimal prompt, one run)
D["loss"] = {"systems": ["Raw chunks (8K)", "Records (8K)", "Hindsight (124K)"],
             "in_context": [8.6, 8.8, 19.6], "not_retrieved": [31.8, 15.4, 16.6], "not_stored": [0, 12.0, 0]}
fig, ax = plt.subplots(figsize=(2.9, 2.0))
L = D["loss"]; y = range(3)
ax.barh(y, L["in_context"], color=OURS, label="In context, missed by reader")
ax.barh(y, L["not_retrieved"], left=L["in_context"], color=RAW, label="Not in context")
ax.barh(y, L["not_stored"], left=[a + b for a, b in zip(L["in_context"], L["not_retrieved"])], color=REC, hatch="///", edgecolor="white", label="Never stored")
for i in y:
    tot = L["in_context"][i] + L["not_retrieved"][i] + L["not_stored"][i]
    ax.text(tot + 0.8, i, f"{tot:.0f}", va="center", fontsize=7.5, color="#2F3E4E")
ax.set_yticks(list(y)); ax.set_yticklabels(L["systems"]); ax.invert_yaxis(); ax.set_xlim(0, 46)
ax.set_xlabel("Points lost (of 100)"); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, handlelength=1.4, columnspacing=0.8)
fig.tight_layout(); fig.savefig("fig4_loss.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# --- Fig 5: readers (tuned prompt; filled = three runs, open = one run: gpt-6-astra)
D["readers"] = {"readers": ["gpt-6-luna", "DeepSeek-\nV4.1-Flash", "gpt-6.1-sol", "gpt-6-astra"],
  "runs": {"100K": [3, 3, 3, 1], "1M": [3, 3, 3, 1]},
  "100K": {"Hindsight": [68.2, 75.3, 76.8, 75.3], "Hindsight+CC": [69.5, 76.0, 77.0, 76.7], "Ledger+Compile": [76.8, 76.8, 78.9, 79.7]},
  "1M": {"Hindsight": [70.4, 75.2, 78.8, 80.4], "Hindsight+CC": [71.2, 76.7, 78.9, 79.5], "Ledger+Compile": [75.4, 79.9, 80.2, 80.9]}}
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.2), sharey=True)
for ax, tier in zip(axs, ("100K", "1M")):
    xs = list(range(4)); runs = D["readers"]["runs"][tier]
    for name, col, m in (("Hindsight", HS, "s"), ("Hindsight+CC", HSCC, "D"), ("Ledger+Compile", OURS, "o")):
        v = D["readers"][tier][name]
        ax.plot(xs, v, "-", color=col, alpha=0.6, lw=1.2)
        for i, (x, y) in enumerate(zip(xs, v)):
            full = runs[i] == 3
            ax.plot([x], [y], m, color=col, ms=5.5, mfc=col if full else "white", mec=col, mew=1.3,
                    label=name if i == 0 else None)
    ax.set_xticks(xs); ax.set_xticklabels(D["readers"]["readers"], fontsize=6.5); ax.set_title(f"BEAM-{tier}", fontsize=8.5)
    ax.set_xlim(-0.35, 3.35); ax.set_ylim(66, 82)
axs[0].set_ylabel("Score")
axs[1].legend(frameon=False, loc="lower right", fontsize=7.5)
fig.tight_layout(); fig.savefig("fig5_readers.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
# --- Fig 6: per-ability difference, ours minus Hindsight+CC (gpt-6-luna, three runs)
ab = ["Abstention", "Contradiction", "Event ordering", "Info. extraction", "Instruction following", "Knowledge update",
      "Multi-session", "Preference", "Summarization", "Temporal"]
hs100 = [40.0, 70.0, 60.2, 87.7, 71.7, 59.2, 77.1, 95.4, 63.2, 70.4]; us100 = [68.3, 74.4, 63.6, 85.8, 85.8, 78.3, 82.5, 93.3, 56.1, 79.6]
hs1m = [55.8, 62.1, 41.4, 82.7, 71.9, 86.7, 78.9, 90.0, 62.8, 80.0]; us1m = [57.5, 68.3, 45.8, 86.7, 91.7, 99.2, 75.7, 88.3, 53.0, 88.3]
d100 = [a - b for a, b in zip(us100, hs100)]; d1m = [a - b for a, b in zip(us1m, hs1m)]
D["ability"] = {"abilities": ab, "diff_100K": d100, "diff_1M": d1m}
order = sorted(range(10), key=lambda i: d100[i] + d1m[i])
fig, ax = plt.subplots(figsize=(2.9, 2.6))
yy = range(10)
ax.barh([y + 0.2 for y in yy], [d100[i] for i in order], height=0.38, color=OURS, label="BEAM-100K")
ax.barh([y - 0.2 for y in yy], [d1m[i] for i in order], height=0.38, color="#A9C1DC", label="BEAM-1M")
ax.axvline(0, color="#2F3E4E", lw=0.8); ax.set_yticks(list(yy)); ax.set_yticklabels([ab[i] for i in order])
ax.set_xlabel("Ledger+Compile minus Hindsight+CC"); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right", fontsize=7.5)
fig.tight_layout(); fig.savefig("fig6_ability.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
json.dump(D, open("plot_data.json", "w"), indent=1); print("ok")
