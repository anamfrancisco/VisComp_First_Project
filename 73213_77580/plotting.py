###############################################################################
## Computer Vision 2026-2027 - NOVA FCT
## Project Stage 1 - Plotting helpers (given; you may adapt them)
###############################################################################

import matplotlib.pyplot as plt


def plot_signals(signals, gt, start=0, end=None, path=None):
    """Plot one or more dissimilarity signals with the ground-truth cuts and flashes overlaid.

    signals: dict {name: sequence d where d[t] compares frames t-1 and t}, e.g. {"histogram": d_hist}
    gt:      the ground-truth dict loaded from the video's JSON file
    start, end: range of frames to show (showing ~30 s at a time keeps the plot readable)
    path:    if given, the figure is also saved to this file

    How to call it (e.g., in your notebook), with your three signals for dev_01:

        gt = json.load(open("../input/dev_01.json"))
        plotting.plot_signals({"pixel": d_pixel, "histogram": d_hist, "ecr": d_ecr}, gt,
                              start=500, end=1250, path="../output/signals_dev_01.png")
    """
    end = end if end is not None else len(next(iter(signals.values())))
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 2.2 * len(signals)), sharex=True, squeeze=False)
    for ax, (name, d) in zip(axes[:, 0], signals.items()):
        ax.plot(range(start, end), d[start:end], lw=0.8, label=name)
        for i, t in enumerate(gt["transitions"]):
            if start <= t["start"] < end:
                ax.axvline(t["start"], color="tab:red", ls="--", lw=0.8, label="ground-truth cut" if i == 0 else None)
        for f in gt["distractors"]:
            if start <= f["start"] < end:
                ax.axvspan(f["start"] - 0.5, f["end"] + 0.5, color="tab:orange", alpha=0.4)
        ax.set_ylabel(name)
    axes[-1, 0].set_xlabel("frame index t")
    axes[0, 0].set_title(f"{gt['video']}: red dashed = ground-truth cuts, orange = flashes")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_pr_curve(curves, title="Precision-recall curve", path=None):
    """Plot one precision-recall curve per signal and mark the threshold with the best F1.

    curves: dict {name: list of (threshold, precision, recall)}, one entry per threshold you tried,
            e.g. {"histogram": [(0.10, 0.45, 0.98), (0.20, 0.61, 0.95), ...], "pixel": [...]}
    title:  title of the figure
    path:   if given, the figure is also saved to this file
    Returns the figure.

    How to call it. You compute the points: for every threshold T, run YOUR cut detector with that threshold
    on the dev videos, score the predicted cuts with the given metrics functions (Section 7) and store
    (T, precision, recall). Then pass all the points to this function:

        gts = {n: json.load(open(f"../input/{n}.json")) for n in ["dev_01", "dev_02"]}
        curves = {"histogram": []}
        for T in np.linspace(0.05, 0.95, 50):            # thresholds to try: choose a sensible range
            counts = []
            for n in gts:
                pred_cuts = ...                          # YOUR detector on video n with threshold T
                gt_cuts = [t["start"] for t in gts[n]["transitions"]]
                counts.append(metrics.match_cuts(gt_cuts, pred_cuts))
            tp, fp, fn = (sum(c[k] for c in counts) for k in range(3))   # sum the counts over videos
            p, r, f1 = metrics.precision_recall_f1(tp, fp, fn)
            curves["histogram"].append((T, p, r))
        # ... the same for "pixel" and "ecr" ...
        plotting.plot_pr_curve(curves, title="dev videos", path="../output/pr_curve.png")
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, points in curves.items():
        points = sorted(points, key=lambda x: x[2])      # by recall, so the line is drawn left to right
        thresholds = [x[0] for x in points]
        precision = [x[1] for x in points]
        recall = [x[2] for x in points]
        line, = ax.plot(recall, precision, marker=".", ms=3, lw=1, label=name)
        f1 = [2 * p * r / (p + r) if p + r > 0 else 0.0 for p, r in zip(precision, recall)]
        b = max(range(len(f1)), key=f1.__getitem__)
        ax.plot(recall[b], precision[b], "o", color=line.get_color(), ms=7, mfc="none")
        ax.annotate(f"T={thresholds[b]:.3g}, F1={f1[b]:.2f}", (recall[b], precision[b]),
                    textcoords="offset points", xytext=(-8, -14), ha="right", fontsize=8, color=line.get_color())
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
