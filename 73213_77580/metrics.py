###############################################################################
## Computer Vision 2026-2027 - NOVA FCT
## Project Stage 1 - Evaluation metrics (see Section 7 of the assignment)
##
## Everything in this file is given. Do not change its behaviour: every group must
## compute the metrics in exactly the same way. tp1.py already calls evaluate_video,
## sum_counts and print_table; you only have to produce the predictions.
###############################################################################

TOLERANCE = 2  # frames


def match_cuts(gt_cuts, pred_cuts, tol=TOLERANCE):
    """One-to-one greedy matching of predicted and ground-truth cuts.

    gt_cuts, pred_cuts: lists of frame indices (the first frame of each new shot).
    Returns (TP, FP, FN).
    """
    # all candidate pairs closer than the tolerance, closest first
    pairs = sorted((abs(p - g), i, j)
                   for i, g in enumerate(gt_cuts)
                   for j, p in enumerate(pred_cuts) if abs(p - g) <= tol)
    used_gt, used_pred = set(), set()
    for _, i, j in pairs:
        if i not in used_gt and j not in used_pred:
            used_gt.add(i)
            used_pred.add(j)
    tp = len(used_gt)
    return tp, len(pred_cuts) - tp, len(gt_cuts) - tp


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def flash_false_alarms(distractors, pred_cuts, tol=TOLERANCE):
    """Number of distractors (flashes) with at least one predicted cut within `tol` frames."""
    return sum(any(d["start"] - tol <= p <= d["end"] + tol for p in pred_cuts) for d in distractors)


def pairwise_scene_counts(gt_scenes, pred_scenes):
    """gt_scenes / pred_scenes: dict {shot id: label}. Returns (TP, FP, FN) over all pairs of shots.

    Labels are only compared for equality: their names do not matter.
    """
    ids = sorted(gt_scenes)
    tp = fp = fn = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            same_gt = gt_scenes[i] == gt_scenes[j]
            same_pred = pred_scenes[i] == pred_scenes[j]
            if same_gt and same_pred:
                tp += 1
            elif same_pred:
                fp += 1
            elif same_gt:
                fn += 1
    return tp, fp, fn


###############################################################################
## Applying the metrics to the videos
###############################################################################


def evaluate_video(gt, pred):
    """Counts for one video.

    gt:   the ground-truth dict (the video's JSON file in ../input)
    pred: your prediction dict (the same format you write to ../output, Section 8)
    Returns a dict {"cut": (TP, FP, FN), "flash_fa": int, "num_flashes": int, "scene": (TP, FP, FN)}
    """
    gt_cuts = [t["start"] for t in gt["transitions"]]
    pred_cuts = [int(t["start"]) for t in pred.get("transitions", [])]
    # shot ids as strings on both sides (the keys of a JSON object are always strings);
    # a shot without a predicted label is treated as a scene of its own
    gt_scenes = {str(s["id"]): s["scene"] for s in gt["shots"]}
    pred_labels = {str(k): str(v) for k, v in pred.get("scenes", {}).items()}
    pred_scenes = {i: pred_labels.get(i, "_unlabelled_" + i) for i in gt_scenes}
    return {"cut": match_cuts(gt_cuts, pred_cuts),
            "flash_fa": flash_false_alarms(gt["distractors"], pred_cuts),
            "num_flashes": len(gt["distractors"]),
            "scene": pairwise_scene_counts(gt_scenes, pred_scenes)}


def sum_counts(results):
    """Sum the counts of several videos (e.g., the two dev videos).

    results: list of dicts returned by evaluate_video
    Returns one dict with the same keys, where every count is the sum over the videos.
    Totals are computed from these summed counts, never by averaging per-video scores (Section 7).
    """
    return {"cut": tuple(sum(r["cut"][k] for r in results) for k in range(3)),
            "flash_fa": sum(r["flash_fa"] for r in results),
            "num_flashes": sum(r["num_flashes"] for r in results),
            "scene": tuple(sum(r["scene"][k] for r in results) for k in range(3))}


def print_table(rows):
    """Print the summary table of Section 8.

    rows: list of (name, counts) pairs, where counts comes from evaluate_video (one video) or from sum_counts
          (e.g., ("DEV", sum_counts(dev_results)), ("TEST", sum_counts(test_results))).
    """
    print(f"{'video':10s} {'cut-P':>6s} {'cut-R':>6s} {'cut-F1':>7s} {'flash-FA':>9s} "
          f"{'scene-P':>8s} {'scene-R':>8s} {'scene-F1':>9s}")
    for name, c in rows:
        cp, cr, cf = precision_recall_f1(*c["cut"])
        sp, sr, sf = precision_recall_f1(*c["scene"])
        fa = f"{c['flash_fa']}/{c['num_flashes']}"
        print(f"{name:10s} {cp:6.3f} {cr:6.3f} {cf:7.3f} {fa:>9s} {sp:8.3f} {sr:8.3f} {sf:9.3f}")
