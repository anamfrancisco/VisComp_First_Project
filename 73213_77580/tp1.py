###############################################################################
## Computer Vision 2026-2027 - NOVA FCT
## Project Stage 1 - Shot boundary detection and recurring scenes
##
## Student 1 (number)
## Student 2 (number)
##
###############################################################################

import glob
import json
import os

import cv2
import numpy as np

import grouping   # group_shots: union-find grouping of shots (Part 3, Task 7)
import metrics

INPUT_DIR = os.path.join("..", "input")
OUTPUT_DIR = os.path.join("..", "output")


def read_frames(path, size=(320, 180)):
    """Read every frame of a video, resized to `size`. Returns a list of BGR images."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


def process_video(frames, gt_shots):
    """Return (transitions, scenes) for one video.

    transitions: list of {"type": "cut", "start": t, "end": t}, t = first frame of the new shot (Parts 1 and 2)
    scenes:      dict {str(shot id): label} for the ground-truth shots (Part 3)
    """
    transitions, scenes = [], {}
    # TODO: Parts 1, 2 and 3. Everything else (writing the output, evaluation, summary table) is done for you.
    return transitions, scenes


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}                                    # video name -> counts from metrics.evaluate_video
    for video_path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4"))):
        name = os.path.splitext(os.path.basename(video_path))[0]
        gt = json.load(open(os.path.join(INPUT_DIR, name + ".json")))
        frames = read_frames(video_path)
        transitions, scenes = process_video(frames, gt["shots"])
        pred = {"video": name + ".mp4", "transitions": transitions, "scenes": scenes}
        with open(os.path.join(OUTPUT_DIR, name + ".json"), "w") as f:
            json.dump(pred, f, indent=1, default=int)    # default=int: accepts NumPy integers
        results[name] = metrics.evaluate_video(gt, pred)

    # summary table (Section 8): one row per video, then the dev / test totals
    dev = [results[n] for n in sorted(results) if n.startswith("dev")]
    test = [results[n] for n in sorted(results) if n.startswith("test")]
    metrics.print_table([(n, results[n]) for n in sorted(results)]
                        + [("DEV", metrics.sum_counts(dev)), ("TEST", metrics.sum_counts(test))])
