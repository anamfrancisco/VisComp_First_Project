###############################################################################
## Computer Vision 2026-2027 - NOVA FCT
## Project Stage 1 - Shot boundary detection and recurring scenes
##
## Tomás Alexandre (73213)
## Ana Francisco (77580)
##
###############################################################################

import glob
import json
import os

import cv2
import numpy as np
from typing import List, Tuple, Optional

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

def pixel_difference(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Calcula a media das diferenças absolutas dos niveis de cinzento entre duas frames
    """
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(g1 - g2)))

def histogram_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Calcula a distancia (x^2) entre dois histogramas"""
    # Garante float32/float64 e achatamento
    h1 = h1.astype(np.float32).ravel()
    h2 = h2.astype(np.float32).ravel()
    
    # Normalizacao (soma = 1)
    s1, s2 = np.sum(h1), np.sum(h2)
    if s1 > 0: h1 /= s1
    if s2 > 0: h2 /= s2
    
    denom = h1 + h2
    mask = denom > 0
    
    x2 = 0.5 * np.sum(((h1[mask] - h2[mask]) ** 2) / denom[mask])
    return float(x2)

def edge_change_ratio(f1: np.ndarray, f2: np.ndarray, low_thresh: int = 100, high_thresh: int = 200, kernel_size: int = 5) -> float:
    """
    Calcula o Edge Change Ratio (ECR) entre duas frames consecutivas f1 (t-1) e f2 (t)
    """
    # 1. Converter para niveis de cinzento e detetar contornos com Canny (Cria mapas binarios)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    
    e1 = cv2.Canny(g1, low_thresh, high_thresh) > 0
    e2 = cv2.Canny(g2, low_thresh, high_thresh) > 0
    
    # Contagem total de pixels de contorno em cada frame (|E_t-1| e |E_t|)
    num_e1 = np.sum(e1)
    num_e2 = np.sum(e2)
    
    # 2. Dilatar os mapas de contorno com um kernel (ex: 5x5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    d1 = cv2.dilate(e1.astype(np.uint8), kernel) > 0
    d2 = cv2.dilate(e2.astype(np.uint8), kernel) > 0
    
    # 3. Calcular pixels que entram (X_in) e pixels que saem (X_out)
    x_in = np.sum(e2 & ~d1)   # Contornos de f2 que nao caem na dilatada de f1
    x_out = np.sum(e1 & ~d2)  # Contornos de f1 que nao caem na dilatada de f2
    
    # 4. Normalizacao e tratamento de divisao por zero (ex: frames completamente pretas)
    r_in = (x_in / num_e2) if num_e2 > 0 else 0.0
    r_out = (x_out / num_e1) if num_e1 > 0 else 0.0
    
    return float(max(r_in, r_out))

import cv2
import numpy as np

def compute_pixel_diff_signal(frames: list) -> np.ndarray:
    """Calcula o sinal d(t) para Pixel Difference."""
    n_frames = len(frames)
    d = np.zeros(n_frames, dtype=np.float32)
    
    for t in range(1, n_frames):
        d[t] = pixel_difference(frames[t-1], frames[t])
        
    return d


def compute_histogram_signal(frames: list, bins: tuple = (16, 16)) -> np.ndarray:
    """
    Calcula o sinal d(t) para Histogram Distance usando o espaco de cor HSV (H e S).
    Pre-calcula os histogramas para otimizar o tempo de execucao.
    """
    n_frames = len(frames)
    d = np.zeros(n_frames, dtype=np.float32)
    
    # 1. Pre-calcular o histograma de cada frame em HSV (canais H e S)
    hists = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        # Usa os canais 0 (Hue) e 1 (Saturation)
        hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
        hists.append(hist)
        
    # 2. Calcular a distancia entre frames consecutivas
    for t in range(1, n_frames):
        d[t] = histogram_distance(hists[t-1], hists[t])
        
    return d


def compute_ecr_signal(frames: list) -> np.ndarray:
    """Calcula o sinal d(t) para Edge Change Ratio (ECR)."""
    n_frames = len(frames)
    d = np.zeros(n_frames, dtype=np.float32)
    
    for t in range(1, n_frames):
        d[t] = edge_change_ratio(frames[t-1], frames[t])
        
    return d