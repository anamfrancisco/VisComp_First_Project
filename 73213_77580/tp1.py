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

import grouping   # group_shots: union-find grouping of shots (Part 3, Task 7)
import metrics

INPUT_DIR = os.path.join("..", "input")
OUTPUT_DIR = os.path.join("..", "output")

# Parametros do detetor de cortes (escolhidos no DEV, ver report.ipynb)
CUT_SIGNAL = "histogram"     # sinal principal: "pixel", "histogram" ou "ecr"
CUT_THRESHOLD = 0.0400
CONFIRM_SIGNAL = "pixel"     # sinal de confirmacao (None para usar so o principal)
CONFIRM_THRESHOLD = 33.3024
MIN_GAP = 5                  # janela (+-frames) para ficar so com o maximo local

# Parametros da verificacao com SIFT (Parte 2, escolhidos no DEV, ver report.ipynb)
VERIFY_CUTS = True           # False para usar so a Parte 1
GAP = 4                      # compara as frames t-GAP e t+GAP (um flash dura no maximo 3 frames)
RATIO = 0.7                  # ratio test de Lowe (como no lab 3)
USE_RANSAC = False           # True para contar so os inliers da homografia (deu o mesmo F1)
MIN_MATCHES = 31             # com pelo menos MIN_MATCHES matches e a mesma cena -> rejeita o corte


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


###############################################################################
## Parte 1 - Funcoes de dissimilaridade
###############################################################################

def pixel_difference(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Calcula a media das diferenças absolutas dos niveis de cinzento entre duas frames
    """
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(g1 - g2)))


def histogram_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Calcula a distancia (x^2) entre dois histogramas (resultado entre 0 e 1)"""
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


def compute_edges(frame: np.ndarray, low_thresh: int = 100, high_thresh: int = 200, kernel_size: int = 5):
    """Mapa de contornos (Canny) de uma frame e a sua versao dilatada"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh) > 0
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(edges.astype(np.uint8), kernel) > 0
    return edges, dilated


def ecr_from_edges(e1, d1, e2, d2) -> float:
    """ECR a partir dos contornos (e) e contornos dilatados (d) de t-1 (1) e t (2)"""
    num_e1 = np.sum(e1)
    num_e2 = np.sum(e2)

    x_in = np.sum(e2 & ~d1)   # Contornos de f2 que nao caem na dilatada de f1
    x_out = np.sum(e1 & ~d2)  # Contornos de f1 que nao caem na dilatada de f2

    # Tratamento de divisao por zero (ex: frames completamente pretas)
    r_in = (x_in / num_e2) if num_e2 > 0 else 0.0
    r_out = (x_out / num_e1) if num_e1 > 0 else 0.0

    return float(max(r_in, r_out))


def edge_change_ratio(f1: np.ndarray, f2: np.ndarray, low_thresh: int = 100, high_thresh: int = 200, kernel_size: int = 5) -> float:
    """
    Calcula o Edge Change Ratio (ECR) entre duas frames consecutivas f1 (t-1) e f2 (t)
    """
    e1, d1 = compute_edges(f1, low_thresh, high_thresh, kernel_size)
    e2, d2 = compute_edges(f2, low_thresh, high_thresh, kernel_size)
    return ecr_from_edges(e1, d1, e2, d2)


###############################################################################
## Parte 1 - Sinais d(t), com d(0) = 0
###############################################################################

def compute_pixel_diff_signal(frames: list) -> np.ndarray:
    """Calcula o sinal d(t) para Pixel Difference."""
    n_frames = len(frames)
    d = np.zeros(n_frames, dtype=np.float32)

    for t in range(1, n_frames):
        d[t] = pixel_difference(frames[t-1], frames[t])

    return d


def compute_histogram_signal(frames: list, bins: tuple = (8, 8, 8), space: str = "rgb") -> np.ndarray:
    """
    Calcula o sinal d(t) para Histogram Distance.
    space="hsv": histograma 2D dos canais H e S; space="rgb": histograma 3D dos canais B, G e R.
    Pre-calcula os histogramas para otimizar o tempo de execucao.
    """
    n_frames = len(frames)
    d = np.zeros(n_frames, dtype=np.float32)

    # 1. Pre-calcular o histograma de cada frame
    hists = []
    for f in frames:
        if space == "hsv":
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            # Usa os canais 0 (Hue) e 1 (Saturation)
            hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
        else:
            hist = cv2.calcHist([f], [0, 1, 2], None, list(bins), [0, 256, 0, 256, 0, 256])
        hists.append(hist)

    # 2. Calcular a distancia entre frames consecutivas
    for t in range(1, n_frames):
        d[t] = histogram_distance(hists[t-1], hists[t])

    return d


def compute_ecr_signal(frames: list, kernel_size: int = 5) -> np.ndarray:
    """
    Calcula o sinal d(t) para Edge Change Ratio (ECR).
    Os contornos de cada frame sao calculados uma so vez (cada frame entra em dois pares).
    """
    n_frames = len(frames)
    d = np.zeros(n_frames, dtype=np.float32)

    edges = [compute_edges(f, kernel_size=kernel_size) for f in frames]
    for t in range(1, n_frames):
        e1, d1 = edges[t-1]
        e2, d2 = edges[t]
        d[t] = ecr_from_edges(e1, d1, e2, d2)

    return d


def compute_signal(frames: list, name: str) -> np.ndarray:
    """Calcula um dos tres sinais pelo nome ("pixel", "histogram" ou "ecr")."""
    if name == "pixel":
        return compute_pixel_diff_signal(frames)
    if name == "histogram":
        return compute_histogram_signal(frames)
    if name == "ecr":
        return compute_ecr_signal(frames)
    raise ValueError(f"sinal desconhecido: {name}")


###############################################################################
## Parte 1 - Do sinal aos cortes
###############################################################################

def detect_cuts(signal: np.ndarray, threshold: float, min_gap: int = MIN_GAP) -> list:
    """
    Frames acima do limiar que sao maximo local numa janela de +-min_gap frames.
    Se 2 frames seguidas passarem o limiar no mesmo corte, so a maior conta
    (uma previsao repetida contaria como falso positivo no match_cuts).
    """
    cuts = []
    for t in np.where(signal > threshold)[0]:
        lo, hi = max(0, t - min_gap), min(len(signal), t + min_gap + 1)
        if signal[t] == signal[lo:hi].max():
            cuts.append(int(t))
    return cuts


def detect_cuts_combined(signal: np.ndarray, threshold: float,
                         confirm: np.ndarray, confirm_threshold: float, min_gap: int = MIN_GAP) -> list:
    """Cortes do sinal principal que sao confirmados pelo segundo sinal (tambem acima do seu limiar)"""
    return [t for t in detect_cuts(signal, threshold, min_gap) if confirm[t] > confirm_threshold]


###############################################################################
## Parte 2 - Verificacao dos cortes candidatos com SIFT
## (baseado no match_images_FLANN e no filter_matches_RANSAC do lab3_match.py)
###############################################################################

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2)    # NORM_L2 porque os descritores SIFT sao vetores reais


def sift_matches(f1: np.ndarray, f2: np.ndarray, ratio: float = RATIO):
    """
    Keypoints SIFT das duas frames e os matches que passam o ratio test de Lowe.
    Devolve (kp1, kp2, good_matches).
    """
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    # frames sem keypoints (ex: frame quase uniforme) -> nao ha matches
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return kp1, kp2, []

    # para cada keypoint de f1, os 2 vizinhos mais proximos em f2
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for pair in raw_matches:
        # so aceita se o melhor for claramente melhor que o segundo
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good_matches.append(pair[0])
    return kp1, kp2, good_matches


def ransac_inliers(kp1, kp2, matches, threshold: float = 5.0) -> list:
    """Matches que concordam com uma homografia (RANSAC). Com menos de 4 matches nao da para calcular."""
    if len(matches) < 4:
        return []
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    cv2.setRNGSeed(0)    # o RANSAC e aleatorio: fixar a semente para os resultados serem sempre iguais
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
    if H is None:
        return []
    return [m for m, keep in zip(matches, mask.ravel()) if keep]


def count_matches(f1: np.ndarray, f2: np.ndarray, use_ransac: bool = USE_RANSAC) -> int:
    """Numero de matches entre duas frames (inliers do RANSAC se use_ransac=True)"""
    kp1, kp2, good = sift_matches(f1, f2)
    if use_ransac:
        return len(ransac_inliers(kp1, kp2, good))
    return len(good)


def candidate_matches(frames: list, cuts: list, gap: int = GAP, use_ransac: bool = USE_RANSAC) -> list:
    """Para cada corte candidato t, numero de matches entre as frames t-gap e t+gap"""
    counts = []
    for t in cuts:
        before = max(0, t - gap)
        after = min(len(frames) - 1, t + gap)
        counts.append(count_matches(frames[before], frames[after], use_ransac))
    return counts


def verify_cuts(frames: list, cuts: list, min_matches: int = MIN_MATCHES,
                gap: int = GAP, use_ransac: bool = USE_RANSAC) -> list:
    """
    Fica so com os cortes em que as frames antes e depois NAO se parecem.
    Muitos matches -> mesmo conteudo (flash, movimento...) -> falso alarme.
    """
    counts = candidate_matches(frames, cuts, gap, use_ransac)
    return [t for t, n in zip(cuts, counts) if n < min_matches]


def process_video(frames, gt_shots):
    """Return (transitions, scenes) for one video.

    transitions: list of {"type": "cut", "start": t, "end": t}, t = first frame of the new shot (Parts 1 and 2)
    scenes:      dict {str(shot id): label} for the ground-truth shots (Part 3)
    """
    # Partes 1 e 2: cortes
    signal = compute_signal(frames, CUT_SIGNAL)
    if CONFIRM_SIGNAL is None:
        cuts = detect_cuts(signal, CUT_THRESHOLD)
    else:
        confirm = compute_signal(frames, CONFIRM_SIGNAL)
        cuts = detect_cuts_combined(signal, CUT_THRESHOLD, confirm, CONFIRM_THRESHOLD)

    # Parte 2: verificacao dos candidatos com SIFT
    if VERIFY_CUTS:
        cuts = verify_cuts(frames, cuts)
    transitions = [{"type": "cut", "start": t, "end": t} for t in cuts]

    # Parte 3: TODO (por agora cada shot e uma cena)
    scenes = {}
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
