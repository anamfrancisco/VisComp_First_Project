###############################################################################
## Computer Vision 2026-2027 - NOVA FCT
## Project Stage 1 - Shot boundary detection and recurring scenes
##
## Tomás Alexandre (73213)
## Ana Francisco (77580)
###############################################################################

import glob
import json
import os
import cv2
import numpy as np

import grouping
import metrics

INPUT_DIR = os.path.join("..", "input")
OUTPUT_DIR = os.path.join("..", "output")

# parametros que deram melhor no dev (ver report.ipynb)

# Parte 1 - deteccao de cortes
CUT_SIGNAL = "histogram"
CUT_THRESHOLD = 0.04
CONFIRM_SIGNAL = "pixel"      # segundo sinal para confirmar (None para nao usar)
CONFIRM_THRESHOLD = 33.3
MIN_GAP = 5                   # dois cortes nunca estao a menos de MIN_GAP frames

# Parte 2 - verificacao com SIFT
VERIFY_CUTS = True
GAP = 4                       # compara t-GAP com t+GAP (evita os frames do flash)
RATIO = 0.7                   # ratio test de Lowe
USE_RANSAC = False
MIN_MATCHES = 31              # com mais matches do que isto e o mesmo shot

# Parte 3 - cenas repetidas
N_KEYFRAMES = 5
KEYFRAME_MARGIN = 0.1         # ignora 10% no inicio e no fim do shot
SCENE_SIFT_FEATURES = 300     # menos keypoints para caber nos 5 minutos

SCENE_KIND = "sift"
SCENE_KEYFRAMES = "multi"
SCENE_HOW = "max"
SCENE_THRESHOLD = 38


def read_frames(path, size=(320, 180)):
    # le o video todo e reduz cada frame para size
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Parte 1 - sinais de dissemelhanca e cortes
# ---------------------------------------------------------------------------

def pixel_difference(f1, f2):
    # media da diferenca absoluta dos niveis de cinzento
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(g1 - g2)
    return float(np.mean(diff))


def histogram_distance(h1, h2):
    # distancia chi^2 entre dois histogramas (normalizados, resultado entre 0 e 1)
    h1 = h1.astype(np.float32).flatten()
    h2 = h2.astype(np.float32).flatten()

    s1 = np.sum(h1)
    s2 = np.sum(h2)
    if s1 > 0:
        h1 = h1 / s1
    if s2 > 0:
        h2 = h2 / s2

    soma = h1 + h2
    mask = soma > 0          # evita dividir por zero nos bins vazios
    dist = 0.5 * np.sum(((h1[mask] - h2[mask]) ** 2) / soma[mask])
    return float(dist)


def compute_edges(frame, low_thresh=100, high_thresh=200, kernel_size=5):
    # arestas de Canny e a versao dilatada (para o ECR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh) > 0
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(edges.astype(np.uint8), kernel) > 0
    return edges, dilated


def ecr_from_edges(e1, d1, e2, d2):
    # Edge Change Ratio: fracao de arestas que entram (x_in) ou saem (x_out)
    n1 = np.sum(e1)
    n2 = np.sum(e2)

    x_in = np.sum(e2 & ~d1)
    x_out = np.sum(e1 & ~d2)

    # se um frame nao tem arestas (ex: frame preto) essa parte conta como 0
    if n2 > 0:
        r_in = x_in / n2
    else:
        r_in = 0.0

    if n1 > 0:
        r_out = x_out / n1
    else:
        r_out = 0.0

    return float(max(r_in, r_out))


def edge_change_ratio(f1, f2, low_thresh=100, high_thresh=200, kernel_size=5):
    e1, d1 = compute_edges(f1, low_thresh, high_thresh, kernel_size)
    e2, d2 = compute_edges(f2, low_thresh, high_thresh, kernel_size)
    return ecr_from_edges(e1, d1, e2, d2)


def compute_pixel_diff_signal(frames):
    d = np.zeros(len(frames), dtype=np.float32)
    for t in range(1, len(frames)):
        d[t] = pixel_difference(frames[t-1], frames[t])
    return d


def compute_histogram_signal(frames, bins=(8, 8, 8), space="rgb"):
    d = np.zeros(len(frames), dtype=np.float32)

    # calcula o histograma de cada frame uma vez so
    hists = []
    for i in range(len(frames)):
        f = frames[i]
        if space == "hsv":
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [bins[0], bins[1]], [0, 180, 0, 256])
        else:
            h = cv2.calcHist([f], [0, 1, 2], None, list(bins), [0, 256, 0, 256, 0, 256])
        hists.append(h)

    for t in range(1, len(frames)):
        d[t] = histogram_distance(hists[t-1], hists[t])
    return d


def compute_ecr_signal(frames, kernel_size=5):
    d = np.zeros(len(frames), dtype=np.float32)

    # as arestas de cada frame tambem so se calculam uma vez
    edges_list = []
    for f in frames:
        edges_list.append(compute_edges(f, kernel_size=kernel_size))

    for t in range(1, len(frames)):
        e1, d1 = edges_list[t-1]
        e2, d2 = edges_list[t]
        d[t] = ecr_from_edges(e1, d1, e2, d2)
    return d


def compute_signal(frames, name):
    if name == "pixel":
        return compute_pixel_diff_signal(frames)
    elif name == "histogram":
        return compute_histogram_signal(frames)
    elif name == "ecr":
        return compute_ecr_signal(frames)
    return None


def detect_cuts(signal, threshold, min_gap=MIN_GAP):
    # corte em t se o sinal passa o limiar e e o maximo numa janela de +-min_gap
    # (assim um pico largo so da um corte)
    cuts = []
    for t in range(len(signal)):
        if signal[t] > threshold:
            lo = max(0, t - min_gap)
            hi = min(len(signal), t + min_gap + 1)
            if signal[t] == np.max(signal[lo:hi]):
                cuts.append(int(t))
    return cuts


def detect_cuts_combined(signal, threshold, confirm, confirm_threshold, min_gap=MIN_GAP):
    # so fica o corte se o segundo sinal tambem passar o seu limiar
    c1 = detect_cuts(signal, threshold, min_gap)
    res = []
    for t in c1:
        if confirm[t] > confirm_threshold:
            res.append(t)
    return res


# ---------------------------------------------------------------------------
# Parte 2 - verificacao dos cortes com SIFT (como no lab3_match.py)
# ---------------------------------------------------------------------------

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2)


def ratio_test(des1, des2, ratio=RATIO):
    # matches que passam o ratio test de Lowe
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m in matches:
        if len(m) == 2:
            if m[0].distance < ratio * m[1].distance:
                good.append(m[0])
    return good


def sift_matches(f1, f2, ratio=RATIO):
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    good = ratio_test(des1, des2, ratio)
    return kp1, kp2, good


def ransac_inliers(kp1, kp2, matches, threshold=5.0):
    # matches que sao inliers de uma homografia (precisa de pelo menos 4)
    if len(matches) < 4:
        return []
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)
    if H is None or mask is None:
        return []
    inliers = []
    mask = mask.ravel()
    for i in range(len(matches)):
        if mask[i]:
            inliers.append(matches[i])
    return inliers


def count_matches(f1, f2, use_ransac=USE_RANSAC):
    kp1, kp2, good = sift_matches(f1, f2)
    if use_ransac:
        return len(ransac_inliers(kp1, kp2, good))
    return len(good)


def candidate_matches(frames, cuts, gap=GAP, use_ransac=USE_RANSAC):
    # numero de matches entre t-gap e t+gap para cada candidato
    res = []
    for t in cuts:
        idx1 = max(0, t - gap)
        idx2 = min(len(frames) - 1, t + gap)
        n = count_matches(frames[idx1], frames[idx2], use_ransac)
        res.append(n)
    return res


def verify_cuts(frames, cuts, min_matches=MIN_MATCHES, gap=GAP, use_ransac=USE_RANSAC):
    # se os frames antes e depois tem muitos matches e o mesmo shot -> rejeita
    counts = candidate_matches(frames, cuts, gap, use_ransac)
    finais = []
    for i in range(len(cuts)):
        if counts[i] < min_matches:
            finais.append(cuts[i])
    return finais


# ---------------------------------------------------------------------------
# Parte 3 - keyframes, semelhanca entre shots e agrupamento
# ---------------------------------------------------------------------------

scene_sift = cv2.SIFT_create(nfeatures=SCENE_SIFT_FEATURES)


def middle_keyframe(shot):
    meio = (shot["start"] + shot["end"]) // 2
    return [meio]


def sample_keyframes(shot, n=N_KEYFRAMES, margin=KEYFRAME_MARGIN):
    # n keyframes espalhados pelo shot, sem as pontas
    start = shot["start"]
    end = shot["end"]
    tam = end - start + 1
    lo = start + int(margin * tam)
    hi = end - int(margin * tam)
    if n == 1 or hi <= lo:      # shot muito curto: fica so o do meio
        return middle_keyframe(shot)

    kfs = []
    vals = np.linspace(lo, hi, n)
    for v in vals:
        idx = int(round(v))
        if idx not in kfs:
            kfs.append(idx)
    return kfs


def shot_keyframes(gt_shots, mode="multi", n=N_KEYFRAMES):
    res = []
    for s in gt_shots:
        if mode == "middle":
            res.append(middle_keyframe(s))
        else:
            res.append(sample_keyframes(s, n))
    return res


def combine_keyframe_sims(sims, how="max"):
    # junta as semelhancas dos keyframes de dois shots num so valor
    # max: o melhor par | mean: media de todos os pares
    # best_mean: para cada keyframe o melhor do outro shot, e depois a media
    sims = np.asarray(sims, dtype=np.float32)
    if how == "max":
        return float(np.max(sims))
    elif how == "mean":
        return float(np.mean(sims))
    elif how == "best_mean":
        m1 = np.mean(np.max(sims, axis=1))
        m2 = np.mean(np.max(sims, axis=0))
        return float((m1 + m2) / 2.0)
    return 0.0


def keyframe_histogram(frame):
    # o mesmo histograma RGB 8x8x8 da Parte 1
    return cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])


def keyframe_descriptors(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, des = scene_sift.detectAndCompute(g, None)
    return des


def keyframe_similarity_matrix(frames, keyframe_list, kind="color"):
    # semelhanca entre todos os pares de keyframes do video
    # color: 1 - chi^2  |  sift: numero de matches que passam o ratio test
    k = len(keyframe_list)
    feats = []
    for t in keyframe_list:
        if kind == "color":
            feats.append(keyframe_histogram(frames[t]))
        else:
            feats.append(keyframe_descriptors(frames[t]))

    sims = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(i + 1, k):
            if kind == "color":
                sims[i, j] = 1.0 - histogram_distance(feats[i], feats[j])
            else:
                sims[i, j] = len(ratio_test(feats[i], feats[j]))
            sims[j, i] = sims[i, j]
    return sims


def shot_similarity_matrix(frames, keyframes, kind="color", how="max", kf_sims=None):
    # matriz shot x shot; keyframes[i] = lista de keyframes do shot i
    # kf_sims pode vir ja calculada (no notebook) para nao repetir os matches
    keyframe_list = []
    for kfs in keyframes:
        for t in kfs:
            keyframe_list.append(t)
    if kf_sims is None:
        kf_sims = keyframe_similarity_matrix(frames, keyframe_list, kind)

    # os keyframes do shot i estao entre first[i] e first[i+1]-1 em keyframe_list
    first = [0]
    for kfs in keyframes:
        first.append(first[-1] + len(kfs))

    n = len(keyframes)
    sim = np.zeros((n, n), dtype=np.float32)
    for a in range(n):
        for b in range(a + 1, n):
            block = kf_sims[first[a]:first[a + 1], first[b]:first[b + 1]]
            val = combine_keyframe_sims(block, how)
            sim[a, b] = val
            sim[b, a] = val

    # um shot e sempre igual a si proprio (a diagonal nao conta no agrupamento)
    max_v = np.max(sim) if np.max(sim) > 0 else 1.0
    for a in range(n):
        sim[a, a] = max_v
    return sim


def process_video(frames, gt_shots):
    # Parte 1: candidatos a corte
    signal = compute_signal(frames, CUT_SIGNAL)
    if CONFIRM_SIGNAL is None:
        cuts = detect_cuts(signal, CUT_THRESHOLD)
    else:
        confirm = compute_signal(frames, CONFIRM_SIGNAL)
        cuts = detect_cuts_combined(signal, CUT_THRESHOLD, confirm, CONFIRM_THRESHOLD)

    # Parte 2: tira os falsos alarmes
    if VERIFY_CUTS:
        cuts = verify_cuts(frames, cuts)

    transitions = []
    for t in cuts:
        transitions.append({"type": "cut", "start": t, "end": t})

    # Parte 3: agrupa os shots do ground truth em cenas
    keyframes = shot_keyframes(gt_shots, SCENE_KEYFRAMES)
    sim = shot_similarity_matrix(frames, keyframes, SCENE_KIND, SCENE_HOW)
    labels = grouping.group_shots(sim, SCENE_THRESHOLD)
    scenes = {str(s["id"]): label for s, label in zip(gt_shots, labels)}
    return transitions, scenes


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}
    for video_path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4"))):
        name = os.path.splitext(os.path.basename(video_path))[0]
        with open(os.path.join(INPUT_DIR, name + ".json")) as f:
            gt = json.load(f)
        frames = read_frames(video_path)
        transitions, scenes = process_video(frames, gt["shots"])
        pred = {"video": name + ".mp4", "transitions": transitions, "scenes": scenes}
        with open(os.path.join(OUTPUT_DIR, name + ".json"), "w") as f:
            json.dump(pred, f, indent=1, default=int)
        results[name] = metrics.evaluate_video(gt, pred)

    dev = [results[n] for n in sorted(results) if n.startswith("dev")]
    test = [results[n] for n in sorted(results) if n.startswith("test")]
    metrics.print_table([(n, results[n]) for n in sorted(results)]
                        + [("DEV", metrics.sum_counts(dev)), ("TEST", metrics.sum_counts(test))])