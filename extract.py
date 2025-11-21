import onnxruntime as ort
import numpy as np
import cv2
import re
import os

DET_MODEL = "models/det.onnx"
REC_MODEL = "models/rec.onnx"
ALPHABET = "models/alphabet.txt"
IMAGE_PATH = ["images/im1.png"]

# --- LOAD ALPHABET ---
with open(ALPHABET, "r", encoding="utf-8") as f:
    alphabet = "".join([ln.strip() for ln in f.readlines()])

blank_id = 0

# --- SESSIONS ---
det = ort.InferenceSession(DET_MODEL, providers=["CPUExecutionProvider"])
rec = ort.InferenceSession(REC_MODEL, providers=["CPUExecutionProvider"])

# --- FIXED CROP (your selection) ---
X1, Y1, X2, Y2 = 12, 425, 989, 455



def resize_pad(img, size=(640, 640)):
    h, w = img.shape[:2]
    tw, th = size
    s = min(tw / w, th / h)
    nw, nh = int(w * s), int(h * s)
    r = cv2.resize(img, (nw, nh))
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas[:nh, :nw] = r
    return canvas, nw, nh


def detect_boxes(img):
    inp, nw, nh = resize_pad(img)
    x = inp.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    out = det.run(None, {det.get_inputs()[0].name: x})[0]

    prob = out[0, 0] if out.ndim == 4 else out[0]
    prob = prob[:nh, :nw]
    prob = cv2.resize(prob, (img.shape[1], img.shape[0]))

    _, b = cv2.threshold((prob * 255).astype(np.uint8), 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for ct in contours:
        if cv2.contourArea(ct) < 200:
            continue
        rect = cv2.minAreaRect(ct)
        boxes.append(cv2.boxPoints(rect).astype(int))

    return sorted(boxes, key=lambda bb: np.mean(bb[:, 0]))


def prepare_rec_input(img, box, height=48):
    x, y, w, h = cv2.boundingRect(box)
    crop = img[y:y+h, x:x+w]
    if crop.size == 0:
        return None
    new_w = max(10, int(w * (height / h)))
    r = cv2.resize(crop, (new_w, height))
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None].astype(np.float32), w


def recognize(inp):
    out = rec.run(None, {rec.get_inputs()[0].name: inp})[0][0]
    ids = np.argmax(out, axis=1)

    chars = []
    prev = -1
    for i in ids:
        if i != prev and i != blank_id:
            idx = i - 1
            if 0 <= idx < len(alphabet):
                chars.append(alphabet[idx])
        prev = i
    return "".join(chars)


def clean_code_basic(s):
    if not s:
        return None
    t = s.upper().replace("I", "1").replace("L", "1").replace(" ", "")

    if t.isdigit() and len(t) == 4 and int(t) > 2000:
        return "G" + t[-3:]

    if re.match(r"^\d{1,4}$", t):
        return "G" + t

    if re.match(r"^G\d{1,4}$", t):
        return t

    return t


def kmeans_1d(xs, k=3, iters=20):
    xs = np.array(xs, dtype=float)
    if xs.size == 0:
        return np.array([], dtype=int), np.array([])
    centers = np.linspace(xs.min(), xs.max(), k)

    for _ in range(iters):
        d = np.abs(xs[:, None] - centers[None, :])
        labels = d.argmin(axis=1)
        for j in range(k):
            if (labels == j).any():
                centers[j] = xs[labels == j].mean()

    return labels, centers


def extract_codes(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    # --- THE ONLY CHANGE: crop before processing ---
    crop = img[Y1:Y2, X1:X2]

    boxes = detect_boxes(crop)
    items = []

    for b in boxes:
        rec_in_w = prepare_rec_input(crop, b)
        if rec_in_w is None:
            continue
        rec_in, box_w = rec_in_w
        txt = recognize(rec_in)

        cx = int(np.mean(b[:, 0]))
        cy = int(np.mean(b[:, 1]))

        items.append({"text": txt, "cx": cx, "cy": cy, "box_w": box_w})

    if not items:
        return [None, None, None]

    xs = np.array([it["cx"] for it in items], float)

    if len(xs) < 3:
        w = crop.shape[1]
        col_w = w // 3
        groups = {0: [], 1: [], 2: []}
        for it in items:
            col = int(min(2, it["cx"] // col_w))
            groups[col].append(it)
        ordered_groups = [groups[0], groups[1], groups[2]]

    else:
        labels, _ = kmeans_1d(xs, k=3)
        groups = {}
        for it, lb in zip(items, labels):
            groups.setdefault(lb, []).append(it)
        ordered_groups = [
            groups[k] for k in sorted(groups.keys(), key=lambda kk: np.mean([it["cx"] for it in groups[kk]]))
        ]

    code_like_widths = []
    for it in items:
        s = it["text"].strip().upper().replace("I", "1").replace("L", "1")
        m = re.match(r"^G?(\d{1,4})$", s)
        if m:
            numchars = len(m.group(1))
            code_like_widths.append(it["box_w"] / max(1, numchars))

    median_char_w = np.median(code_like_widths) if code_like_widths else None

    anchors = []
    for g in ordered_groups:
        if not g:
            anchors.append((99999, None, None))
            continue

        g_sorted = sorted(g, key=lambda x: x["cy"])
        cand = None
        cand_w = None

        for it in g_sorted:
            s = it["text"].strip().upper().replace("I", "1").replace("L", "1")
            if re.match(r"^G?\d{1,4}$", s):
                cand = s
                cand_w = it["box_w"]
                break

        if cand is None:
            best = min(g, key=lambda x: (x["cx"], x["cy"]))
            cand = best["text"].strip()
            cand_w = best["box_w"]

        anchors.append((np.mean([it["cx"] for it in g]), cand, cand_w))

    anchors_sorted = sorted(anchors, key=lambda a: a[0])

    final_codes = []
    for _, raw_text, box_w in anchors_sorted:
        if raw_text is None:
            final_codes.append(None)
            continue

        s = raw_text.strip().upper().replace("I", "1").replace("L", "1")

        fixed = None
        m = re.match(r"^G(\d{3})$", s)

        if m and median_char_w and box_w:
            if (box_w / 3.0) > (1.25 * median_char_w):
                fixed = "G1" + m.group(1)

        final_codes.append(fixed if fixed else clean_code_basic(s))

    return final_codes


if __name__ == "__main__":
    for path in IMAGE_PATH:
        print(extract_codes(path))
