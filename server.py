import uvicorn
from fastapi import FastAPI, Query
import requests
import numpy as np
import cv2
import re
import onnxruntime as ort

# -------------------------
# Model Paths
# -------------------------
DET_MODEL = "models/det.onnx"
REC_MODEL = "models/rec.onnx"
ALPHABET = "models/alphabet.txt"

# -------------------------
# Load alphabet
# -------------------------
with open(ALPHABET, "r", encoding="utf-8") as f:
    alphabet = "".join([ln.strip() for ln in f.readlines()])

blank_id = 0

# -------------------------
# Load sessions
# -------------------------
det = ort.InferenceSession(DET_MODEL, providers=["CPUExecutionProvider"])
rec = ort.InferenceSession(REC_MODEL, providers=["CPUExecutionProvider"])

# -------------------------
# Crop region
# -------------------------
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
    cnts, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for ct in cnts:
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
    import numpy as np
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


def extract_codes(path=None, img=None):
    if img is None:
        img = cv2.imread(path)

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

    # fallback grouping
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

    final_codes = []
    for g in ordered_groups:
        if not g:
            final_codes.append(None)
            continue
        g_sorted = sorted(g, key=lambda x: x["cy"])
        raw = g_sorted[0]["text"]
        final_codes.append(clean_code_basic(raw))

    return final_codes


# -------------------------
# API Server
# -------------------------
app = FastAPI()


@app.get("/extract")
def extract_api(image_url: str):
    resp = requests.get(image_url, timeout=10)
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    codes = extract_codes(img=img)
    return {"codes": codes}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080)
