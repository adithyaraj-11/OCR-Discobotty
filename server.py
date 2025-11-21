import uvicorn
from fastapi import FastAPI, Request
import requests
import numpy as np
import cv2
import urllib.parse

from extract import extract_codes  

app = FastAPI()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "*/*"
}

@app.get("/")
def home():
    return {"status": "OCR API running"}


@app.get("/extract")
def extract_api(request: Request):

    image_url = request.query_params.get("image_url")

    if not image_url:
        return {"error": "Missing ?image_url="}

    # Decode URL (Discord URLs contain encoded chars)
    image_url = urllib.parse.unquote(image_url)

    # Download the image
    try:
        resp = requests.get(image_url, headers=HEADERS, timeout=20)
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code} while downloading image"}

    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Failed to decode image. Add &format=png maybe."}

    codes = extract_codes(img=img)

    return {"codes": codes}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080)
