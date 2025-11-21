import uvicorn
import numpy as np
import cv2
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from extract import extract_codes

# 1. Lifecycle management for HTTP Client
@asynccontextmanager
async def lifespan(app: FastAPI):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; OCR-Bot/1.0)",
        "Accept": "*/*"
    }
    async with httpx.AsyncClient(headers=headers, timeout=20.0, follow_redirects=True) as client:
        app.state.http_client = client
        yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def home():
    return {"status": "OCR API running"}

@app.get("/extract")
async def extract_api(
    image_url: str = Query(..., description="Public URL of the image")
):
    # 2. Basic SSRF check
    if "localhost" in image_url or "127.0.0.1" in image_url:
         raise HTTPException(status_code=400, detail="Internal URLs not allowed.")

    client: httpx.AsyncClient = app.state.http_client

    # 3. Async Download
    try:
        resp = await client.get(image_url)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Download failed: {resp.status_code}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Network error: {str(e)}")

    # 4. Decoding
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    # 5. Non-blocking Extraction (Critical Step)
    # We run the CPU-heavy 'extract_codes' in a separate thread so the
    # main asyncio loop stays free to handle other incoming requests.
    try:
        result = await asyncio.to_thread(extract_codes, img=img)
        return {"codes": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080)