# webapi/main.py
import io
import math
import time
from typing import Tuple, List, Dict

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Optional: enable HEIC/HEIF via Pillow if available
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pass

# Make sure OpenCV doesn't spawn its own threads (we manage threading ourselves)
cv2.setNumThreads(1)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# -------------------- utility & processing helpers --------------------

def ensure_odd(k: int) -> int:
    k = int(k)
    return k if (k % 2 == 1 and k > 0) else max(1, k - 1)

def process_ops(bgr_img: np.ndarray, do_gray: bool, do_blur: bool, ksize: int) -> np.ndarray:
    out = bgr_img
    if do_gray:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if do_blur and ksize > 1:
        k = ensure_odd(ksize)
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out

def split_tiles_with_halo(img: np.ndarray, n_tiles: int, halo: int) -> List[Dict[str, Tuple[int, int]]]:
    """Vertical stripes with halo overlap; returns dicts with 'slice' and 'core' indices."""
    h = img.shape[0]
    n_tiles = max(1, int(n_tiles))
    starts = np.linspace(0, h, n_tiles + 1, dtype=int)
    tiles = []
    for i in range(n_tiles):
        r0, r1 = starts[i], starts[i + 1]
        if r1 <= r0:
            continue
        a = max(0, r0 - halo)
        b = min(h, r1 + halo)
        core_start = r0 - a
        core_end = core_start + (r1 - r0)
        tiles.append({"slice": (a, b), "core": (core_start, core_end)})
    return tiles

def stitch_tiles_vert(tiles_proc: List[np.ndarray]) -> np.ndarray:
    return np.vstack(tiles_proc) if tiles_proc else None

def process_single_thread(img: np.ndarray, do_gray: bool, do_blur: bool, ksize: int):
    t0 = time.perf_counter()
    out = process_ops(img, do_gray, do_blur, ksize)
    dt = time.perf_counter() - t0
    return out, dt

def process_multi_thread(img: np.ndarray, do_gray: bool, do_blur: bool, ksize: int, num_threads: int):
    """Synchronized: preserves tile order via futures/index mapping."""
    if num_threads <= 1:
        return process_single_thread(img, do_gray, do_blur, ksize)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    halo = (ensure_odd(ksize) // 2) if do_blur else 0
    tiles_info = split_tiles_with_halo(img, num_threads, halo)

    def worker(tile_info):
        a, b = tile_info["slice"]
        core_start, core_end = tile_info["core"]
        tile = img[a:b, :, :]
        proc = process_ops(tile, do_gray, do_blur, ksize)
        return proc[core_start:core_end, :, :]

    t0 = time.perf_counter()
    results = [None] * len(tiles_info)
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        fut2idx = {ex.submit(worker, ti): idx for idx, ti in enumerate(tiles_info)}
        for fut in as_completed(fut2idx):
            results[fut2idx[fut]] = fut.result()
    out = stitch_tiles_vert(results)
    out = out[: img.shape[0], : img.shape[1], :]
    dt = time.perf_counter() - t0
    return out, dt

def process_unsync_arrivalorder(img: np.ndarray, do_gray: bool, do_blur: bool, ksize: int, num_threads: int):
    """
    Unsynchronized (arrival-order): tiles are stitched in the order threads finish.
    This can cause stripe misordering when thread completion order differs.
    """
    if num_threads <= 1:
        return process_single_thread(img, do_gray, do_blur, ksize)

    import threading

    halo = (ensure_odd(ksize) // 2) if do_blur else 0
    tiles_info = split_tiles_with_halo(img, num_threads, halo)

    results: List[np.ndarray] = []
    lock = threading.Lock()

    def worker(tile_info):
        a, b = tile_info["slice"]
        core_start, core_end = tile_info["core"]
        tile = img[a:b, :, :]
        proc = process_ops(tile, do_gray, do_blur, ksize)
        crop = proc[core_start:core_end, :, :]
        # Append by completion (no positional index) -> real schedule-dependent order
        with lock:
            results.append(crop)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(ti,)) for ti in tiles_info]
    for th in threads: th.start()
    for th in threads: th.join()

    out = stitch_tiles_vert(results)
    out = out[: img.shape[0], : img.shape[1], :]
    dt = time.perf_counter() - t0
    return out, dt

def process_unsync_sharedbuffer(img: np.ndarray, do_gray: bool, do_blur: bool, ksize: int, num_threads: int):
    """
    Unsynchronized (shared-buffer race): all threads write full tiles back into a
    shared output buffer (including halo). Overlapping rows race (last-writer-wins),
    producing real seam artifacts without any artificial shuffle.
    """
    if num_threads <= 1:
        return process_single_thread(img, do_gray, do_blur, ksize)

    import threading

    halo = (ensure_odd(ksize) // 2) if do_blur else 0
    tiles_info = split_tiles_with_halo(img, num_threads, halo)

    out = np.empty_like(img)

    def worker(tile_info):
        a, b = tile_info["slice"]
        tile = img[a:b, :, :]
        proc = process_ops(tile, do_gray, do_blur, ksize)
        # Write back *full* tile; halo regions overlap -> last writer wins
        out[a:b, :, :] = proc

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(ti,)) for ti in tiles_info]
    for th in threads: th.start()
    for th in threads: th.join()
    dt = time.perf_counter() - t0
    return out, dt

def encode_image_png(bgr_img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    import base64
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

def decode_image_bytes(data: bytes) -> np.ndarray | None:
    """Try OpenCV first; fall back to Pillow (handles HEIC if pillow-heif registered)."""
    npbuf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        rgb = np.array(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None

# -------------------- API routes --------------------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/process")
async def process_endpoint(
    file: UploadFile = File(...),
    do_gray: int = Form(0),
    do_blur: int = Form(1),
    ksize: int = Form(9),
    num_threads: int = Form(4),
    sync: int = Form(1),     # 1 = synchronized, 0 = unsynchronized
    race: int = Form(1),     # only used when sync=0: 1 = shared-buffer race, 0 = arrival-order
):
    data = await file.read()
    bgr = decode_image_bytes(data)
    if bgr is None:
        return JSONResponse({"error": "Could not decode image (use PNG/JPG/HEIC)."}, status_code=400)

    do_gray_b = bool(int(do_gray))
    do_blur_b = bool(int(do_blur))
    k = ensure_odd(int(ksize))
    p = max(1, int(num_threads))
    sync_b = bool(int(sync))
    race_b = bool(int(race))

    # Always measure single-thread for baseline/speedup
    _, t_single = process_single_thread(bgr, do_gray_b, do_blur_b, k)

    if sync_b:
        out, t_multi = process_multi_thread(bgr, do_gray_b, do_blur_b, k, p)
        mode = "synchronized"
    else:
        if race_b:
            out, t_multi = process_unsync_sharedbuffer(bgr, do_gray_b, do_blur_b, k, p)
            mode = "unsync_sharedbuffer"
        else:
            out, t_multi = process_unsync_arrivalorder(bgr, do_gray_b, do_blur_b, k, p)
            mode = "unsync_arrivalorder"

    speedup = (t_single / t_multi) if t_multi > 0 else float("inf")

    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    return {
        "image": encode_image_png(out),
        "elapsed_single_ms": int(t_single * 1000),
        "elapsed_multi_ms": int(t_multi * 1000),
        "speedup": speedup,
        "mode": mode,
        "params": {
            "do_gray": do_gray_b, "do_blur": do_blur_b,
            "ksize": k, "num_threads": p, "sync": sync_b, "race": race_b
        },
    }

# Compatibility route (accepts the same form fields)
@app.post("/api/run")
async def process_endpoint_compat(
    file: UploadFile = File(...),
    do_gray: int = Form(0),
    do_blur: int = Form(1),
    ksize: int = Form(9),
    num_threads: int = Form(4),
    sync: int = Form(1),
    race: int = Form(1),
):
    return await process_endpoint(file, do_gray, do_blur, ksize, num_threads, sync, race)

@app.get("/")
async def redirect_to_app():
    return RedirectResponse("/app")


# Mount static under /app so it never shadows /api/* or /health
app.mount("/app", StaticFiles(directory="static", html=True), name="static")





