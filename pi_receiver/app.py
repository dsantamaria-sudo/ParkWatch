"""ParkWatch — receiver on the Raspberry Pi.

ESP32-CAM -> POST /upload (raw photo) -> this service:
  1. Validates the received JPEG.
  2. Filters by daylight (astral, civil twilight + margin).
  3. Resizes/compresses.
  4. Uploads to Azure Blob Storage (`raw/` container) with metadata.
  5. If Blob fails, stores it in a local spool and retries in the background.

Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from astral import LocationInfo
from astral.sun import dawn, dusk
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from PIL import Image

load_dotenv()

# --- Configuration (environment variables, see .env.example) -------------
LAT = float(os.getenv("PW_LAT", "40.4168"))   # placeholder (Madrid): set the real one in .env
LON = float(os.getenv("PW_LON", "-3.7038"))
TZ = ZoneInfo(os.getenv("PW_TZ", "Europe/Madrid"))
MARGIN_MIN = int(os.getenv("PW_MARGIN_MIN", "15"))

CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER = os.getenv("PW_CONTAINER", "raw")

MAX_WIDTH = int(os.getenv("PW_MAX_WIDTH", "1600"))
JPEG_QUALITY = int(os.getenv("PW_JPEG_QUALITY", "80"))
# Clockwise degrees to straighten the photo depending on how the camera is mounted
ROTATE_DEG = int(os.getenv("PW_ROTATE", "180"))
_ROTATIONS = {
    90: Image.Transpose.ROTATE_270,   # PIL rotates counter-clockwise: 270 CCW = 90 CW
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}
if ROTATE_DEG not in (0, *_ROTATIONS):
    raise ValueError(f"PW_ROTATE must be 0, 90, 180 or 270 (got {ROTATE_DEG})")
# Horizontal mirror (left <-> right), applied after rotating
MIRROR = os.getenv("PW_MIRROR", "1") == "1"

SPOOL_DIR = Path(os.getenv("PW_SPOOL_DIR", "./spool"))
RETRY_EVERY_S = int(os.getenv("PW_RETRY_EVERY_S", "120"))
MIN_INPUT_BYTES = 5_000  # drops truncated/corrupt frames from the ESP32

SPOOL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("parkwatch")

_location = LocationInfo("ParkWatch", "Spain", str(TZ), LAT, LON)
_container = BlobServiceClient.from_connection_string(CONN_STR).get_container_client(CONTAINER)
_upload_lock = threading.Lock()

app = FastAPI(title="ParkWatch receiver")


# --- Helpers ----------------------------------------------------------------
def is_daylight(now_utc: datetime) -> bool:
    """True if `now_utc` falls within today's civil twilight +/- margin."""
    local_now = now_utc.astimezone(TZ)
    d = local_now.date()
    window_start = dawn(_location.observer, date=d, tzinfo=TZ) - timedelta(minutes=MARGIN_MIN)
    window_end = dusk(_location.observer, date=d, tzinfo=TZ) + timedelta(minutes=MARGIN_MIN)
    return window_start <= local_now <= window_end


def is_valid_jpeg(data: bytes) -> bool:
    return len(data) >= MIN_INPUT_BYTES and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


def process_image(raw: bytes) -> bytes:
    """Straightens, resizes (if needed) and recompresses to save space in Blob."""
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    if ROTATE_DEG:
        img = img.transpose(_ROTATIONS[ROTATE_DEG])
    if MIRROR:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        img = img.resize((MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return out.getvalue()


def blob_name(ts: datetime) -> str:
    # partitioned by date so thousands of blobs do not pile up in a single "directory"
    return f"{ts:%Y/%m/%d}/parkwatch_{ts:%Y%m%dT%H%M%SZ}.jpg"


def upload_to_blob(name: str, data: bytes, ts: datetime, extra_meta: dict | None = None) -> None:
    metadata = {"captured_utc": ts.isoformat(), **(extra_meta or {})}
    _container.upload_blob(
        name=name,
        data=data,
        overwrite=True,
        metadata=metadata,
        content_settings=ContentSettings(content_type="image/jpeg"),
    )


def spool_path(ts: datetime) -> Path:
    return SPOOL_DIR / f"{ts:%Y%m%dT%H%M%SZ}.jpg"


def retry_spool_loop() -> None:
    """Background thread: retries uploading whatever was spooled because of network/Blob failures."""
    while True:
        time.sleep(RETRY_EVERY_S)
        pending = sorted(SPOOL_DIR.glob("*.jpg"))
        if not pending:
            continue
        log.info("retrying %d spooled photo(s)", len(pending))
        for f in pending:
            try:
                ts = datetime.strptime(f.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                with _upload_lock:
                    upload_to_blob(blob_name(ts), f.read_bytes(), ts, {"retried": "true"})
                f.unlink()
                log.info("spool -> Blob OK: %s", f.name)
            except Exception as exc:  # noqa: BLE001 - stay alive no matter what
                log.warning("spool -> Blob failed for %s: %s", f.name, exc)
                break  # probably no network: do not insist with the rest right now


@app.on_event("startup")
def _on_startup() -> None:
    threading.Thread(target=retry_spool_loop, daemon=True).start()
    pending = len(list(SPOOL_DIR.glob("*.jpg")))
    log.info("ParkWatch receiver ready. Photos pending in spool: %d", pending)


# --- Endpoints ----------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "ok": True,
        "utc": now.isoformat(),
        "daylight_now": is_daylight(now),
        "spool_pending": len(list(SPOOL_DIR.glob("*.jpg"))),
    }


@app.post("/upload")
async def upload(request: Request) -> dict:
    raw = await request.body()
    ts = datetime.now(timezone.utc).replace(microsecond=0)

    if not is_valid_jpeg(raw):
        log.warning("invalid frame discarded (%d bytes)", len(raw))
        raise HTTPException(status_code=400, detail="Invalid or incomplete JPEG")

    if not is_daylight(ts):
        log.info("discarded, outside daylight: %s", ts.isoformat())
        return {"status": "discarded_night", "captured_utc": ts.isoformat()}

    try:
        processed = process_image(raw)
    except Exception as exc:  # noqa: BLE001
        log.error("could not process the image: %s", exc)
        raise HTTPException(status_code=422, detail="Could not process the image") from exc

    name = blob_name(ts)
    try:
        with _upload_lock:
            upload_to_blob(name, processed, ts)
        log.info("uploaded to Blob: %s (%d KB)", name, len(processed) // 1024)
        return {"status": "uploaded", "blob": name, "captured_utc": ts.isoformat()}
    except Exception as exc:  # noqa: BLE001
        spool_path(ts).write_bytes(processed)
        log.error("Blob upload failed, stored in spool (%s): %s", spool_path(ts).name, exc)
        return {"status": "spooled", "captured_utc": ts.isoformat()}
