from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, PlainTextResponse
import asyncio
import time
from typing import Optional

app = FastAPI()

latest_frame: Optional[bytes] = None
latest_timestamp_us: int = 0
latest_received_at: float = 0.0
frame_counter: int = 0
lock = asyncio.Lock()

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>ESP32-CAM Backend</title>
      <style>
        body { font-family: Arial, sans-serif; background:#111; color:#eee; padding:20px; }
        img { max-width: 100%; height: auto; border: 1px solid #333; border-radius: 8px; }
        a { color: #7dd3fc; }
      </style>
    </head>
    <body>
      <h1>ESP32-CAM Backend Feed</h1>
      <p><a href="/snapshot" target="_blank">Snapshot</a></p>
      <img src="/stream" alt="stream">
      <pre id="status">loading...</pre>
      <script>
        async function refresh() {
          const r = await fetch('/status');
          document.getElementById('status').textContent = await r.text();
        }
        setInterval(refresh, 1000);
        refresh();
      </script>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_frame(request: Request):
    global latest_frame, latest_timestamp_us, latest_received_at, frame_counter

    content_type = request.headers.get("content-type", "")
    if content_type != "image/jpeg":
        raise HTTPException(status_code=415, detail="Expected image/jpeg")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    ts_header = request.headers.get("x-frame-timestamp-us", "0")
    try:
        ts_us = int(ts_header)
    except ValueError:
        ts_us = 0

    async with lock:
        latest_frame = body
        latest_timestamp_us = ts_us
        latest_received_at = time.time()
        frame_counter += 1

    return {"ok": True, "size": len(body), "frames": frame_counter}

@app.get("/snapshot")
async def snapshot():
    async with lock:
        frame = latest_frame

    if frame is None:
        raise HTTPException(status_code=404, detail="No frame yet")

    return Response(content=frame, media_type="image/jpeg")

async def mjpeg_generator():
    boundary = b"--frame\r\n"
    while True:
      async with lock:
          frame = latest_frame

      if frame is None:
          await asyncio.sleep(0.05)
          continue

      yield boundary
      yield b"Content-Type: image/jpeg\r\n"
      yield f"Content-Length: {len(frame)}\r\n\r\n".encode()
      yield frame
      yield b"\r\n"

      await asyncio.sleep(0.03)

@app.get("/stream")
async def stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/status")
async def status():
    async with lock:
        size = len(latest_frame) if latest_frame else 0
        ts = latest_timestamp_us
        rx = latest_received_at
        count = frame_counter

    age = time.time() - rx if rx else -1
    text = (
        f"frames_received: {count}\n"
        f"latest_size: {size}\n"
        f"latest_timestamp_us: {ts}\n"
        f"frame_age_s: {age:.3f}\n"
    )
    return PlainTextResponse(text)