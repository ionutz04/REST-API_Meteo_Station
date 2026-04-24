#!/usr/bin/env python3
"""
Run cloud_cnn.engine on the Jetson Orin Nano GPU and label live frames
from the MJPEG camera feed.

Requires JetPack 6+ (TensorRT 10) + pycuda. Build the engine first with:
    python convert_to_trt.py --onnx cloud_cnn.onnx --out cloud_cnn.engine

Usage:
    python infer_trt.py --engine cloud_cnn.engine --feed http://127.0.0.1:5000/video_feed
"""
import argparse
import json
import time

import cv2
import numpy as np
import requests
import tensorrt as trt
import pycuda.autoinit  # noqa: F401  (initializes the CUDA context)
import pycuda.driver as cuda

CLASSES = [
    "clear", "thin_high", "layered_mid_low", "convective",
    "precipitating", "fog_haze", "unknown",
]

INPUT_NAME = "input"
LOGITS_NAME = "logits"
RAIN_NAME = "rain_pct"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def grab_frame_from_mjpeg(url: str, timeout: int = 20) -> np.ndarray:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        buf = b""
        for chunk in r.iter_content(chunk_size=4096):
            buf += chunk
            s = buf.find(b"\xff\xd8")
            e = buf.find(b"\xff\xd9", s + 2) if s != -1 else -1
            if s != -1 and e != -1:
                arr = np.frombuffer(buf[s:e + 2], dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    raise RuntimeError("MJPEG stream ended before a full frame was received")


def preprocess(bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    s = 256 / min(h, w)
    img = cv2.resize(img, (int(round(w * s)), int(round(h * s))))
    h, w = img.shape[:2]
    y0, x0 = (h - 224) // 2, (w - 224) // 2
    img = img[y0:y0 + 224, x0:x0 + 224].astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)[None]  # NCHW
    return np.ascontiguousarray(img, dtype=np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class TrtRunner:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self.host_buffers = {}
        self.device_buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape))
            host = cuda.pagelocked_empty(size, dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.host_buffers[name] = (host, shape)
            self.device_buffers[name] = dev
            self.ctx.set_tensor_address(name, int(dev))
            self.bindings.append(int(dev))

    def infer(self, x: np.ndarray):
        in_host, _ = self.host_buffers[INPUT_NAME]
        np.copyto(in_host, x.ravel())
        cuda.memcpy_htod_async(self.device_buffers[INPUT_NAME], in_host, self.stream)
        self.ctx.execute_async_v3(stream_handle=self.stream.handle)

        outputs = {}
        for name, (host, shape) in self.host_buffers.items():
            if name == INPUT_NAME:
                continue
            cuda.memcpy_dtoh_async(host, self.device_buffers[name], self.stream)
            outputs[name] = (host, shape)
        self.stream.synchronize()
        return {n: h.reshape(s).copy() for n, (h, s) in outputs.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="cloud_cnn.engine")
    p.add_argument("--feed", default="http://127.0.0.1:5000/video_feed")
    p.add_argument("--interval", type=float, default=2.0,
                   help="Seconds between inferences; 0 means single-shot")
    args = p.parse_args()

    runner = TrtRunner(args.engine)

    while True:
        try:
            frame = grab_frame_from_mjpeg(args.feed)
            x = preprocess(frame)
            t0 = time.time()
            outs = runner.infer(x)
            dt_ms = (time.time() - t0) * 1000.0

            logits = outs[LOGITS_NAME].reshape(-1)
            rain_pct = float(outs[RAIN_NAME].reshape(-1)[0]) * 100.0
            probs = softmax(logits)
            idx = int(np.argmax(probs))
            print(json.dumps({
                "cloud_class": CLASSES[idx] if idx < len(CLASSES) else str(idx),
                "class_confidence": round(float(probs[idx]), 3),
                "rain_pct": round(rain_pct, 1),
                "infer_ms": round(dt_ms, 2),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }))
        except Exception as e:
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}))

        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
