#!/usr/bin/env python3
"""
Convert cloud_cnn.onnx -> cloud_cnn.engine (TensorRT) for the Jetson Orin Nano.

Run this on the Jetson itself (TensorRT is preinstalled with JetPack 6).
FP16 by default; pass --int8 for ~2x throughput at a small accuracy cost
(uses calibration images sampled from your dataset/train/).

Usage:
    python convert_to_trt.py --onnx cloud_cnn.onnx --out cloud_cnn.engine
    python convert_to_trt.py --onnx cloud_cnn.onnx --out cloud_cnn.engine --int8 --dataset-dir dataset
"""
import argparse
import os
import sys
from typing import List

import numpy as np
import tensorrt as trt
from PIL import Image

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
INPUT_SHAPE = (1, 3, 224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_calib_images(dataset_dir: str, n: int) -> List[str]:
    train_dir = os.path.join(dataset_dir, "train")
    out: List[str] = []
    for root, _, files in os.walk(train_dir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append(os.path.join(root, fn))
    out.sort()
    return out[:n]


def preprocess(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((256, 256))
    left, top = (256 - 224) // 2, (256 - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[None]  # NCHW
    return np.ascontiguousarray(arr, dtype=np.float32)


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths: List[str], cache_path: str):
        super().__init__()
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        self._cuda = cuda
        self.images = image_paths
        self.cache_path = cache_path
        self.idx = 0
        self.batch_size = 1
        self.device_input = cuda.mem_alloc(int(np.prod(INPUT_SHAPE)) * 4)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        if self.idx >= len(self.images):
            return None
        try:
            arr = preprocess(self.images[self.idx])
        except Exception as e:
            print(f"calib skip {self.images[self.idx]}: {e}", file=sys.stderr)
            self.idx += 1
            return self.get_batch(names)
        self.idx += 1
        if self.idx % 25 == 0:
            print(f"calib {self.idx}/{len(self.images)}")
        self._cuda.memcpy_htod(self.device_input, arr)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as f:
            f.write(cache)


def build_engine(onnx_path: str, engine_path: str, fp16: bool, int8: bool,
                 calib_images: List[str]) -> None:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    if int8:
        if not builder.platform_has_fast_int8:
            raise RuntimeError("Platform has no fast INT8")
        if not calib_images:
            raise RuntimeError("INT8 needs --dataset-dir with images under train/")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Int8Calibrator(calib_images, engine_path + ".calib")
        print(f"INT8 enabled with {len(calib_images)} calibration images")

    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(inp.name, INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build returned None")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"wrote {engine_path}  size={os.path.getsize(engine_path)/1e6:.1f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="cloud_cnn.onnx")
    p.add_argument("--out", default="cloud_cnn.engine")
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--fp32", action="store_true", help="Disable FP16 (default is FP16)")
    p.add_argument("--calib-count", type=int, default=200)
    args = p.parse_args()

    calib = list_calib_images(args.dataset_dir, args.calib_count) if args.int8 else []
    build_engine(args.onnx, args.out, fp16=not args.fp32, int8=args.int8, calib_images=calib)


if __name__ == "__main__":
    main()
