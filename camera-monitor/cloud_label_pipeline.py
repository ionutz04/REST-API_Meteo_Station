#!/usr/bin/env python3
import os
import csv
import json
import time
import base64
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any

import random
import shutil

import requests
from dotenv import load_dotenv

ROUTELLM_URL = "https://routellm.abacus.ai/v1/chat/completions"
API_KEY_ENV = "ROUTE_LLM_API_KEY"

# Coarse mapping from the LLM's qualitative rain class to a numeric target
# the CNN regression head can learn against. These are reasonable priors;
# tweak as you collect ground-truth rain measurements from the meteo station.
RAIN_PCT_MAP = {
    "none": 2.0,
    "low": 20.0,
    "medium": 55.0,
    "high": 85.0,
}

CLOUD_CLASSES = [
    "clear", "thin_high", "layered_mid_low", "convective",
    "precipitating", "fog_haze", "unknown",
]

SYSTEM_PROMPT = """
You are labeling sky-camera images for a meteorological dataset.
Return STRICT JSON only.
Tasks:
1. Classify the image into one coarse cloud class from:
   [\"clear\", \"thin_high\", \"layered_mid_low\", \"convective\", \"precipitating\", \"fog_haze\", \"unknown\"]
2. Estimate cloud_cover_pct as integer 0..100.
3. Estimate rain_potential as one of [\"none\", \"low\", \"medium\", \"high\"].
4. Provide confidence 0..1.
5. Flag image_quality issues from [\"glare\", \"overexposed\", \"underexposed\", \"blur\", \"raindrops_on_lens\", \"obstruction\", \"none\"].
6. Provide brief rationale under 30 words.
JSON schema:
{
  \"cloud_class\": str,
  \"cloud_cover_pct\": int,
  \"rain_potential\": str,
  \"confidence\": float,
  \"image_quality\": [str],
  \"rationale\": str
}
""".strip()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _grab_frame_from_mjpeg(url: str, timeout: int = 20) -> bytes:
    """Pull exactly one JPEG frame from a multipart/x-mixed-replace MJPEG stream."""
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        buf = b""
        # JPEG SOI/EOI markers
        soi = b"\xff\xd8"
        eoi = b"\xff\xd9"
        for chunk in r.iter_content(chunk_size=4096):
            if not chunk:
                continue
            buf += chunk
            start = buf.find(soi)
            if start == -1:
                # keep buffer bounded while waiting for SOI
                if len(buf) > 1 << 20:
                    buf = buf[-(1 << 20):]
                continue
            end = buf.find(eoi, start + 2)
            if end == -1:
                continue
            return buf[start:end + 2]
    raise RuntimeError("MJPEG stream ended before a full frame was received")


def snapshot_to_file(url: str, out_dir: str, timeout: int = 20) -> str:
    ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    fn = os.path.join(out_dir, f"snap_{ts}.jpg")

    # Probe content type to decide between MJPEG stream and a plain JPEG endpoint.
    try:
        head = requests.head(url, timeout=timeout, allow_redirects=True)
        ctype = head.headers.get("Content-Type", "").lower()
    except requests.RequestException:
        ctype = ""

    if "multipart" in ctype or url.rstrip("/").endswith("/video_feed"):
        data = _grab_frame_from_mjpeg(url, timeout=timeout)
    else:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.content
        # If the server returned HTML (e.g. the index page), fall back to MJPEG.
        if not data.startswith(b"\xff\xd8"):
            data = _grab_frame_from_mjpeg(url.rstrip("/") + "/video_feed" if not url.endswith("/video_feed") else url,
                                          timeout=timeout)

    with open(fn, "wb") as f:
        f.write(data)
    return fn


def img_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{b}"


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("model response did not contain JSON")


def call_routellm(api_key: str, model: str, image_path: str, temperature: float = 0.0) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Label this sky image for meteorological use. Respond with strict JSON only."},
                    {"type": "image_url", "image_url": {"url": img_to_data_url(image_path)}},
                ],
            },
        ],
    }
    resp = requests.post(ROUTELLM_URL, headers=headers, data=json.dumps(payload), timeout=120)
    if not resp.ok:
        # Surface the server's error body so auth/model issues are debuggable.
        body = resp.text[:500].replace("\n", " ")
        raise RuntimeError(f"RouteLLM HTTP {resp.status_code}: {body}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    parsed["_raw_model"] = model
    return parsed


def ping_routellm(api_key: str, model: str) -> Dict[str, Any]:
    """Tiny text-only request to verify auth and model access."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": "ping"}],
    }
    resp = requests.post(ROUTELLM_URL, headers=headers, data=json.dumps(payload), timeout=30)
    return {"status": resp.status_code, "body": resp.text[:500]}


def normalize_vote(x: Dict[str, Any]) -> Dict[str, Any]:
    valid_classes = {"clear", "thin_high", "layered_mid_low", "convective", "precipitating", "fog_haze", "unknown"}
    valid_rain = {"none", "low", "medium", "high"}
    valid_quality = {"glare", "overexposed", "underexposed", "blur", "raindrops_on_lens", "obstruction", "none"}

    cloud_class = str(x.get("cloud_class", "unknown")).strip().lower()
    if cloud_class not in valid_classes:
        cloud_class = "unknown"

    rain_potential = str(x.get("rain_potential", "low")).strip().lower()
    if rain_potential not in valid_rain:
        rain_potential = "low"

    try:
        cover = int(round(float(x.get("cloud_cover_pct", 0))))
    except Exception:
        cover = 0
    cover = max(0, min(100, cover))

    try:
        conf = float(x.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    q = x.get("image_quality", ["none"])
    if not isinstance(q, list):
        q = ["none"]
    q = [str(i).strip().lower() for i in q if str(i).strip().lower() in valid_quality]
    q = q or ["none"]

    rationale = str(x.get("rationale", "")).strip()[:200]

    return {
        "cloud_class": cloud_class,
        "cloud_cover_pct": cover,
        "rain_potential": rain_potential,
        "confidence": conf,
        "image_quality": sorted(set(q)),
        "rationale": rationale,
        "_raw_model": x.get("_raw_model", "unknown")
    }


def weighted_majority(votes: List[Dict[str, Any]], field: str) -> str:
    score = {}
    for v in votes:
        key = v[field]
        score[key] = score.get(key, 0.0) + max(v.get("confidence", 0.0), 0.05)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def aggregate_votes(votes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not votes:
        return {
            "cloud_class": "unknown",
            "cloud_cover_pct": None,
            "rain_potential": "low",
            "confidence": 0.0,
            "image_quality": ["none"],
            "consensus": "no_votes"
        }

    cloud_class = weighted_majority(votes, "cloud_class")
    rain_potential = weighted_majority(votes, "rain_potential")
    wsum = sum(max(v["confidence"], 0.05) for v in votes)
    cover = round(sum(v["cloud_cover_pct"] * max(v["confidence"], 0.05) for v in votes) / wsum)

    quality = sorted(set(q for v in votes for q in v["image_quality"]))
    agreement = sum(1 for v in votes if v["cloud_class"] == cloud_class) / len(votes)
    conf = round(sum(v["confidence"] for v in votes) / len(votes), 3)

    return {
        "cloud_class": cloud_class,
        "cloud_cover_pct": int(cover),
        "rain_potential": rain_potential,
        "confidence": conf,
        "image_quality": quality,
        "consensus": round(agreement, 3)
    }


def append_csv(path: str, row: Dict[str, Any], header: List[str]):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def rain_potential_to_pct(rain_potential: str, cloud_cover_pct: int) -> float:
    """Combine the qualitative rain class with cloud cover into a single 0..100 target.

    Heuristic: 70% from the rain-class prior, 30% nudge from how cloudy the sky is.
    Replace this with measured rainfall labels once you have ground truth.
    """
    base = RAIN_PCT_MAP.get(rain_potential, 20.0)
    cover = max(0, min(100, int(cloud_cover_pct)))
    pct = 0.7 * base + 0.3 * cover
    return round(max(0.0, min(100.0, pct)), 2)


def accept_for_dataset(agg: Dict[str, Any], min_conf: float, min_consensus: float) -> bool:
    if agg["cloud_class"] == "unknown":
        return False
    if not isinstance(agg.get("consensus"), (int, float)):
        return False
    if agg["confidence"] < min_conf:
        return False
    if agg["consensus"] < min_consensus:
        return False
    bad_quality = {"blur", "overexposed", "underexposed", "obstruction"}
    if any(q in bad_quality for q in agg.get("image_quality", [])):
        return False
    return True


def add_to_dataset(dataset_dir: str,
                   image_path: str,
                   agg: Dict[str, Any],
                   raw_votes: List[Dict[str, Any]],
                   val_ratio: float,
                   rng: random.Random) -> Dict[str, Any]:
    """Copy the snapshot into an ImageFolder-style tree and append to manifest.csv.

    Layout:
        dataset_dir/
            train/<cloud_class>/<file>.jpg
            val/<cloud_class>/<file>.jpg
            manifest.csv         # one row per accepted sample
    """
    split = "val" if rng.random() < val_ratio else "train"
    cls = agg["cloud_class"]
    out_dir = os.path.join(dataset_dir, split, cls)
    ensure_dir(out_dir)

    fname = os.path.basename(image_path)
    dst = os.path.join(out_dir, fname)
    if os.path.abspath(image_path) != os.path.abspath(dst):
        shutil.copy2(image_path, dst)

    rain_pct = rain_potential_to_pct(agg["rain_potential"], agg["cloud_cover_pct"])
    cls_idx = CLOUD_CLASSES.index(cls) if cls in CLOUD_CLASSES else CLOUD_CLASSES.index("unknown")

    manifest_row = {
        "split": split,
        "image_path": os.path.relpath(dst, dataset_dir),
        "cloud_class": cls,
        "cloud_class_idx": cls_idx,
        "cloud_cover_pct": agg["cloud_cover_pct"],
        "rain_potential": agg["rain_potential"],
        "rain_pct": rain_pct,
        "confidence": agg["confidence"],
        "consensus": agg["consensus"],
        "image_quality": "|".join(agg["image_quality"]),
        "timestamp_utc": iso_now(),
        "votes_json": json.dumps(raw_votes, ensure_ascii=False),
    }
    manifest_header = list(manifest_row.keys())
    append_csv(os.path.join(dataset_dir, "manifest.csv"), manifest_row, manifest_header)

    # Also persist the class index map so training code can stay in sync.
    classes_file = os.path.join(dataset_dir, "classes.json")
    if not os.path.exists(classes_file):
        with open(classes_file, "w") as f:
            json.dump(CLOUD_CLASSES, f, indent=2)

    return manifest_row


def main():
    load_dotenv()

    p = argparse.ArgumentParser(description="Snapshot + Abacus RouteLLM labeling pipeline for cloud dataset bootstrapping")
    p.add_argument("--snapshot-url", default="http://127.0.0.1:5000/video_feed",
                   help="MJPEG stream (e.g. /video_feed from the PySide6 camera app) or a plain JPEG endpoint")
    p.add_argument("--models", default="gpt-5,claude-sonnet-4-6,gemini-3.1-pro",
                   help="Comma-separated RouteLLM model slugs (see https://abacus.ai/help/developer-platform/route-llm). "
                        "Use 'route-llm' to let the router pick.")
    p.add_argument("--snap-dir", default="snapshots")
    p.add_argument("--csv", default="labels.csv")
    p.add_argument("--jsonl", default="labels.jsonl")
    p.add_argument("--interval", type=int, default=0, help="Seconds between captures; 0 means one-shot")
    p.add_argument("--count", type=int, default=1, help="How many snapshots to process")
    p.add_argument("--ping", action="store_true",
                   help="Send a tiny text-only request to verify the API key + model access, then exit")
    p.add_argument("--dataset-dir", default="dataset",
                   help="Output directory for the CNN training dataset (ImageFolder layout + manifest.csv). "
                        "Set to '' to disable dataset writing.")
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="Skip samples whose aggregated label confidence is below this threshold")
    p.add_argument("--min-consensus", type=float, default=0.5,
                   help="Skip samples where models disagree more than this fraction")
    p.add_argument("--val-ratio", type=float, default=0.15,
                   help="Fraction of accepted samples to put into the val/ split")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for the train/val split")
    args = p.parse_args()

    rng = random.Random(args.seed)

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Missing environment variable {API_KEY_ENV} (set it in .env)")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise RuntimeError("No models specified")

    if args.ping:
        for m in models:
            r = ping_routellm(api_key, m)
            print(f"[{m}] HTTP {r['status']}: {r['body']}")
        return

    header = [
        "timestamp_utc", "image_path", "cloud_class", "cloud_cover_pct", "rain_potential",
        "confidence", "consensus", "image_quality", "votes_json"
    ]

    for _ in range(args.count):
        img = snapshot_to_file(args.snapshot_url, args.snap_dir)
        raw_votes = []
        for model in models:
            try:
                v = call_routellm(api_key, model, img)
                raw_votes.append(normalize_vote(v))
            except Exception as e:
                raw_votes.append({
                    "cloud_class": "unknown",
                    "cloud_cover_pct": 0,
                    "rain_potential": "low",
                    "confidence": 0.0,
                    "image_quality": ["none"],
                    "rationale": f"error: {type(e).__name__}: {e}",
                    "_raw_model": model
                })

        agg = aggregate_votes(raw_votes)
        row = {
            "timestamp_utc": iso_now(),
            "image_path": img,
            "cloud_class": agg["cloud_class"],
            "cloud_cover_pct": agg["cloud_cover_pct"],
            "rain_potential": agg["rain_potential"],
            "confidence": agg["confidence"],
            "consensus": agg["consensus"],
            "image_quality": "|".join(agg["image_quality"]),
            "votes_json": json.dumps(raw_votes, ensure_ascii=False)
        }
        append_csv(args.csv, row, header)
        with open(args.jsonl, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False))

        if args.dataset_dir and accept_for_dataset(agg, args.min_confidence, args.min_consensus):
            ds_row = add_to_dataset(args.dataset_dir, img, agg, raw_votes, args.val_ratio, rng)
            print(json.dumps({"dataset": ds_row["image_path"], "split": ds_row["split"],
                              "rain_pct": ds_row["rain_pct"]}, ensure_ascii=False))
        elif args.dataset_dir:
            print(json.dumps({"dataset": "skipped",
                              "reason": "low confidence/consensus or bad quality"}))

        if args.interval > 0:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
