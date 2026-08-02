"""Simple TCP detector server for shared object-detector inference.

Use with --detector-device server://HOST:PORT in training.
"""

import argparse
import io
import os
import socket
import sys
import threading
from pathlib import Path
from typing import List

from PIL import Image
from multiprocessing.connection import Listener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.factory import build_detector


MAX_BATCH_IMAGES = 128
MAX_IMAGE_BYTES = 16 * 1024 * 1024
_DETECTOR_LOCK = threading.Lock()


def _bytes_to_images(blob_list: List[bytes]) -> List[Image.Image]:
    """
    Decode a list of PNG byte blobs into PIL images.

    Args:
        blob_list: List of PNG-encoded byte strings.

    Returns:
        List of decoded PIL images.
    """
    if len(blob_list) > MAX_BATCH_IMAGES:
        raise ValueError(f"batch exceeds {MAX_BATCH_IMAGES} images")
    imgs = []
    for b in blob_list:
        if not isinstance(b, bytes) or len(b) > MAX_IMAGE_BYTES:
            raise ValueError("invalid or oversized encoded image")
        img = Image.open(io.BytesIO(b))
        img.verify()
        img = Image.open(io.BytesIO(b)).convert("RGB")
        imgs.append(img)
    return imgs


def handle_client(conn, det):
    """
    Serve a single client connection until it closes.

    Args:
        conn: Multiprocessing connection object.
        det: Detector wrapper instance.
    """
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if not isinstance(msg, dict):
            continue
        request_type = msg.get("type")
        if request_type in ("infer_batch", "infer_detections_batch"):
            try:
                images = _bytes_to_images(msg.get("images", []))
            except Exception as exc:
                conn.send({"ok": False, "error": f"invalid_images: {exc}"})
                continue
            conf = msg.get("conf")
            iou = msg.get("iou")
            target_id = msg.get("target_id")
            try:
                with _DETECTOR_LOCK:
                    if conf is not None:
                        det.conf = float(conf)
                    if iou is not None:
                        det.iou = float(iou)
                    if target_id is not None:
                        det.target_id = int(target_id)
                    if request_type == "infer_detections_batch":
                        out = det.infer_detections_batch(images)
                    else:
                        out = det.infer_confidence_batch(images)
                conn.send(out)
            except Exception as exc:
                conn.send({"ok": False, "error": f"inference_failed: {exc}"})
        elif request_type == "describe":
            conn.send(
                {
                    "ok": True,
                    "target_id": int(getattr(det, "target_id", 0)),
                    "id_to_name": dict(getattr(det, "id_to_name", {}) or {}),
                }
            )
        elif request_type == "ping":
            conn.send({"ok": True})
        else:
            conn.send({"ok": False, "error": "unknown_request"})


def main() -> int:
    """
    Run a blocking detector server loop.

    Returns:
        Exit code.
    """
    ap = argparse.ArgumentParser(description="YOLO detector server for multi-env training.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5009)
    ap.add_argument("--model", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--target-class", default="stop sign")
    ap.add_argument("--detector", default="yolo",
                    help="Detector backend: yolo, torchvision, or rtdetr.")
    ap.add_argument("--detector-model", default="",
                    help="Torchvision model name (e.g., fasterrcnn_resnet50_fpn_v2).")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--authkey",
        default=os.getenv("DETECTOR_AUTHKEY", ""),
        help="Optional shared secret (or set DETECTOR_AUTHKEY). Required for non-localhost use.",
    )
    args = ap.parse_args()

    if str(args.detector).lower() == "yolo" and not args.model:
        raise ValueError("--model is required for YOLO detectors.")

    det = build_detector(
        detector_type=str(args.detector),
        detector_model=str(args.detector_model) if args.detector_model else None,
        yolo_weights=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        target_class=args.target_class,
        debug=args.debug,
    )

    addr = (args.host, int(args.port))
    authkey = args.authkey.encode("utf-8") if args.authkey else None
    if args.host not in ("127.0.0.1", "localhost", "::1") and authkey is None:
        raise ValueError("non-localhost detector servers require --authkey or DETECTOR_AUTHKEY")
    listener = Listener(addr, family="AF_INET", authkey=authkey)
    print(f"[detector_server] listening on {args.host}:{args.port}")

    while True:
        try:
            conn = listener.accept()
            thread = threading.Thread(
                target=_serve_and_close,
                args=(conn, det),
                daemon=True,
            )
            thread.start()
        except (OSError, socket.error) as e:
            print("[detector_server] socket error:", e)
            break
        except KeyboardInterrupt:
            break

    listener.close()
    return 0


def _serve_and_close(conn, det) -> None:
    try:
        handle_client(conn, det)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
