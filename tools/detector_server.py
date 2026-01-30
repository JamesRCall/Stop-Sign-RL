"""Simple TCP detector server for shared YOLO inference.

Use with --detector-device server://HOST:PORT in training.
"""

import argparse
import io
import socket
import sys
from pathlib import Path
from typing import List

from PIL import Image
from multiprocessing.connection import Listener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.factory import build_detector


def _bytes_to_images(blob_list: List[bytes]) -> List[Image.Image]:
    """
    Decode a list of PNG byte blobs into PIL images.

    Args:
        blob_list: List of PNG-encoded byte strings.

    Returns:
        List of decoded PIL images.
    """
    imgs = []
    for b in blob_list:
        try:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            imgs.append(img)
        except Exception:
            imgs.append(Image.new("RGB", (64, 64), (0, 0, 0)))
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
        if msg.get("type") == "infer_batch":
            images = _bytes_to_images(msg.get("images", []))
            conf = msg.get("conf")
            iou = msg.get("iou")
            if conf is not None:
                det.conf = float(conf)
            if iou is not None:
                det.iou = float(iou)
            out = det.infer_confidence_batch(images)
            conn.send(out)
        elif msg.get("type") == "ping":
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
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5009)
    ap.add_argument("--model", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--target-class", default="stop sign")
    ap.add_argument("--detector", default="yolo",
                    help="Detector backend: yolo, torchvision, or detr.")
    ap.add_argument("--detector-model", default="",
                    help="Torchvision model name (e.g., fasterrcnn_resnet50_fpn_v2).")
    ap.add_argument("--debug", action="store_true")
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
    listener = Listener(addr, family="AF_INET")
    print(f"[detector_server] listening on {args.host}:{args.port}")

    while True:
        try:
            conn = listener.accept()
            handle_client(conn, det)
            conn.close()
        except (OSError, socket.error) as e:
            print("[detector_server] socket error:", e)
            break
        except KeyboardInterrupt:
            break

    listener.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
