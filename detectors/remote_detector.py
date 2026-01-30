"""Client wrapper for a remote YOLO detector server."""

from __future__ import annotations

import io
import socket
from typing import List, Optional

from PIL import Image
from multiprocessing.connection import Client


class RemoteDetectorWrapper:
    """
    Simple client that sends images to a detector server.

    Server address format: server://host:port
    """

    def __init__(
        self,
        server_addr: str,
        conf: float = 0.10,
        iou: float = 0.45,
        target_id: int = 11,
        debug: bool = False,
        timeout_s: float = 30.0,
    ):
        """
        Args:
            server_addr: server://host:port address string.
            conf: Confidence threshold.
            iou: IoU threshold.
            target_id: Target class id.
            debug: Enable debug logging.
            timeout_s: Socket timeout in seconds.
        """
        self.server_addr = str(server_addr)
        self.conf = float(conf)
        self.iou = float(iou)
        self.target_id = int(target_id)
        self.debug = bool(debug)
        self.timeout_s = float(timeout_s)

        host, port = self._parse_addr(self.server_addr)
        self._address = (host, port)
        self._conn: Optional[Client] = None

    def _parse_addr(self, addr: str) -> tuple[str, int]:
        """Parse server://host:port into (host, port)."""
        if not addr.lower().startswith("server://"):
            raise ValueError("server_addr must start with server://")
        host_port = addr[len("server://") :]
        if ":" not in host_port:
            raise ValueError("server_addr must be server://host:port")
        host, port_s = host_port.rsplit(":", 1)
        return host, int(port_s)

    def _connect(self) -> Client:
        """Create or reuse a Client connection."""
        if self._conn is not None:
            return self._conn
        conn = Client(self._address, family="AF_INET")
        self._conn = conn
        return conn

    def _send_images(self, pil_images: List[Image.Image]) -> List[float]:
        payload = []
        for im in pil_images:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            payload.append(buf.getvalue())
        msg = {"type": "infer_batch", "images": payload, "conf": self.conf, "iou": self.iou}
        try:
            conn = self._connect()
            conn.send(msg)
            return list(conn.recv())
        except (EOFError, OSError, socket.timeout) as e:
            if self.debug:
                print(f"[RemoteDetectorWrapper] connection error: {e}")
            try:
                if self._conn is not None:
                    self._conn.close()
            finally:
                self._conn = None
            return [0.0 for _ in pil_images]

    def infer_confidence(self, pil_image) -> float:
        """Return the target-class confidence for a single image."""
        return float(self.infer_confidence_batch([pil_image])[0])

    def infer_confidence_batch(self, pil_images) -> list[float]:
        """Return confidences for a list of images."""
        if not pil_images:
            return []
        return self._send_images(pil_images)

    def infer_detections_batch(self, pil_images) -> list[dict]:
        """
        Return detection summaries for a list of images.

        Remote detector servers may not support full detection payloads, so this
        fallback returns zeroed metrics to keep the env running.
        """
        if not pil_images:
            return []
        return [
            {
                "target_conf": 0.0,
                "target_box": None,
                "top_conf": 0.0,
                "top_class": None,
                "top_box": None,
                "boxes": [],
                "confs": [],
                "clss": [],
            }
            for _ in pil_images
        ]
