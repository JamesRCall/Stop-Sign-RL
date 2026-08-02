"""Client wrapper for a remote detector server."""

from __future__ import annotations

import io
import os
import socket
from typing import Any, List, Optional

from PIL import Image
from multiprocessing.connection import Client

from detectors.class_names import ClassReference, resolve_class_id


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
        target_class: ClassReference = "stop sign",
        debug: bool = False,
        timeout_s: float = 30.0,
        authkey: Optional[str] = None,
    ):
        """
        Args:
            server_addr: server://host:port address string.
            conf: Confidence threshold.
            iou: IoU threshold.
            target_class: Source class name or integer id.
            debug: Enable debug logging.
            timeout_s: Socket timeout in seconds.
            authkey: Optional shared secret; defaults to DETECTOR_AUTHKEY.
        """
        self.server_addr = str(server_addr)
        self.conf = float(conf)
        self.iou = float(iou)
        self.debug = bool(debug)
        self.timeout_s = float(timeout_s)
        configured_authkey = authkey if authkey is not None else os.getenv("DETECTOR_AUTHKEY", "")
        self._authkey = configured_authkey.encode("utf-8") if configured_authkey else None

        host, port = self._parse_addr(self.server_addr)
        self._address = (host, port)
        self._conn: Optional[Client] = None
        self.id_to_name: dict[int, str] = {}
        self.target_id = 0
        self._load_metadata(target_class)

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
        conn = Client(self._address, family="AF_INET", authkey=self._authkey)
        self._conn = conn
        return conn

    def _request(self, message: dict) -> Any:
        try:
            conn = self._connect()
            conn.send(message)
            if not conn.poll(self.timeout_s):
                raise TimeoutError(
                    f"remote detector timed out after {self.timeout_s:.1f}s"
                )
            return conn.recv()
        except (EOFError, OSError, socket.timeout, TimeoutError) as exc:
            if self.debug:
                print(f"[RemoteDetectorWrapper] connection error: {exc}")
            try:
                if self._conn is not None:
                    self._conn.close()
            finally:
                self._conn = None
            raise ConnectionError(
                f"remote detector request failed for {self.server_addr}"
            ) from exc

    @staticmethod
    def _raise_server_error(response: Any) -> None:
        if isinstance(response, dict) and response.get("ok") is False:
            raise RuntimeError(
                f"remote detector server error: {response.get('error', 'unknown error')}"
            )

    def _load_metadata(self, target_class: ClassReference) -> None:
        response = self._request({"type": "describe"})
        if not isinstance(response, dict) or not response.get("ok", False):
            raise ConnectionError(
                f"Could not read detector metadata from {self.server_addr}. "
                "Start the repository's current tools/detector_server.py first."
            )
        names = response.get("id_to_name", {}) or {}
        self.id_to_name = {int(k): str(v) for k, v in dict(names).items()}
        self.target_id = self.resolve_class_id(target_class, role="source class")

    def resolve_class_id(self, class_ref: ClassReference, *, role: str = "class") -> int:
        """Resolve a class using metadata supplied by the server."""
        return resolve_class_id(self.id_to_name, class_ref, role=role)

    def _encode_images(self, pil_images: List[Image.Image]) -> List[bytes]:
        payload = []
        for im in pil_images:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            payload.append(buf.getvalue())
        return payload

    def infer_confidence(self, pil_image) -> float:
        """Return the target-class confidence for a single image."""
        return float(self.infer_confidence_batch([pil_image])[0])

    def infer_confidence_batch(self, pil_images) -> list[float]:
        """Return confidences for a list of images."""
        if not pil_images:
            return []
        message = {
            "type": "infer_batch",
            "images": self._encode_images(pil_images),
            "conf": self.conf,
            "iou": self.iou,
            "target_id": int(self.target_id),
        }
        response = self._request(message)
        self._raise_server_error(response)
        if not isinstance(response, list) or len(response) != len(pil_images):
            raise RuntimeError(
                "remote detector returned a malformed confidence response"
            )
        return [float(v) for v in response]

    def infer_detections_batch(self, pil_images) -> list[dict]:
        """
        Return detection summaries for a list of images.

        The structured payload is required for spatially valid
        misclassification objectives.
        """
        if not pil_images:
            return []
        message = {
            "type": "infer_detections_batch",
            "images": self._encode_images(pil_images),
            "conf": self.conf,
            "iou": self.iou,
            "target_id": int(self.target_id),
        }
        response = self._request(message)
        self._raise_server_error(response)
        if not isinstance(response, list) or len(response) != len(pil_images):
            raise RuntimeError(
                "remote detector returned a malformed detection response"
            )
        return list(response)
