#!/usr/bin/env python3
"""
stage0_tensor.py: Shared utility for Stage 0 screen stream tensor.

Manages a growing symbolic tensor (num_capture,) on disk.
Each element is a raw screen capture or tool interaction text.

Tensor layout:
  {TENSOR_DIR}/{TENSOR_UID}/shape         — JSON: [num_capture]
  {TENSOR_DIR}/{TENSOR_UID}/storage/0/data — capture 0
  {TENSOR_DIR}/{TENSOR_UID}/storage/1/data — capture 1
  ...
"""

import fcntl
import json
import os
import time

TENSOR_DIR = os.environ.get(
    "STAGE0_TENSOR_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage0_tensor"),
)
TENSOR_UID = "screen_stream"
LOCK_PATH = os.path.join(TENSOR_DIR, TENSOR_UID, ".lock")


def _tensor_root():
    return os.path.join(TENSOR_DIR, TENSOR_UID)


def _ensure_tensor():
    """Ensure tensor directory exists."""
    root = _tensor_root()
    os.makedirs(os.path.join(root, "storage"), exist_ok=True)


def _read_shape():
    """Read current shape. Returns [0] if tensor is new."""
    shape_path = os.path.join(_tensor_root(), "shape")
    if os.path.isfile(shape_path):
        with open(shape_path, "r") as f:
            return json.loads(f.read())
    return [0]


def _write_shape(shape):
    """Write shape to disk."""
    shape_path = os.path.join(_tensor_root(), "shape")
    with open(shape_path, "w") as f:
        f.write(json.dumps(shape))


def _storage_path(flat_index):
    """Get the storage file path for a flat index."""
    digits = list(str(flat_index))
    return os.path.join(
        _tensor_root(), "storage", os.path.join(*digits), "data",
    )


def append_capture(content: str) -> int:
    """Append a screen capture to the tensor. Returns the new flat index.

    Thread/process safe via file lock.
    """
    _ensure_tensor()
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)

    with open(LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            shape = _read_shape()
            idx = shape[0]

            # Write content
            path = _storage_path(idx)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            # Update shape
            shape[0] = idx + 1
            _write_shape(shape)

            return idx
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def read_capture(idx: int) -> str:
    """Read a capture by flat index."""
    path = _storage_path(idx)
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def num_captures() -> int:
    """Return current number of captures."""
    _ensure_tensor()
    return _read_shape()[0]
