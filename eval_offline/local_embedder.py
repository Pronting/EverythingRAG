"""Offline BGE-M3 embedder for evaluation only.

Loads the local ONNX export from ``eval_tmp_model`` (hardlinks to the HF cache),
does CLS pooling + L2 normalisation.  No network, no fastembed registry.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "eval_tmp_model"


class LocalBgeM3Embedder:
    fingerprint = "local-bge-m3-onnx"
    dim = 1024

    def __init__(self, model_dir: Path = MODEL_DIR, max_length: int = 512, threads: int = 8) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(Path(model_dir) / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_length)
        self._tok.no_padding()
        self.max_length = max_length
        self._lock = threading.Lock()

    def _encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        encs = [self._tok.encode(t) for t in texts]
        width = max(len(e.ids) for e in encs)
        ids = np.zeros((len(encs), width), dtype=np.int64)
        mask = np.zeros((len(encs), width), dtype=np.int64)
        for row, enc in enumerate(encs):
            n = len(enc.ids)
            ids[row, :n] = enc.ids
            mask[row, :n] = 1
        return ids, mask

    def embed(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            ids, mask = self._encode(batch)
            with self._lock:
                hidden = self._sess.run(
                    ["token_embeddings"],
                    {"input_ids": ids, "attention_mask": mask},
                )[0]
            cls = hidden[:, 0, :]
            cls = cls / np.clip(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12, None)
            out.extend(cls.astype(np.float32).tolist())
        return out

    # app Embedder protocol
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
