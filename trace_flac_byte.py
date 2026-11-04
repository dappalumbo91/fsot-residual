#!/usr/bin/env python3
"""Byte-plane entropy of the flac1 residual. Not a compressor pass.

Declared before it is run. flac1's first difference left the low byte
near 8 bits under an order-0 coder. This script measures, on that same
residual, the order-0 entropy and the order-1 conditional entropy of
each byte plane. It does not pack an archive, does not try a second
difference, and does not fit a coefficient.

The ideal order-1 size is n * H(byte | previous byte) / 8, summed over
the eight planes. A coder cannot land under that ideal. The print names
whether a context on the existing residual can reach FLAC. It does not
choose a predictor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import flac1
import paths
import refine_fullband as band


def bits(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    kept = counts[counts > 0].astype(np.float64)
    prob = kept / total
    return float(-(prob * np.log2(prob)).sum())


def plane_entropy(plane: bytes) -> tuple[float, float]:
    data = np.frombuffer(plane, dtype=np.uint8)
    h0 = bits(np.bincount(data, minlength=256))
    if data.size < 2:
        return h0, h0
    pairs = (data[:-1].astype(np.int32) << 8) | data[1:].astype(np.int32)
    joint = bits(np.bincount(pairs, minlength=65536))
    previous = bits(np.bincount(data[:-1], minlength=256))
    return h0, joint - previous


def main() -> None:
    flac = flac1.flac_bytes()
    songs = []
    for stem in band.STEMS:
        print(f"song {stem}", flush=True)
        path = band.SONGS[band.STEMS.index(stem)]
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        planes = []
        for stream in (mid, side):
            planes.extend(flac1.split_planes(flac1.zigzag_u32(flac1.difference(stream))))
        rows = []
        ideal0 = 0.0
        ideal1 = 0.0
        for index, plane in enumerate(planes):
            h0, h1 = plane_entropy(plane)
            n = len(plane)
            ideal0 += n * h0 / 8.0
            ideal1 += n * h1 / 8.0
            rows.append({"plane": index, "n": n, "h0": h0, "h1": h1})
            print(f"  p{index} h0 {h0:.3f} h1 {h1:.3f}", flush=True)
        flac_n = flac[stem]
        row = {
            "stem": stem,
            "samples": int(left.size),
            "ideal_order0_bytes": ideal0,
            "ideal_order1_bytes": ideal1,
            "flac_bytes": flac_n,
            "ideal0_over_flac": ideal0 / flac_n,
            "ideal1_over_flac": ideal1 / flac_n,
            "planes": rows,
        }
        songs.append(row)
        print(
            f"  ideal0 {ideal0:.0f} ideal1 {ideal1:.0f} flac {flac_n} "
            f"ratios {row['ideal0_over_flac']:.3f} {row['ideal1_over_flac']:.3f}",
            flush=True,
        )
    path = paths.save_score("flac_byte_trace.json", {
        "rule": "order-0 and order-1 entropy of the flac1 first-difference byte planes; no archive",
        "songs": songs,
    })
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
