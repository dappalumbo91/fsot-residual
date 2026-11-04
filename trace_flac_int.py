#!/usr/bin/env python3
"""Joint entropy of the flac2 second-difference residual. Not a compressor pass.

Declared before it is run. flac2 stored that residual as four byte planes.
The sum of the plane entropies is not the entropy of the integer. This
script measures the order-0 entropy of the zigzag uint32 itself, mid and
side, and compares the ideal size with FLAC level 8.

Ideal bytes are n * H(integer) / 8, summed over the two streams. A coder
of this residual cannot land under that ideal. The byte-plane sum is
printed so the factorization gap is visible. It is not a second candidate.

This script does not pack, does not try a third difference, does not fit
a coefficient, and does not switch Skrillex back to the first difference.

Verdict, from the seven ratios, first match:
- over: every ideal is above its FLAC file. No coder of this residual
  can pass. Do not pack. Do not score a third difference or an LPC fit.
- under: every ideal is under its FLAC file. The next contract, written
  after this print and not by this script, is one Rice coder of this same
  residual. One k per stream,
  k = clamp(floor(log2(mean(abs(second difference))) + 0.5), 0, 31).
  Unary escape: a quotient >= 255 is 255 ones, a zero, then the zigzag
  integer as raw uint32. k is not searched.
- mixed: at least one song is at or over FLAC and at least one is under.
  A coder of this residual cannot pass the songs whose ideal is over
  FLAC. Do not pack. Do not switch a song to the first difference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import flac1
import flac2
import paths
import refine_fullband as band


def symbol_bits(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    counts = np.unique(values, return_counts=True)[1].astype(np.float64)
    total = float(counts.sum())
    prob = counts / total
    return float(-(prob * np.log2(prob)).sum())


def rice_k(delta: np.ndarray) -> int:
    mean_abs = float(np.mean(np.abs(delta))) if delta.size else 0.0
    if mean_abs <= 0.0:
        return 0
    k = int(np.floor(np.log2(mean_abs) + 0.5))
    return max(0, min(31, k))


def stream_row(samples: np.ndarray) -> dict:
    delta = flac2.second_difference(samples)
    folded = flac1.zigzag_u32(delta)
    h_int = symbol_bits(folded)
    plane_bits = 0.0
    for plane in flac1.split_planes(folded):
        plane_bits += symbol_bits(np.frombuffer(plane, dtype=np.uint8))
    n = int(folded.size)
    return {
        "n": n,
        "h_integer": h_int,
        "h_planes": plane_bits,
        "ideal_integer_bytes": n * h_int / 8.0,
        "ideal_plane_bytes": n * plane_bits / 8.0,
        "mean_abs": float(np.mean(np.abs(delta))) if n else 0.0,
        "rice_k": rice_k(delta),
    }


def verdict_name(ratios: list[float]) -> str:
    if all(ratio > 1.0 for ratio in ratios):
        return "over"
    if all(ratio < 1.0 for ratio in ratios):
        return "under"
    return "mixed"


def self_check() -> None:
    uniform = np.arange(256, dtype=np.uint8)
    if abs(symbol_bits(uniform) - 8.0) > 1e-9:
        raise SystemExit("uniform byte entropy is not 8")
    if symbol_bits(np.zeros(64, dtype=np.uint32)) != 0.0:
        raise SystemExit("a constant integer is not 0 bits")
    if rice_k(np.zeros(8, dtype=np.int64)) != 0:
        raise SystemExit("a zero residual does not take k 0")
    if rice_k(np.full(8, 8, dtype=np.int64)) != 3:
        raise SystemExit("mean abs 8 does not take k 3")
    print("flac integer self-check ok", flush=True)


def main() -> None:
    self_check()
    flac = flac1.flac_bytes()
    songs = []
    for stem in band.STEMS:
        print(f"song {stem}", flush=True)
        path = band.SONGS[band.STEMS.index(stem)]
        left, right, _rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        streams = {"mid": stream_row(mid), "side": stream_row(side)}
        ideal = streams["mid"]["ideal_integer_bytes"] + streams["side"]["ideal_integer_bytes"]
        planes = streams["mid"]["ideal_plane_bytes"] + streams["side"]["ideal_plane_bytes"]
        flac_n = flac[stem]
        row = {
            "stem": stem,
            "ideal_integer_bytes": ideal,
            "ideal_plane_bytes": planes,
            "flac_bytes": flac_n,
            "ideal_over_flac": ideal / flac_n,
            "streams": streams,
        }
        songs.append(row)
        print(
            f"  integer {ideal:.0f} planes {planes:.0f} flac {flac_n} "
            f"ratio {row['ideal_over_flac']:.3f} "
            f"k {streams['mid']['rice_k']}/{streams['side']['rice_k']}",
            flush=True,
        )
    name = verdict_name([row["ideal_over_flac"] for row in songs])
    text = {
        "over": "no coder of the second-difference residual can pass; do not pack; do not score a third difference or an LPC fit",
        "under": "one Rice coder of this same residual; one k per stream from floor(log2(mean abs)+0.5); unary escape at quotient 255; no k search",
        "mixed": "a coder of this residual cannot pass the songs whose ideal is over FLAC; do not pack; do not switch a song to the first difference",
    }[name]
    path = paths.save_score("flac_int_trace.json", {
        "rule": "order-0 entropy of the flac2 zigzag integer; no archive",
        "verdict": name,
        "next": text,
        "songs": songs,
    })
    print(f"verdict {name}", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
