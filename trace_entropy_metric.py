#!/usr/bin/env python3
"""Entropy-metric trace. Declared before this run.

The object stored in place of a compressed file is one float64: the
order-0 Shannon entropy of that file's bytes, in bits per byte. The
file itself is not stored. No song archive is written. A permutation
of a file keeps the metric and changes the bytes. The preimage size
is the number of byte strings with the same histogram.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fspack
import paths

ROOT = Path(__file__).resolve().parent
FOLD1 = paths.CURRENT / "fold1"
OPUS = ROOT / "results" / "reference" / "opus"
STEMS = (
    "bruno_mars_just_the_way_you_are",
    "cage_the_elephant_aint_no_rest",
    "skrillex_first_of_the_year",
    "vnv_nation_beloved",
)
RULE = (
    "store one float64 byte-entropy and not the file; "
    "permutation keeps the metric; no archive"
)


def preimage_bits(counts: np.ndarray) -> float:
    """log2 of the number of strings with this exact histogram."""
    total = int(counts.sum())
    bits = math.lgamma(total + 1) / math.log(2.0)
    for count in counts:
        n = int(count)
        if n > 1:
            bits -= math.lgamma(n + 1) / math.log(2.0)
    return bits


def measure(blob: bytes) -> dict:
    counts = np.bincount(np.frombuffer(blob, dtype=np.uint8), minlength=256)
    entropy = fspack.shannon_entropy(blob)
    ideal = len(blob) * entropy / 8.0
    missing = preimage_bits(counts)
    order = np.arange(len(blob))
    rng = np.random.default_rng(0)
    shuffled = np.frombuffer(blob, dtype=np.uint8).copy()
    rng.shuffle(shuffled)
    same_metric = fspack.shannon_entropy(shuffled.tobytes()) == entropy
    return {
        "file_bytes": len(blob),
        "metric_bytes": 8,
        "entropy_bits_per_byte": entropy,
        "ideal_bytes": ideal,
        "preimage_bits": missing,
        "preimage_bytes": missing / 8.0,
        "shuffle_keeps_metric": bool(same_metric),
        "shuffle_same_bytes": bool(shuffled.tobytes() == blob),
    }


def main() -> None:
    rows = []
    for stem in STEMS:
        fold = (FOLD1 / f"{stem}_q2048.fs2a").read_bytes()
        opus = (OPUS / f"{stem}_opus96.opus").read_bytes()
        fold_row = measure(fold)
        opus_row = measure(opus)
        rows.append({"stem": stem, "fold1": fold_row, "opus96": opus_row})
        print(
            f"{stem} fold1 bytes {fold_row['file_bytes']} "
            f"metric {fold_row['metric_bytes']} "
            f"H {fold_row['entropy_bits_per_byte']:.6f} "
            f"ideal {fold_row['ideal_bytes']:.1f} "
            f"preimage_bytes {fold_row['preimage_bytes']:.1f} "
            f"shuffle_keeps {fold_row['shuffle_keeps_metric']} "
            f"shuffle_same {fold_row['shuffle_same_bytes']}",
            flush=True,
        )
        print(
            f"  opus bytes {opus_row['file_bytes']} "
            f"H {opus_row['entropy_bits_per_byte']:.6f} "
            f"ideal {opus_row['ideal_bytes']:.1f}",
            flush=True,
        )
    metrics = [row["fold1"]["entropy_bits_per_byte"] for row in rows]
    distinct = len({round(value, 6) for value in metrics})
    paths.save_score("entropy_metric_trace.json", {
        "pin": fspack.PIN,
        "declared": {"rule": RULE, "metric": "float64 bits per byte", "archive": False},
        "songs": rows,
        "fold1_metrics_distinct_at_1e-6": distinct,
    })
    print(f"entropy metric trace done distinct {distinct}", flush=True)


if __name__ == "__main__":
    main()
