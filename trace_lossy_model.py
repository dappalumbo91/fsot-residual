#!/usr/bin/env python3
"""Cross-entropy of the frequency tables already stored. Not a compressor pass.

Declared before it is run. trace_lossy_floor.py named a payload contract
for Bruno only: same quotients, same buckets, same tables, payload at
the bucket entropy. That entropy is 744 bytes under Opus. The tables
are the 12-bit frequencies from the eighth-pass normalizer. A coder
that keeps those tables cannot land under their cross-entropy.

This script measures that cross-entropy. It does not change SCALE. It
does not pack. Skrillex, VNV, and Cage stay on the floor verdict from
the previous trace; their numbers are printed with the same formula
and do not open a new contract.

Verdict for a song, first match:
- unreachable, if cross-entropy plus the stored header, zoom, and
  tables is at or over the Opus file. The stored tables cannot clear
  the miss. Do not pack. Do not retune the 12-bit normalizer.
- reachable, if that sum is under the Opus file. The gap between the
  stored payload and the cross-entropy is the only saving a tighter
  coder of these same tables can make. The next contract is that
  coder, written after this print. This script still does not pack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths
import refine_cagequiet as cagequiet
import refine_context as context
import refine_dial as dial
import refine_fullband as band
import refine_gauge as gauge
import refine_under as under
import refine_vowel as vowel
import refine_zoom as zoom

TARGETS = (
    ("bruno_mars_just_the_way_you_are", "r26", under.unpack_song),
    ("skrillex_first_of_the_year", "r16", dial.unpack_song),
    ("vnv_nation_beloved", "r17", vowel.unpack_song),
    ("cage_the_elephant_aint_no_rest", "r34", cagequiet.unpack_song),
)
HEADER_BYTES = 26


def symbol_bits(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    counts = np.unique(values, return_counts=True)[1].astype(np.float64)
    total = float(counts.sum())
    prob = counts / total
    return float(-(prob * np.log2(prob)).sum())


def cross_bytes(chunk: np.ndarray) -> float:
    flat = np.ascontiguousarray(chunk, dtype=np.int32).ravel()
    _symbols, inverse = np.unique(flat, return_inverse=True)
    counts = np.bincount(inverse)
    freq = gauge.normalize(counts).astype(np.float64)
    return float(-(counts * np.log2(freq / gauge.SCALE)).sum()) / 8.0


def plane_costs(plane: np.ndarray) -> tuple[float, float, int]:
    streams = [np.ascontiguousarray(plane[0]).ravel()]
    if plane.shape[0] > 1:
        buckets = context.context_bucket(plane[:-1]).ravel()
        current = np.ascontiguousarray(plane[1:]).ravel()
        order = np.argsort(buckets, kind="stable")
        ordered_b = buckets[order].astype(np.int32)
        ordered_c = current[order]
        cuts = np.flatnonzero(np.diff(ordered_b)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [ordered_b.size]))
        for start, end in zip(starts, ends):
            streams.append(ordered_c[start:end])
    entropy = 0.0
    cross = 0.0
    for stream in streams:
        entropy += stream.size * symbol_bits(stream) / 8.0
        cross += cross_bytes(stream)
    return entropy, cross, len(streams)


def self_check() -> None:
    constant = np.zeros(128, dtype=np.int32)
    if cross_bytes(constant) != 0.0:
        raise SystemExit("a constant stream has cross-entropy")
    half = np.array([0, 1], dtype=np.int32).repeat(2048)
    gap = abs(cross_bytes(half) - 512.0)
    if gap > 1e-6:
        raise SystemExit("a balanced stream is not 512 bytes")
    print("lossy model self-check ok", flush=True)


def main() -> None:
    self_check()
    opus = band.opus_table()
    songs = []
    for stem, pass_name, unpack in TARGETS:
        print(f"song {stem} {pass_name}", flush=True)
        blob = (paths.CURRENT / pass_name / f"{stem}_q2048.fspz").read_bytes()
        _q, _original, mid, _mid_b, side, _side_b = unpack(blob, 44100)
        edges = gauge.band_edges(mid.shape[1], 44100, zoom.HOP)
        entropy = 0.0
        cross = 0.0
        streams = 0
        for plane in (mid, side):
            for lo, hi in zip(edges, edges[1:]):
                band_entropy, band_cross, band_streams = plane_costs(plane[:, lo:hi])
                entropy += band_entropy
                cross += band_cross
                streams += band_streams
        zoom_len = int.from_bytes(blob[HEADER_BYTES:HEADER_BYTES + 4], "little")
        fixed = HEADER_BYTES + 4 + zoom_len
        # Tables and framing sit between the zoom blob and the payloads.
        # Their byte count is archive - fixed - payload, recovered from
        # the floor trace so this script does not invent a second split.
        opus_n = opus[stem]
        row = {
            "stem": stem,
            "pass": pass_name,
            "archive_bytes": len(blob),
            "opus_bytes": opus_n,
            "streams": streams,
            "entropy_bytes": entropy,
            "cross_bytes": cross,
            "fixed_bytes": fixed,
            "quantizer_gap_bytes": cross - entropy,
        }
        songs.append(row)
        print(
            f"  streams {streams} entropy {entropy:.0f} cross {cross:.0f} "
            f"gap {cross - entropy:.0f} fixed {fixed}",
            flush=True,
        )
    floor = {
        row["stem"]: row
        for row in __import__("json").loads(
            (paths.SCORES / "lossy_floor_trace.json").read_text(encoding="utf-8")
        )["songs"]
    }
    for row in songs:
        prior = floor[row["stem"]]
        tables = int(prior["model_bucket_bytes"])
        payload = int(prior["payload_bytes"])
        reached = row["cross_bytes"] + row["fixed_bytes"] + tables
        row["model_bucket_bytes"] = tables
        row["payload_bytes"] = payload
        row["same_table_floor_bytes"] = reached
        row["payload_above_cross_bytes"] = payload - row["cross_bytes"]
        row["verdict"] = "reachable" if reached < prior["opus_bytes"] else "unreachable"
        print(
            f"  {row['stem']} same-table floor {reached:.0f} "
            f"opus {prior['opus_bytes']} payload-above-cross {payload - row['cross_bytes']:.0f} "
            f"verdict {row['verdict']}",
            flush=True,
        )
    path = paths.save_score("lossy_model_trace.json", {
        "rule": "cross-entropy of the stored 12-bit bucket tables; SCALE stays 4096; no archive",
        "songs": songs,
    })
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
