#!/usr/bin/env python3
"""Entropy of the four size-miss archives. Not a compressor pass.

Declared before it is run. The misses are Bruno r26, Skrillex r16,
VNV r17, and Cage r34. The quotients and the zoom stay the unpacked
values. Nothing is coarsened. No rung is spent. Cage 2.8-5.5 kHz is
not raised. This script does not pack.

Per archive it measures:
- B, the bucket-conditional entropy of the stored quotients, in bytes.
  The bucket is the one already in the archive: 0 if the previous
  quotient in that bin is 0, otherwise sign * (1 + floor(log2(abs))).
- O, the order-0 entropy of those same quotients, one model per octave
  per channel.
- the zoom blob already stored. Zoom is not recoded.
- T, the symbol tables, frequency tables, and framing bytes already
  stored around the payloads.
- P, the rANS payloads already stored.

fixed = the 26-byte header plus the zoom length word and the zoom blob.

Verdict, first match:
- floor, if B + fixed is already at or over the Opus 96 file. No coder
  of this bucket context can clear the miss while the zoom bytes stay.
  Do not name a new context. Do not pack.
- payload, if B + fixed + T is under the Opus file. The next contract,
  written after this print and not by this script, keeps the same
  quotients, the same buckets, and the same tables, and writes each
  bucket payload at that bucket's entropy.
- tables, if one order-0 model per octave per channel, including its
  own symbol table and frequency table, is under the Opus file. The
  per-bucket tables are the term. The next contract keeps the same
  quotients and uses that one model per band.
- floor otherwise. Neither named coder clears the miss. Do not pack.
"""

from __future__ import annotations

import struct
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

HEADER_BYTES = 26
TARGETS = (
    ("bruno_mars_just_the_way_you_are", "r26", under.unpack_song),
    ("skrillex_first_of_the_year", "r16", dial.unpack_song),
    ("vnv_nation_beloved", "r17", vowel.unpack_song),
    ("cage_the_elephant_aint_no_rest", "r34", cagequiet.unpack_song),
)


def symbol_bits(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    counts = np.unique(values, return_counts=True)[1].astype(np.float64)
    total = float(counts.sum())
    prob = counts / total
    return float(-(prob * np.log2(prob)).sum())


def bucket_ideal(plane: np.ndarray) -> float:
    first = np.ascontiguousarray(plane[0]).ravel()
    ideal = first.size * symbol_bits(first) / 8.0
    if plane.shape[0] == 1:
        return ideal
    buckets = context.context_bucket(plane[:-1]).ravel()
    current = np.ascontiguousarray(plane[1:]).ravel()
    order = np.argsort(buckets, kind="stable")
    ordered_b = buckets[order].astype(np.int32)
    ordered_c = current[order]
    cuts = np.flatnonzero(np.diff(ordered_b)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [ordered_b.size]))
    for start, end in zip(starts, ends):
        chunk = ordered_c[start:end]
        ideal += chunk.size * symbol_bits(chunk) / 8.0
    return ideal


def order0_ideal(plane: np.ndarray) -> float:
    flat = np.ascontiguousarray(plane).ravel()
    return flat.size * symbol_bits(flat) / 8.0


def order0_model_bytes(plane: np.ndarray) -> int:
    uniq = int(np.unique(plane).size)
    return 12 + 4 * uniq


def walk_plane(blob: bytes, offset: int) -> tuple[int, int, int, int]:
    if offset + 8 > len(blob):
        raise SystemExit("truncated plane header")
    _count, uniq = struct.unpack_from("<II", blob, offset)
    offset += 8
    if offset + uniq * 4 + 4 > len(blob):
        raise SystemExit("truncated plane tables")
    offset += uniq * 4
    (payload_len,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    if offset + payload_len > len(blob):
        raise SystemExit("truncated plane payload")
    offset += payload_len
    return offset, uniq * 4, payload_len, 12


def walk_band(blob: bytes, offset: int) -> tuple[int, int, int, int]:
    offset, table, payload, framing = walk_plane(blob, offset)
    if offset + 2 > len(blob):
        raise SystemExit("truncated context count")
    (n_ctx,) = struct.unpack_from("<H", blob, offset)
    offset += 2
    framing += 2
    for _index in range(n_ctx):
        if offset + 2 > len(blob):
            raise SystemExit("truncated bucket id")
        offset += 2
        framing += 2
        offset, band_table, band_payload, band_frame = walk_plane(blob, offset)
        table += band_table
        payload += band_payload
        framing += band_frame
    return offset, table, payload, framing


def band_costs(blob: bytes) -> tuple[int, int, int]:
    if len(blob) < HEADER_BYTES + 4:
        raise SystemExit("archive shorter than the header")
    offset = HEADER_BYTES
    (zoom_len,) = struct.unpack_from("<I", blob, offset)
    offset += 4 + zoom_len
    if offset > len(blob):
        raise SystemExit("truncated zoom blob")
    table = 0
    payload = 0
    framing = 0
    for _band in range(8):
        offset, band_table, band_payload, band_frame = walk_band(blob, offset)
        table += band_table
        payload += band_payload
        framing += band_frame
    if offset != len(blob):
        raise SystemExit(f"band walk ended at {offset}, archive is {len(blob)}")
    return 4 + zoom_len, table + framing, payload


def verdict_name(bucket_floor: float, bucket_with_tables: float, order0_floor: float, opus: int) -> str:
    if bucket_floor >= opus:
        return "floor"
    if bucket_with_tables < opus:
        return "payload"
    if order0_floor < opus:
        return "tables"
    return "floor"


def self_check() -> None:
    rng = np.random.default_rng(7)
    plane = rng.integers(-3, 4, size=(6, 5), dtype=np.int32)
    blob = context.pack_band(plane)
    offset, table, payload, framing = walk_band(blob, 0)
    if offset != len(blob) or table + payload + framing != len(blob):
        raise SystemExit("band walk does not add up")
    if not np.isfinite(bucket_ideal(plane)) or not np.isfinite(order0_ideal(plane)):
        raise SystemExit("band entropy is not finite")
    if verdict_name(10, 11, 15, 12) != "payload":
        raise SystemExit("payload branch did not match")
    if verdict_name(12, 20, 15, 12) != "floor":
        raise SystemExit("bucket floor did not match")
    if verdict_name(10, 30, 15, 20) != "tables":
        raise SystemExit("tables branch did not match")
    if verdict_name(10, 30, 25, 20) != "floor":
        raise SystemExit("blocked floor did not match")
    print("lossy floor self-check ok", flush=True)


def main() -> None:
    self_check()
    opus = band.opus_table()
    songs = []
    for stem, pass_name, unpack in TARGETS:
        print(f"song {stem} {pass_name}", flush=True)
        archive_path = paths.CURRENT / pass_name / f"{stem}_q2048.fspz"
        blob = archive_path.read_bytes()
        _q, _original, mid, _mid_b, side, _side_b = unpack(blob, 44100)
        zoom_stored, model_bucket, payload = band_costs(blob)
        fixed = HEADER_BYTES + zoom_stored
        if fixed + model_bucket + payload != len(blob):
            raise SystemExit(f"{stem} byte groups do not add up to the archive")
        bucket = 0.0
        marginal = 0.0
        order0_model = 0
        edges = gauge.band_edges(mid.shape[1], 44100, zoom.HOP)
        for plane in (mid, side):
            for lo, hi in zip(edges, edges[1:]):
                part = plane[:, lo:hi]
                bucket += bucket_ideal(part)
                marginal += order0_ideal(part)
                order0_model += order0_model_bytes(part)
        opus_n = opus[stem]
        bucket_floor = bucket + fixed
        bucket_with_tables = bucket + fixed + model_bucket
        order0_floor = marginal + fixed + order0_model
        name = verdict_name(bucket_floor, bucket_with_tables, order0_floor, opus_n)
        row = {
            "stem": stem,
            "pass": pass_name,
            "archive_bytes": len(blob),
            "opus_bytes": opus_n,
            "miss_bytes": len(blob) - opus_n,
            "bucket_ideal_bytes": bucket,
            "order0_ideal_bytes": marginal,
            "zoom_stored_bytes": zoom_stored,
            "model_bucket_bytes": model_bucket,
            "payload_bytes": payload,
            "order0_model_bytes": order0_model,
            "bucket_floor_bytes": bucket_floor,
            "bucket_with_tables_bytes": bucket_with_tables,
            "order0_floor_bytes": order0_floor,
            "verdict": name,
        }
        songs.append(row)
        print(
            f"  archive {len(blob)} opus {opus_n} miss {row['miss_bytes']} "
            f"B {bucket:.0f} O {marginal:.0f} tables {model_bucket} payload {payload} "
            f"verdict {name}",
            flush=True,
        )
    path = paths.save_score("lossy_floor_trace.json", {
        "rule": "bucket entropy and order-0 entropy of the four size-miss quotients; zoom stays; no archive",
        "songs": songs,
    })
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
