#!/usr/bin/env python3
"""Fold 1. Declared before this pass is scored.

The wrong term on the four size misses is the stored 12-bit frequency
table. SCALE stays 4096 and is not searched. The quotients stay the
unpacked values. The tenth-pass bucket stays the gauge index. The zoom
blob is copied byte for byte. Q stays 2048. No rung is spent. No band
is coarsened. No wav is written.

Each plane stores the sorted support once. Counts start at 1 for every
symbol in that support and step by 1 after the symbol is coded. The
decoder holds the same update, so the frequency table is not in the
archive. The phase of the sound is unchanged: these are the same
quotients. A file that lands under Opus is not a budget for a rung.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import zlib
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

ROOT = Path(__file__).resolve().parent
CODER = ROOT / "fold_range.exe"
Q = 2048
RATE = 44100
MAGIC = b"FS2A"
RULE = (
    "same quotients and same tenth-pass buckets; zoom blob copied; "
    "support stored; counts start at 1 and step by 1; "
    "12-bit frequency table not stored; SCALE stays 4096; "
    "no rung; no coarsen; no wav"
)

SONGS = (
    ("bruno_mars_just_the_way_you_are", "r26", under.unpack_song),
    ("skrillex_first_of_the_year", "r16", dial.unpack_song),
    ("vnv_nation_beloved", "r17", vowel.unpack_song),
    ("cage_the_elephant_aint_no_rest", "r34", cagequiet.unpack_song),
)


def run_coder(mode: str, blob: bytes) -> bytes:
    done = subprocess.run(
        [str(CODER), mode],
        input=blob,
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        raise SystemExit(f"fold coder {mode} exited {done.returncode}")
    return done.stdout


def pack_plane(values: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(values, dtype=np.int32).ravel()
    if flat.size and (int(flat.min()) < -32768 or int(flat.max()) > 32767):
        raise SystemExit("a quotient does not fit in int16")
    symbols, inverse = np.unique(flat, return_inverse=True)
    uniq = int(symbols.size)
    if uniq == 0 or uniq > 65535:
        raise SystemExit("alphabet does not fit the coder")
    payload_in = struct.pack("<II", int(inverse.size), uniq)
    payload_in += np.ascontiguousarray(inverse, dtype="<u2").tobytes()
    encoded = run_coder("enc", payload_in)
    if len(encoded) < 4:
        raise SystemExit("coder returned a short payload")
    (payload_len,) = struct.unpack_from("<I", encoded, 0)
    payload = encoded[4:]
    if payload_len != len(payload):
        raise SystemExit("coder payload length does not match")
    blob = struct.pack("<II", int(flat.size), uniq)
    blob += symbols.astype("<i2").tobytes()
    blob += struct.pack("<I", payload_len)
    blob += payload
    return blob


def unpack_plane(blob: bytes, offset: int) -> tuple[np.ndarray, int]:
    n, uniq = struct.unpack_from("<II", blob, offset)
    offset += 8
    if uniq == 0 or uniq > 65535:
        raise SystemExit("alphabet does not fit the coder")
    symbols = np.frombuffer(blob[offset:offset + uniq * 2], dtype="<i2").astype(np.int32)
    offset += uniq * 2
    (payload_len,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    payload = blob[offset:offset + payload_len]
    offset += payload_len
    if len(payload) != payload_len:
        raise SystemExit("truncated fold payload")
    query = struct.pack("<III", n, uniq, payload_len) + payload
    decoded = run_coder("dec", query)
    if len(decoded) != n * 2:
        raise SystemExit("decoded fold length does not match")
    inverse = np.frombuffer(decoded, dtype="<u2").astype(np.int32)
    if inverse.size and int(inverse.max()) >= uniq:
        raise SystemExit("decoded symbol left the alphabet")
    return symbols[inverse], offset


def pack_band(plane: np.ndarray) -> bytes:
    parts = [pack_plane(plane[0])]
    if plane.shape[0] == 1:
        parts.append(struct.pack("<H", 0))
        return b"".join(parts)
    buckets = context.context_bucket(plane[:-1]).ravel()
    current = np.ascontiguousarray(plane[1:]).ravel()
    order = np.argsort(buckets, kind="stable")
    ordered_b = buckets[order]
    ordered_c = current[order]
    cuts = np.flatnonzero(np.diff(ordered_b)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [ordered_b.size]))
    parts.append(struct.pack("<H", int(starts.size)))
    for start, end in zip(starts, ends):
        parts.append(struct.pack("<h", int(ordered_b[start])))
        parts.append(pack_plane(ordered_c[start:end]))
    return b"".join(parts)


def unpack_band(blob: bytes, offset: int, frames: int, width: int) -> tuple[np.ndarray, int]:
    first, offset = unpack_plane(blob, offset)
    if first.size != width:
        raise SystemExit("first frame width does not match the band")
    plane = np.zeros((frames, width), dtype=np.int32)
    plane[0] = first
    (n_ctx,) = struct.unpack_from("<H", blob, offset)
    offset += 2
    streams = {}
    cursors = {}
    for _i in range(n_ctx):
        (bucket_id,) = struct.unpack_from("<h", blob, offset)
        offset += 2
        values, offset = unpack_plane(blob, offset)
        streams[int(bucket_id)] = values
        cursors[int(bucket_id)] = 0
    if frames == 1:
        return plane, offset
    for t in range(frames - 1):
        prev_bucket = context.context_bucket(plane[t])
        for k in range(width):
            bucket_id = int(prev_bucket[k])
            cursor = cursors[bucket_id]
            plane[t + 1, k] = streams[bucket_id][cursor]
            cursors[bucket_id] = cursor + 1
    for bucket_id, values in streams.items():
        if cursors[bucket_id] != values.size:
            raise SystemExit("a context stream was not consumed")
    return plane, offset


def pack_song(source: bytes, mid_qq: np.ndarray, side_qq: np.ndarray) -> bytes:
    (zoom_len,) = struct.unpack_from("<I", source, 26)
    edges = struct.unpack_from("<HHHHH", source, 16)
    parts = [MAGIC, source[4:30 + zoom_len]]
    for plane in (mid_qq, side_qq):
        for lo, hi in zip(edges, edges[1:]):
            parts.append(pack_band(plane[:, lo:hi]))
    return b"".join(parts)


def unpack_song(blob: bytes, rate: int):
    if blob[:4] != MAGIC:
        raise SystemExit("not a fold archive")
    q, original, frames = struct.unpack_from("<III", blob, 4)
    offset = 16
    edges = list(struct.unpack_from("<HHHHH", blob, offset))
    offset += 10
    if edges != gauge.band_edges(edges[-1], rate, zoom.HOP):
        raise SystemExit("archive edges are not the octave cuts")
    (zoom_len,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    zoom_raw = zlib.decompress(blob[offset:offset + zoom_len])
    offset += zoom_len
    expect = frames * 4 * 2
    if len(zoom_raw) != expect:
        raise SystemExit("per-band zoom length does not match")
    mid_b = np.frombuffer(zoom_raw[: frames * 4], dtype=np.uint8).astype(np.int16).copy().reshape(frames, 4)
    side_b = np.frombuffer(zoom_raw[frames * 4 :], dtype=np.uint8).astype(np.int16).copy().reshape(frames, 4)
    planes = []
    for _channel in range(2):
        cols = []
        for band_i in range(4):
            width = edges[band_i + 1] - edges[band_i]
            values, offset = unpack_band(blob, offset, frames, width)
            cols.append(values)
        planes.append(np.column_stack(cols))
    if offset != len(blob):
        raise SystemExit("trailing fold bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    if not CODER.is_file():
        raise SystemExit("fold coder is not built")
    probe = run_coder("self", b"")
    if probe != b"":
        raise SystemExit("fold self returned bytes")
    done = subprocess.run([str(CODER), "self"], capture_output=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"fold self exited {done.returncode}")
    values = np.array([0, -2, 3, 3, -2, 0, 5, -2], dtype=np.int32)
    blob = pack_plane(values)
    got, offset = unpack_plane(blob, 0)
    if offset != len(blob) or not np.array_equal(got, values):
        raise SystemExit("fold plane did not round-trip")
    rng = np.random.default_rng(1)
    plane = rng.integers(-4, 5, size=(6, 8), dtype=np.int32)
    band_blob = pack_band(plane)
    back, band_offset = unpack_band(band_blob, 0, 6, 8)
    if band_offset != len(band_blob) or not np.array_equal(back, plane):
        raise SystemExit("fold band did not round-trip")
    print("fold1 self-check ok", flush=True)


def declared() -> dict:
    return {
        "rule": RULE,
        "songs": [item[0] for item in SONGS],
        "q": Q,
        "magic": MAGIC.decode("ascii"),
    }


def main() -> None:
    self_check()
    out = paths.begin_pass("fold1")
    opus = band.opus_table()
    songs = []
    for stem, folder, unpack in SONGS:
        print(f"song {stem}", flush=True)
        source_path = paths.CURRENT / folder / f"{stem}_q{Q}.fspz"
        source = source_path.read_bytes()
        q, original, mid_qq, mid_b, side_qq, side_b = unpack(source, RATE)
        if q != Q:
            raise SystemExit(f"{stem} source archive is not Q=2048")
        (zoom_len,) = struct.unpack_from("<I", source, 26)
        zoom_blob = source[30:30 + zoom_len]
        blob = pack_song(source, mid_qq, side_qq)
        got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, RATE)
        if got_q != Q or got_n != original:
            raise SystemExit(f"{stem} header did not round-trip")
        if not np.array_equal(got_mid, mid_qq) or not np.array_equal(got_side, side_qq):
            raise SystemExit(f"{stem} quotients did not round-trip")
        if not np.array_equal(got_mb, mid_b) or not np.array_equal(got_sb, side_b):
            raise SystemExit(f"{stem} zoom did not round-trip")
        if blob[30:30 + zoom_len] != zoom_blob:
            raise SystemExit(f"{stem} zoom bytes were rewritten")
        archive_path = out / f"{stem}_q{Q}.fs2a"
        archive_path.write_bytes(blob)
        seconds = original / RATE
        opus_n = opus[stem]
        row = {
            "q": Q,
            "source_pass": folder,
            "source_bytes": len(source),
            "archive_bytes": len(blob),
            "opus96_bytes": opus_n,
            "ratio_vs_opus96": len(blob) / opus_n,
            "ratio_vs_source": len(blob) / len(source),
            "under_opus": len(blob) < opus_n,
            "margin_bytes": opus_n - len(blob),
            "kbps": len(blob) * 8 / seconds / 1000,
            "quotients_match": True,
            "zoom_bytes_match": True,
            "archive": str(archive_path),
        }
        songs.append({"stem": stem, "seconds": seconds, "row": row})
        paths.save_score("fold1_report.json", {
            "pin": "AEB2AD",
            "declared": declared(),
            "songs": songs,
        })
        print(
            f"  archive {len(blob)} source {len(source)} opus {opus_n} "
            f"ratio {row['ratio_vs_opus96']:.6f} margin {row['margin_bytes']} "
            f"{row['kbps']:.1f}kbps",
            flush=True,
        )
    print("fold1 done", flush=True)


if __name__ == "__main__":
    main()
