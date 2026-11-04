#!/usr/bin/env python3
"""Fold 2. Declared before this pass is scored.

The wrong term is the residual. fold1 still stores the MDCT quotient.
The pinned engine puts Acoustics at derived_D_eff("Acoustics"). That
depth divides the octave widths. Each block of that many bins keeps the
parent quotient and stores every other bin as the integer error against
that parent. The parent is the bin at the start of the block. No other
spacing is tried.

The tenth-pass bucket stays the gauge, taken from the previous quotient
in the same bin. The zoom blob is copied. The coder is the fold1 range
coder: support stored as int32, counts start at 1, the 12-bit table is
not stored. SCALE stays 4096. Q stays 2048. Bruno stays at fold1.
No rung. No coarsen. No wav.
"""

from __future__ import annotations

import hashlib
import importlib.util
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fspack
import paths
import refine_cagequiet as cagequiet
import refine_context as context
import refine_dial as dial
import refine_fullband as band
import refine_gauge as gauge
import refine_vowel as vowel
import refine_zoom as zoom

ROOT = Path(__file__).resolve().parent
CODER = ROOT / "fold_range.exe"
Q = 2048
RATE = 44100
MAGIC = b"FS2B"
RULE = (
    "child bin minus the block parent; block length is derived_D_eff(Acoustics); "
    "parent is the first bin of the block; tenth-pass bucket of the previous quotient; "
    "zoom blob copied; support stored; counts start at 1; "
    "12-bit table not stored; Bruno stays at fold1; no rung; no coarsen; no wav"
)

SONGS = (
    ("skrillex_first_of_the_year", "r16", dial.unpack_song),
    ("vnv_nation_beloved", "r17", vowel.unpack_song),
    ("cage_the_elephant_aint_no_rest", "r34", cagequiet.unpack_song),
)


def acoustics_depth() -> int:
    raw = fspack.ENGINE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not digest.startswith(fspack.PIN_PREFIX):
        raise SystemExit("engine pin mismatch")
    spec = importlib.util.spec_from_file_location("fsot_compute_aeb2ad", fspack.ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load the pinned engine")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return int(mod.derived_D_eff("Acoustics"))


def run_coder(mode: str, blob: bytes) -> bytes:
    done = subprocess.run([str(CODER), mode], input=blob, capture_output=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"fold coder {mode} exited {done.returncode}")
    return done.stdout


def fold_plane(plane: np.ndarray, depth: int) -> np.ndarray:
    width = plane.shape[1]
    parent = (np.arange(width) // depth) * depth
    child = (np.arange(width) % depth) != 0
    out = np.array(plane, dtype=np.int64, copy=True)
    out[:, child] = out[:, child] - out[:, parent[child]]
    if out.size and (int(out.min()) < -2147483648 or int(out.max()) > 2147483647):
        raise SystemExit("a folded symbol does not fit in int32")
    return out.astype(np.int32)


def unfold_plane(folded: np.ndarray, depth: int) -> np.ndarray:
    width = folded.shape[1]
    parent = (np.arange(width) // depth) * depth
    child = (np.arange(width) % depth) != 0
    out = np.array(folded, dtype=np.int64, copy=True)
    out[:, child] = out[:, child] + out[:, parent[child]]
    return out.astype(np.int32)


def pack_plane(values: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(values, dtype=np.int32).ravel()
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
    blob += symbols.astype("<i4").tobytes()
    blob += struct.pack("<I", payload_len)
    blob += payload
    return blob


def unpack_plane(blob: bytes, offset: int) -> tuple[np.ndarray, int]:
    n, uniq = struct.unpack_from("<II", blob, offset)
    offset += 8
    if uniq == 0 or uniq > 65535:
        raise SystemExit("alphabet does not fit the coder")
    symbols = np.frombuffer(blob[offset:offset + uniq * 4], dtype="<i4").astype(np.int32)
    offset += uniq * 4
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


def pack_band(plane: np.ndarray, folded: np.ndarray) -> bytes:
    parts = [pack_plane(folded[0])]
    if plane.shape[0] == 1:
        parts.append(struct.pack("<H", 0))
        return b"".join(parts)
    buckets = context.context_bucket(plane[:-1]).ravel()
    current = np.ascontiguousarray(folded[1:]).ravel()
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


def unpack_band(blob: bytes, offset: int, frames: int, width: int, depth: int) -> tuple[np.ndarray, int]:
    first, offset = unpack_plane(blob, offset)
    if first.size != width:
        raise SystemExit("first frame width does not match the band")
    plane = np.zeros((frames, width), dtype=np.int32)
    plane[0] = unfold_plane(first.reshape(1, width), depth)[0]
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
    parent = (np.arange(width) // depth) * depth
    child = (np.arange(width) % depth) != 0
    for t in range(frames - 1):
        prev_bucket = context.context_bucket(plane[t])
        folded = np.empty(width, dtype=np.int64)
        for k in range(width):
            bucket_id = int(prev_bucket[k])
            cursor = cursors[bucket_id]
            folded[k] = int(streams[bucket_id][cursor])
            cursors[bucket_id] = cursor + 1
        restored = folded.copy()
        restored[child] = folded[child] + folded[parent[child]]
        plane[t + 1] = restored
    for bucket_id, values in streams.items():
        if cursors[bucket_id] != values.size:
            raise SystemExit("a context stream was not consumed")
    return plane, offset


def pack_song(source: bytes, mid_qq: np.ndarray, side_qq: np.ndarray, depth: int) -> bytes:
    (zoom_len,) = struct.unpack_from("<I", source, 26)
    edges = struct.unpack_from("<HHHHH", source, 16)
    parts = [MAGIC, source[4:30 + zoom_len]]
    for plane in (mid_qq, side_qq):
        for lo, hi in zip(edges, edges[1:]):
            width = hi - lo
            if width % depth != 0:
                raise SystemExit("an octave width is not a multiple of the acoustics depth")
            block = plane[:, lo:hi]
            parts.append(pack_band(block, fold_plane(block, depth)))
    return b"".join(parts)


def unpack_song(blob: bytes, rate: int, depth: int):
    if blob[:4] != MAGIC:
        raise SystemExit("not a fold2 archive")
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
            values, offset = unpack_band(blob, offset, frames, width, depth)
            cols.append(values)
        planes.append(np.column_stack(cols))
    if offset != len(blob):
        raise SystemExit("trailing fold bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> int:
    if not CODER.is_file():
        raise SystemExit("fold coder is not built")
    done = subprocess.run([str(CODER), "self"], capture_output=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"fold self exited {done.returncode}")
    depth = acoustics_depth()
    for width in (32, 64, 128):
        if width % depth != 0:
            raise SystemExit("acoustics depth does not divide the octave widths")
    plane = np.array(
        [
            [-20000, 20000, 5, 5, 1, 0, 0, 4],
            [-20000, 19900, 5, 7, 1, 0, 3, 4],
        ],
        dtype=np.int32,
    )
    folded = fold_plane(plane, depth)
    if int(folded[0, 1]) != 40000 or not np.array_equal(unfold_plane(folded, depth), plane):
        raise SystemExit("acoustics fold did not invert")
    blob = pack_band(plane, folded)
    back, offset = unpack_band(blob, 0, 2, 8, depth)
    if offset != len(blob) or not np.array_equal(back, plane):
        raise SystemExit("fold2 band did not round-trip")
    print(f"fold2 self-check ok depth {depth}", flush=True)
    return depth


def main() -> None:
    depth = self_check()
    out = paths.begin_pass("fold2")
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
        blob = pack_song(source, mid_qq, side_qq, depth)
        got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, RATE, depth)
        if got_q != Q or got_n != original:
            raise SystemExit(f"{stem} header did not round-trip")
        if not np.array_equal(got_mid, mid_qq) or not np.array_equal(got_side, side_qq):
            raise SystemExit(f"{stem} quotients did not round-trip")
        if not np.array_equal(got_mb, mid_b) or not np.array_equal(got_sb, side_b):
            raise SystemExit(f"{stem} zoom did not round-trip")
        if blob[30:30 + zoom_len] != zoom_blob:
            raise SystemExit(f"{stem} zoom bytes were rewritten")
        archive_path = out / f"{stem}_q{Q}.fs2b"
        archive_path.write_bytes(blob)
        fold1_path = paths.CURRENT / "fold1" / f"{stem}_q{Q}.fs2a"
        fold1_bytes = fold1_path.stat().st_size
        seconds = original / RATE
        opus_n = opus[stem]
        row = {
            "q": Q,
            "depth": depth,
            "source_pass": folder,
            "source_bytes": len(source),
            "fold1_bytes": fold1_bytes,
            "archive_bytes": len(blob),
            "opus96_bytes": opus_n,
            "ratio_vs_opus96": len(blob) / opus_n,
            "ratio_vs_fold1": len(blob) / fold1_bytes,
            "under_opus": len(blob) < opus_n,
            "margin_bytes": opus_n - len(blob),
            "kbps": len(blob) * 8 / seconds / 1000,
            "quotients_match": True,
            "zoom_bytes_match": True,
            "archive": str(archive_path),
        }
        songs.append({"stem": stem, "seconds": seconds, "row": row})
        paths.save_score("fold2_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "rule": RULE,
                "songs": [item[0] for item in SONGS],
                "left_out": "Bruno stays at fold1",
                "q": Q,
                "magic": MAGIC.decode("ascii"),
                "depth": depth,
            },
            "songs": songs,
        })
        print(
            f"  archive {len(blob)} fold1 {fold1_bytes} opus {opus_n} "
            f"ratio {row['ratio_vs_opus96']:.6f} margin {row['margin_bytes']} "
            f"{row['kbps']:.1f}kbps",
            flush=True,
        )
    print("fold2 done", flush=True)


if __name__ == "__main__":
    main()
