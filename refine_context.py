#!/usr/bin/env python3
"""Tenth audio pass. Declared before this pass is scored.

The ninth pass made the four misses smaller and left all four over Opus.
The low band still holds the rate. Differences of those quotients were
already measured, and the marginal entropy went up. The conditional
entropy did not. Knowing the scale of the previous quotient in the same
bin reduces the ideal rate. This pass codes that conditional.

The coefficients and the zoom stay as they are. Q stays 2048.
The floor is not moved. Sublime, Clay, and Eminem are repacked from the
eighth pass. The four misses are repacked from the ninth. The sound
does not change.

Context of a quotient, same bin, previous frame:
- 0 if the previous quotient is 0
- otherwise sign(previous) * (1 + floor(log2(|previous|)))

Each octave band has one static order-0 model per context. Bins in the
band share that model. The first frame of a band has no previous frame
and uses one order-0 model. The coder is the project's rANS.
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fspack
import paths
import refine_gauge as gauge
import refine_miss as miss
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPC"


def context_bucket(values: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.int16)
    nz = values != 0
    if not np.any(nz):
        return out
    mag = np.abs(values[nz]).astype(np.int64)
    exp = np.floor(np.log2(mag)).astype(np.int16)
    sign = np.where(values[nz] > 0, np.int16(1), np.int16(-1))
    out[nz] = sign * (exp + np.int16(1))
    return out


def pack_band(plane: np.ndarray) -> bytes:
    parts = [gauge.pack_plane(plane[0])]
    if plane.shape[0] == 1:
        parts.append(struct.pack("<H", 0))
        return b"".join(parts)
    buckets = context_bucket(plane[:-1]).ravel()
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
        parts.append(gauge.pack_plane(ordered_c[start:end]))
    return b"".join(parts)


def unpack_band(blob: bytes, offset: int, frames: int, width: int) -> tuple[np.ndarray, int]:
    first, offset = gauge.unpack_plane(blob, offset)
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
        values, offset = gauge.unpack_plane(blob, offset)
        streams[int(bucket_id)] = values
        cursors[int(bucket_id)] = 0
    if frames == 1:
        return plane, offset
    for t in range(frames - 1):
        prev_bucket = context_bucket(plane[t])
        for k in range(width):
            bucket_id = int(prev_bucket[k])
            cursor = cursors[bucket_id]
            plane[t + 1, k] = streams[bucket_id][cursor]
            cursors[bucket_id] = cursor + 1
    for bucket_id, values in streams.items():
        if cursors[bucket_id] != values.size:
            raise SystemExit("a context stream was not consumed")
    return plane, offset


def pack_song(q: int, original: int, mid_qq, mid_b, side_qq, side_b, rate: int) -> bytes:
    edges = gauge.band_edges(mid_qq.shape[1], rate, zoom.HOP)
    stride = 1 if np.ndim(mid_b) == 1 else int(mid_b.shape[1])
    parts = [MAGIC, struct.pack("<IIII", q, original, int(mid_qq.shape[0]), stride)]
    parts.append(struct.pack("<" + "H" * len(edges), *edges))
    zoom_raw = np.ascontiguousarray(mid_b, dtype=np.uint8).tobytes()
    zoom_raw += np.ascontiguousarray(side_b, dtype=np.uint8).tobytes()
    zoom_blob = zlib.compress(zoom_raw, 9)
    parts.append(struct.pack("<I", len(zoom_blob)))
    parts.append(zoom_blob)
    for plane in (mid_qq, side_qq):
        for lo, hi in zip(edges, edges[1:]):
            parts.append(pack_band(plane[:, lo:hi]))
    return b"".join(parts)


def unpack_song(blob: bytes, rate: int):
    if blob[:4] != MAGIC:
        raise SystemExit("not a context archive")
    q, original, frames, stride = struct.unpack_from("<IIII", blob, 4)
    offset = 20
    edges = list(struct.unpack_from("<HHHHH", blob, offset))
    offset += 10
    if edges != gauge.band_edges(edges[-1], rate, zoom.HOP):
        raise SystemExit("archive edges are not the octave cuts")
    (zoom_len,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    zoom_raw = zlib.decompress(blob[offset:offset + zoom_len])
    offset += zoom_len
    zoom_count = frames * stride
    if len(zoom_raw) != zoom_count * 2:
        raise SystemExit("zoom index length does not match")
    mid_b = np.frombuffer(zoom_raw[:zoom_count], dtype=np.uint8).copy()
    side_b = np.frombuffer(zoom_raw[zoom_count:], dtype=np.uint8).copy()
    if stride > 1:
        mid_b = mid_b.reshape(frames, stride)
        side_b = side_b.reshape(frames, stride)
    planes = []
    for _channel in range(2):
        cols = []
        for band_i in range(4):
            width = edges[band_i + 1] - edges[band_i]
            values, offset = unpack_band(blob, offset, frames, width)
            cols.append(values)
        planes.append(np.column_stack(cols))
    if offset != len(blob):
        raise SystemExit("trailing context bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    gauge.self_check()
    rng = np.random.default_rng(11)
    trial = rng.integers(-6, 8, (24, 8)).astype(np.int32)
    blob = pack_band(trial)
    back, end = unpack_band(blob, 0, trial.shape[0], trial.shape[1])
    if end != len(blob) or not np.array_equal(back, trial):
        raise SystemExit("context band does not round-trip")
    zeros = np.zeros((12, 4), dtype=np.int32)
    zblob = pack_band(zeros)
    zback, zend = unpack_band(zblob, 0, 12, 4)
    if zend != len(zblob) or not np.array_equal(zback, zeros):
        raise SystemExit("a zero band does not round-trip")
    print("context self-check ok", flush=True)


def load_current(stem: str, rate: int):
    if stem in miss.MISSES:
        path = paths.CURRENT / "r9" / f"{stem}_q{Q}.fspz"
        q, original, mid, mid_b, side, side_b = miss.unpack_song(path.read_bytes(), rate)
        wav = paths.CURRENT / "r9" / f"{stem}_q{Q}.wav"
        label = "r9"
    else:
        path = paths.CURRENT / "r8" / f"{stem}_q{Q}.fspz"
        q, original, mid, mid_b, side, side_b = gauge.unpack_song(path.read_bytes(), rate)
        wav = paths.CURRENT / "r7" / f"{stem}_q{Q}.wav"
        label = "r8"
    if q != Q:
        raise SystemExit(f"{stem} is not the Q=2048 archive")
    return path, label, original, mid, mid_b, side, side_b, wav


def main() -> None:
    self_check()
    out = paths.begin_pass("r10")
    opus_bytes, _previous = zoom_opus()
    songs = []
    for path, stem in zip(zoom_songs(), zoom_stems()):
        print(f"song {stem}", flush=True)
        source, label, original, mid, mid_b, side, side_b, wav = load_current(stem, 44100)
        blob = pack_song(Q, original, mid, mid_b, side, side_b, 44100)
        got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, 44100)
        if (
            got_q != Q
            or got_n != original
            or not np.array_equal(got_mid, mid)
            or not np.array_equal(got_side, side)
            or not np.array_equal(np.ascontiguousarray(got_mb), np.ascontiguousarray(mid_b, dtype=np.uint8))
            or not np.array_equal(np.ascontiguousarray(got_sb), np.ascontiguousarray(side_b, dtype=np.uint8))
        ):
            raise SystemExit(f"{stem} context archive does not rebuild")
        (out / f"{stem}_q{Q}.fspz").write_bytes(blob)
        row = {
            "q": Q,
            "archive_bytes": len(blob),
            "source_pass": label,
            "source_bytes": source.stat().st_size,
            "opus96_bytes": opus_bytes[stem],
            "ratio_vs_opus96": len(blob) / opus_bytes[stem],
            "ratio_vs_source": len(blob) / source.stat().st_size,
            "samples_match_source": True,
            "wav": str(wav),
        }
        songs.append({"stem": stem, "row": row})
        paths.save_score("context_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "context": "sign(prev) * (1 + floor(log2(|prev|))), 0 if prev is 0",
                "model": "one static order-0 rANS per octave band per context",
                "coefficients": "unchanged from r8 for the wins and r9 for the misses",
                "q": Q,
            },
            "songs": songs,
        })
        print(
            f"  archive {len(blob)} {label} {source.stat().st_size} opus {opus_bytes[stem]} "
            f"ratio {row['ratio_vs_opus96']:.3f} vs source {row['ratio_vs_source']:.3f}",
            flush=True,
        )
    print("context done", flush=True)


def zoom_opus():
    import refine_zoom as zoom
    return zoom.opus_and_r5()


def zoom_songs():
    import refine_zoom as zoom
    return zoom.SONGS


def zoom_stems():
    import refine_zoom as zoom
    return zoom.STEMS


if __name__ == "__main__":
    main()
