#!/usr/bin/env python3
"""Lossless pass flac2. Declared after the byte trace and before this score.

flac1's first-difference residual was measured before this run
(trace_flac_byte.py). Ideal order-1 coding of those same byte planes,
which no real coder can beat, is still over FLAC level 8 on every song:
1.248, 1.067, 1.118, 1.344, 1.217, 1.159, 1.085. A context coder of
that residual cannot reach FLAC. The low byte is the predictor, not
the coder.

This pass changes one thing. The residual is the fixed second difference,
the integer extrapolation with no fitted coefficient:

  d[0] = x[0]
  d[1] = x[1] - x[0]
  d[n] = x[n] - 2 * x[n-1] + x[n-2]

Mid and side stay the reversible split from deliver.mid_side. The
difference is zigzagged into uint32, split into four little-endian
byte planes, and each plane is fspack.rans_encode. No third difference.
No LPC. No coefficient. No choice among predictors after the sizes.

Magic is FS2D. The pass directory is results/current/flac2. No playback
wav is written. The test is byte-exact PCM and the archive length
against FLAC level 8. A larger archive stays a miss.
"""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import flac1
import fspack
import paths
import refine_fullband as band

MAGIC = b"FS2D"
STREAMS = flac1.STREAMS
PLANE_COUNT = flac1.PLANE_COUNT


def second_difference(samples: np.ndarray) -> np.ndarray:
    data = np.ascontiguousarray(samples, dtype=np.int64)
    out = np.empty(data.size, dtype=np.int64)
    out[0] = data[0]
    if data.size > 1:
        out[1] = data[1] - data[0]
    if data.size > 2:
        out[2:] = data[2:] - (2 * data[1:-1]) + data[:-2]
    return out


def unsecond(delta: np.ndarray) -> np.ndarray:
    data = np.ascontiguousarray(delta, dtype=np.int64)
    first = np.empty(data.size, dtype=np.int64)
    first[0] = data[0]
    if data.size > 1:
        first[1:] = np.cumsum(data[1:])
    return np.cumsum(first)


def pack_song(mid: np.ndarray, side: np.ndarray, rate: int) -> bytes:
    if mid.shape != side.shape or mid.ndim != 1:
        raise SystemExit("mid and side are not one pair of streams")
    parts = [MAGIC, struct.pack("<III", int(mid.size), int(rate), len(STREAMS))]
    for name, stream in zip(STREAMS, (mid, side)):
        folded = flac1.zigzag_u32(second_difference(stream))
        for index, plane in enumerate(flac1.split_planes(folded)):
            coded = fspack.rans_encode(plane)
            if mid.size > 10000:
                print(f"  {name}{index} {len(plane)} -> {len(coded)}", flush=True)
            parts.append(struct.pack("<I", len(coded)))
            parts.append(coded)
    return b"".join(parts)


def unpack_song(blob: bytes) -> tuple[np.ndarray, np.ndarray, int]:
    if blob[:4] != MAGIC:
        raise SystemExit("not an FS2D archive")
    count, rate, streams = struct.unpack_from("<III", blob, 4)
    if streams != len(STREAMS):
        raise SystemExit("archive does not hold mid and side")
    offset = 16
    restored = []
    for _stream in STREAMS:
        planes = []
        for _plane in range(PLANE_COUNT):
            coded, offset = flac1._read_blob(blob, offset)
            plane = fspack.rans_decode(coded)
            if len(plane) != count:
                raise SystemExit("decoded plane length does not match the header")
            planes.append(plane)
        restored.append(unsecond(flac1.unzigzag_i64(flac1.join_planes(planes))))
    if offset != len(blob):
        raise SystemExit("trailing lossless bytes")
    return restored[0], restored[1], rate


def self_check() -> None:
    source = np.array([0, 1, 3, 6, 10, -4], dtype=np.int64)
    delta = second_difference(source)
    if not np.array_equal(delta, np.array([0, 1, 1, 1, 1, -18], dtype=np.int64)):
        raise SystemExit("second difference is not the fixed extrapolation")
    if not np.array_equal(unsecond(delta), source):
        raise SystemExit("second difference did not invert")
    rng = np.random.default_rng(34)
    left = rng.integers(-32768, 32768, 3000, dtype=np.int32)
    right = rng.integers(-32768, 32768, 3000, dtype=np.int32)
    mid, side = deliver.mid_side(left, right)
    blob = pack_song(mid, side, 44100)
    got_mid, got_side, rate = unpack_song(blob)
    if rate != 44100 or not np.array_equal(got_mid, mid) or not np.array_equal(got_side, side):
        raise SystemExit("second-difference residual did not round-trip")
    got_l, got_r = deliver.unmid_side(got_mid, got_side)
    if not np.array_equal(got_l, left) or not np.array_equal(got_r, right):
        raise SystemExit("PCM did not round-trip")
    print("flac2 self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("flac2")
    flac = flac1.flac_bytes()
    haar = flac1.haar_bytes()
    songs = []
    for stem in band.STEMS:
        print(f"song {stem}", flush=True)
        started = time.perf_counter()
        path = band.SONGS[band.STEMS.index(stem)]
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        blob = pack_song(mid, side, rate)
        got_mid, got_side, got_rate = unpack_song(blob)
        if got_rate != rate or not np.array_equal(got_mid, mid) or not np.array_equal(got_side, side):
            raise SystemExit(f"{stem} archive does not rebuild the residual")
        got_l, got_r = deliver.unmid_side(got_mid, got_side)
        if not np.array_equal(got_l, left) or not np.array_equal(got_r, right):
            raise SystemExit(f"{stem} archive is not the source PCM")
        (out / f"{stem}.fs2d").write_bytes(blob)
        flac_n = flac[stem]
        haar_n = haar[stem]
        row = {
            "archive_bytes": len(blob),
            "flac_bytes": flac_n,
            "haar_q1_bytes": haar_n,
            "ratio_vs_flac": len(blob) / flac_n,
            "ratio_vs_haar_q1": len(blob) / haar_n,
            "under_flac": len(blob) < flac_n,
            "seconds": left.size / rate,
            "elapsed_s": round(time.perf_counter() - started, 2),
        }
        songs.append({"stem": stem, "source": str(path), "row": row})
        paths.save_score("flac2_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "predictor": "reversible mid/side, fixed second difference, zigzag uint32, four byte planes",
                "why": "ideal order-1 of the first-difference residual is still over FLAC on every song",
                "coder": "fspack.rans_encode static order-0",
                "magic": "FS2D",
                "not_done": "no third difference, no LPC, no coefficient fit, no playback wav",
                "incumbent": "FLAC level 8 in codec_sizes.json",
            },
            "songs": songs,
        })
        print(
            f"  archive {len(blob)} flac {flac_n} ratio {row['ratio_vs_flac']:.3f} "
            f"haar {haar_n} ratio {row['ratio_vs_haar_q1']:.3f} "
            f"exact {row['elapsed_s']:.1f}s",
            flush=True,
        )
    print("flac2 done", flush=True)


if __name__ == "__main__":
    main()
