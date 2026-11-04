#!/usr/bin/env python3
"""Lossless pass flac1. Declared before this pass is scored.

The exact grade's wrong term is the predictor. Q=1 stores a 3-level
integer Haar as int32 inside zlib. That time model is weaker than FLAC.
This pass does not retune the listening ladder, Q, the bucket, or any
scored rung. It does not search predictors.

One predictor:

  reversible mid/side, deliver.mid_side:
    side = L - R
    mid = R + side // 2
  first difference on each of those two streams:
    d[0] = x[0]
    d[n] = x[n] - x[n - 1]
  zigzag of that int64 difference into an unsigned integer:
    (d << 1) XOR (d >> 63)
  stored as uint32, little-endian
  four byte planes per stream, plane 0 the least significant byte
  each plane is fspack.rans_encode (static order-0)

No second difference. No LPC. No coefficient is fitted. No later
choice among predictors. A larger archive than FLAC level 8 stays a miss.

Magic is FSPJ. Framing, not a second model: each rANS blob is preceded
by its own byte length so the eight blobs can be sliced. The pass
directory is results/current/flac1. No playback wav is written. The
test is byte-exact restore of the int16 PCM and the archive length
against the FLAC level 8 sizes in results/scores/codec_sizes.json.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import fspack
import paths
import refine_fullband as band

MAGIC = b"FSPJ"
STREAMS = ("mid", "side")
PLANE_COUNT = 4


def difference(samples: np.ndarray) -> np.ndarray:
    data = np.ascontiguousarray(samples, dtype=np.int64)
    out = np.empty(data.size, dtype=np.int64)
    out[0] = data[0]
    if data.size > 1:
        out[1:] = data[1:] - data[:-1]
    return out


def undifference(delta: np.ndarray) -> np.ndarray:
    return np.cumsum(np.ascontiguousarray(delta, dtype=np.int64), dtype=np.int64)


def zigzag_u32(delta: np.ndarray) -> np.ndarray:
    data = np.ascontiguousarray(delta, dtype=np.int64)
    folded = np.bitwise_xor(np.left_shift(data, 1), np.right_shift(data, 63))
    if int(folded.min()) < 0 or int(folded.max()) > 0xFFFFFFFF:
        raise SystemExit("first difference does not fit in uint32 after zigzag")
    return folded.astype(np.uint32)


def unzigzag_i64(folded: np.ndarray) -> np.ndarray:
    raw = np.ascontiguousarray(folded, dtype=np.uint64)
    magnitude = (raw >> np.uint64(1)).astype(np.int64)
    sign = (-(raw & np.uint64(1)).astype(np.int64))
    return np.bitwise_xor(magnitude, sign)


def split_planes(folded: np.ndarray) -> list[bytes]:
    raw = np.ascontiguousarray(folded, dtype="<u4").view(np.uint8).reshape(-1, PLANE_COUNT)
    return [np.ascontiguousarray(raw[:, plane]).tobytes() for plane in range(PLANE_COUNT)]


def join_planes(planes: list[bytes]) -> np.ndarray:
    count = len(planes[0])
    raw = np.empty((count, PLANE_COUNT), dtype=np.uint8)
    for plane, blob in enumerate(planes):
        if len(blob) != count:
            raise SystemExit("byte planes differ in length")
        raw[:, plane] = np.frombuffer(blob, dtype=np.uint8)
    return np.ascontiguousarray(raw).view("<u4").reshape(count).copy()


def pack_song(mid: np.ndarray, side: np.ndarray, rate: int) -> bytes:
    if mid.shape != side.shape or mid.ndim != 1:
        raise SystemExit("mid and side are not one pair of streams")
    parts = [MAGIC, struct.pack("<III", int(mid.size), int(rate), len(STREAMS))]
    for name, stream in zip(STREAMS, (mid, side)):
        folded = zigzag_u32(difference(stream))
        for index, plane in enumerate(split_planes(folded)):
            coded = fspack.rans_encode(plane)
            if mid.size > 10000:
                print(f"  {name}{index} {len(plane)} -> {len(coded)}", flush=True)
            parts.append(struct.pack("<I", len(coded)))
            parts.append(coded)
    return b"".join(parts)


def _read_blob(blob: bytes, offset: int) -> tuple[bytes, int]:
    if offset + 4 > len(blob):
        raise SystemExit("truncated plane length")
    (nbytes,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    end = offset + nbytes
    if end > len(blob):
        raise SystemExit("truncated rANS plane")
    return blob[offset:end], end


def unpack_song(blob: bytes) -> tuple[np.ndarray, np.ndarray, int]:
    if blob[:4] != MAGIC:
        raise SystemExit("not an FSPJ archive")
    count, rate, streams = struct.unpack_from("<III", blob, 4)
    if streams != len(STREAMS):
        raise SystemExit("archive does not hold mid and side")
    offset = 16
    restored = []
    for _stream in STREAMS:
        planes = []
        for _plane in range(PLANE_COUNT):
            coded, offset = _read_blob(blob, offset)
            plane = fspack.rans_decode(coded)
            if len(plane) != count:
                raise SystemExit("decoded plane length does not match the header")
            planes.append(plane)
        restored.append(undifference(unzigzag_i64(join_planes(planes))))
    if offset != len(blob):
        raise SystemExit("trailing lossless bytes")
    return restored[0], restored[1], rate


def self_check() -> None:
    folded = zigzag_u32(np.array([0, 1, -1, 131070, -131070], dtype=np.int64))
    back = unzigzag_i64(folded)
    if not np.array_equal(back, np.array([0, 1, -1, 131070, -131070], dtype=np.int64)):
        raise SystemExit("zigzag did not round-trip")
    planes = split_planes(np.array([1, 256], dtype=np.uint32))
    if planes[0] != bytes([1, 0]) or planes[1] != bytes([0, 1]):
        raise SystemExit("plane 0 is not the least significant byte")
    if join_planes(planes).tolist() != [1, 256]:
        raise SystemExit("byte planes did not rejoin")
    rng = np.random.default_rng(31)
    left = rng.integers(-32768, 32768, 4000, dtype=np.int32)
    right = rng.integers(-32768, 32768, 4000, dtype=np.int32)
    mid, side = deliver.mid_side(left, right)
    blob = pack_song(mid, side, 44100)
    got_mid, got_side, rate = unpack_song(blob)
    if rate != 44100 or not np.array_equal(got_mid, mid) or not np.array_equal(got_side, side):
        raise SystemExit("mid/side residual did not round-trip")
    got_l, got_r = deliver.unmid_side(got_mid, got_side)
    if not np.array_equal(got_l, left) or not np.array_equal(got_r, right):
        raise SystemExit("PCM did not round-trip")
    print("flac1 self-check ok", flush=True)


def flac_bytes() -> dict[str, int]:
    payload = json.loads((paths.SCORES / "codec_sizes.json").read_text(encoding="utf-8"))
    return {row["stem"]: int(row["flac"]) for row in payload["rows"]}


def haar_bytes() -> dict[str, int]:
    payload = json.loads((paths.SCORES / "codec_sizes.json").read_text(encoding="utf-8"))
    return {row["stem"]: int(row["fspz_q1"]) for row in payload["rows"]}


def encode_song(stem: str, flac: dict[str, int], haar: dict[str, int], out: Path) -> dict:
    path = band.SONGS[band.STEMS.index(stem)]
    print(f"song {stem}", flush=True)
    started = time.perf_counter()
    left, right, rate = deliver.decode_pcm(path)
    if left.shape != right.shape:
        raise SystemExit(f"{stem} channels differ")
    mid, side = deliver.mid_side(left, right)
    blob = pack_song(mid, side, rate)
    got_mid, got_side, got_rate = unpack_song(blob)
    if got_rate != rate or not np.array_equal(got_mid, mid) or not np.array_equal(got_side, side):
        raise SystemExit(f"{stem} archive does not rebuild the residual")
    got_l, got_r = deliver.unmid_side(got_mid, got_side)
    if not np.array_equal(got_l, left) or not np.array_equal(got_r, right):
        raise SystemExit(f"{stem} archive is not the source PCM")
    (out / f"{stem}.fspj").write_bytes(blob)
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
    print(
        f"  archive {len(blob)} flac {flac_n} ratio {row['ratio_vs_flac']:.3f} "
        f"haar {haar_n} ratio {row['ratio_vs_haar_q1']:.3f} "
        f"exact {row['elapsed_s']:.1f}s",
        flush=True,
    )
    return {"stem": stem, "source": str(path), "row": row}


def main() -> None:
    self_check()
    out = paths.begin_pass("flac1")
    flac = flac_bytes()
    haar = haar_bytes()
    songs = []
    for stem in band.STEMS:
        songs.append(encode_song(stem, flac, haar, out))
        paths.save_score("flac1_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "predictor": "reversible mid/side, then one first difference, zigzag uint32, four byte planes",
                "coder": "fspack.rans_encode static order-0",
                "magic": "FSPJ",
                "not_done": "no second difference, no LPC, no coefficient fit, no playback wav",
                "incumbent": "FLAC level 8 in codec_sizes.json",
            },
            "songs": songs,
        })
    print("flac1 done", flush=True)


if __name__ == "__main__":
    main()
