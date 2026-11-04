#!/usr/bin/env python3
"""Eighth audio pass. Declared before this pass is scored.

The seventh pass is the sound. Q=2048, the zoom, and the coefficients
stay as that pass scored them. The mean test is not moved.

The wrong term is the residual code. One zlib stream mixed the four
octave bands and stored every quotient as int32. The band is the gauge.
The decoder already knows it from the bin index, so the rate is the
entropy of the quotient given the band.

This pass, fixed here:
- Same seventh-pass coefficients at Q=2048 only.
- Octave edges are fs/16, fs/8, and fs/4, the Haar cuts.
- Each band is static order-0 rANS, the project's coder, alphabet at most 4096.
- Quotients are int16. A value that does not fit stops the pass.
- Zoom indices stay zlib-9 bytes.
- Playback is the seventh-pass wav. This pass writes the archive only.
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
import deliver
import fspack
import paths
import refine_fullband as band
import refine_mean as mean
import refine_zoom as zoom

ROOT = Path(__file__).resolve().parent
CODER = ROOT / "gauge_rans.exe"
SCALE = 4096
Q = 2048
MAGIC = b"FSPG"


def band_edges(n_bins: int, rate: int, hop: int) -> list[int]:
    edges = [0]
    for divisor in (16, 8, 4):
        cut = rate / divisor
        k_max = int(np.floor(cut * (2 * hop) / rate - 0.5))
        edges.append(k_max + 1)
    edges.append(n_bins)
    return edges


def normalize(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    slots = int(counts.size)
    if slots == 0 or slots > SCALE:
        raise SystemExit("alphabet does not fit the order-0 table")
    if total <= 0:
        raise SystemExit("empty residual")
    raw = counts * SCALE // total
    seen = counts > 0
    raw[seen & (raw == 0)] = 1
    diff = SCALE - int(raw.sum())
    if diff > 0:
        order = np.argsort(-counts, kind="stable")
        k = 0
        while diff > 0:
            raw[int(order[k % slots])] += 1
            diff -= 1
            k += 1
            if k > SCALE * 4:
                raise SystemExit("normalize failed to fill")
    elif diff < 0:
        order = np.argsort(-raw, kind="stable")
        k = 0
        while diff < 0:
            idx = int(order[k % slots])
            floor = 1 if counts[idx] > 0 else 0
            if raw[idx] > floor:
                raw[idx] -= 1
                diff += 1
            k += 1
            if k > SCALE * 8:
                raise SystemExit("normalize failed to trim")
    if int(raw.sum()) != SCALE or np.any(raw <= 0):
        raise SystemExit("frequency table is not a partition")
    return raw.astype(np.uint16)


def _run_coder(mode: str, blob: bytes) -> bytes:
    done = subprocess.run(
        [str(CODER), mode],
        input=blob,
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        err = done.stderr.decode("utf-8", "replace")
        raise SystemExit(f"gauge coder {mode} exited {done.returncode} {err}")
    return done.stdout


def pack_plane(values: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(values, dtype=np.int32).ravel()
    if flat.size and (int(flat.min()) < -32768 or int(flat.max()) > 32767):
        raise SystemExit("a quotient does not fit in int16")
    symbols, inverse = np.unique(flat, return_inverse=True)
    counts = np.bincount(inverse)
    freq = normalize(counts)
    payload_in = struct.pack("<II", int(inverse.size), int(symbols.size))
    payload_in += freq.astype("<u2").tobytes()
    payload_in += inverse.astype("<u2").tobytes()
    encoded = _run_coder("enc", payload_in)
    if len(encoded) < 4:
        raise SystemExit("coder returned a short payload")
    (payload_len,) = struct.unpack_from("<I", encoded, 0)
    payload = encoded[4:]
    if payload_len != len(payload):
        raise SystemExit("coder payload length does not match")
    blob = struct.pack("<II", int(flat.size), int(symbols.size))
    blob += symbols.astype("<i2").tobytes()
    blob += freq.tobytes()
    blob += struct.pack("<I", payload_len)
    blob += payload
    return blob


def unpack_plane(blob: bytes, offset: int) -> tuple[np.ndarray, int]:
    n, uniq = struct.unpack_from("<II", blob, offset)
    offset += 8
    symbols = np.frombuffer(blob[offset:offset + uniq * 2], dtype="<i2").astype(np.int32)
    offset += uniq * 2
    freq = np.frombuffer(blob[offset:offset + uniq * 2], dtype="<u2")
    offset += uniq * 2
    (payload_len,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    payload = blob[offset:offset + payload_len]
    offset += payload_len
    if len(payload) != payload_len:
        raise SystemExit("truncated band payload")
    query = struct.pack("<III", n, uniq, payload_len)
    query += freq.tobytes()
    query += payload
    decoded = _run_coder("dec", query)
    if len(decoded) != n * 2:
        raise SystemExit("decoded band length does not match")
    inverse = np.frombuffer(decoded, dtype="<u2").astype(np.int32)
    if int(inverse.max(initial=0)) >= uniq:
        raise SystemExit("decoded symbol left the alphabet")
    return symbols[inverse], offset


def pack_song(q: int, original: int, mid_qq, mid_b, side_qq, side_b, rate: int) -> bytes:
    edges = band_edges(mid_qq.shape[1], rate, zoom.HOP)
    if edges[-1] != mid_qq.shape[1] or edges != sorted(edges):
        raise SystemExit("octave edges are not a partition")
    parts = [MAGIC]
    parts.append(struct.pack("<III", q, original, int(mid_qq.shape[0])))
    parts.append(struct.pack("<" + "H" * len(edges), *edges))
    zoom_raw = np.ascontiguousarray(mid_b, dtype=np.uint8).tobytes()
    zoom_raw += np.ascontiguousarray(side_b, dtype=np.uint8).tobytes()
    zoom_blob = zlib.compress(zoom_raw, 9)
    parts.append(struct.pack("<I", len(zoom_blob)))
    parts.append(zoom_blob)
    for plane in (mid_qq, side_qq):
        for lo, hi in zip(edges, edges[1:]):
            parts.append(pack_plane(plane[:, lo:hi]))
    return b"".join(parts)


def unpack_song(blob: bytes, rate: int):
    if blob[:4] != MAGIC:
        raise SystemExit("not a gauge archive")
    q, original, frames = struct.unpack_from("<III", blob, 4)
    offset = 16
    edges = list(struct.unpack_from("<HHHHH", blob, offset))
    offset += 10
    if edges != band_edges(edges[-1], rate, zoom.HOP):
        raise SystemExit("archive edges are not the octave cuts")
    (zoom_len,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    zoom_raw = zlib.decompress(blob[offset:offset + zoom_len])
    offset += zoom_len
    if len(zoom_raw) != frames * 2:
        raise SystemExit("zoom index length does not match")
    mid_b = np.frombuffer(zoom_raw[:frames], dtype=np.uint8).astype(np.int8).copy()
    side_b = np.frombuffer(zoom_raw[frames:], dtype=np.uint8).astype(np.int8).copy()
    planes = []
    for _channel in range(2):
        cols = []
        for _band in range(4):
            values, offset = unpack_plane(blob, offset)
            width = edges[len(cols) + 1] - edges[len(cols)]
            if values.size != frames * width:
                raise SystemExit("band length does not match the octave cut")
            cols.append(values.reshape(frames, width))
        plane = np.column_stack(cols)
        if plane.shape != (frames, edges[-1]):
            raise SystemExit("band widths do not rebuild the frame")
        planes.append(plane)
    if offset != len(blob):
        raise SystemExit("trailing gauge bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    if not CODER.exists():
        raise SystemExit("gauge coder is not built")
    trial = np.array([0, 1, 1, -2, 0, 4, 1, 0, -2, 7], dtype=np.int32)
    blob = pack_plane(trial)
    back, end = unpack_plane(blob, 0)
    if end != len(blob) or not np.array_equal(back, trial):
        raise SystemExit("order-0 plane does not round-trip")
    zeros = np.zeros(64, dtype=np.int32)
    zblob = pack_plane(zeros)
    zback, zend = unpack_plane(zblob, 0)
    if zend != len(zblob) or not np.array_equal(zback, zeros):
        raise SystemExit("a zero band does not round-trip")
    print("gauge self-check ok", flush=True)


def r7_bytes() -> dict[str, int]:
    report = json.loads((paths.SCORES / "mean_report.json").read_text(encoding="utf-8"))
    found = {}
    for song in report["songs"]:
        for row in song["ladder"]:
            if int(row["q"]) == Q:
                found[song["stem"]] = int(row["archive_bytes"])
    return found


def main() -> None:
    self_check()
    out = paths.begin_pass("r8")
    opus, _previous = zoom.opus_and_r5()
    earlier = r7_bytes()
    songs = []
    for path, stem in zip(zoom.SONGS, zoom.STEMS):
        print(f"song {stem}", flush=True)
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        original, mid_qq, mid_b, side_qq, side_b = mean.encode_channels(mid, side, Q)
        blob = pack_song(Q, original, mid_qq, mid_b, side_qq, side_b, rate)
        got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, rate)
        if (
            got_q != Q
            or got_n != original
            or not np.array_equal(got_mid, mid_qq)
            or not np.array_equal(got_side, side_qq)
            or not np.array_equal(got_mb, mid_b)
            or not np.array_equal(got_sb, side_b)
        ):
            raise SystemExit(f"{stem} archive does not rebuild the seventh-pass coefficients")
        (out / f"{stem}_q{Q}.fspz").write_bytes(blob)
        row = {
            "q": Q,
            "archive_bytes": len(blob),
            "r7_bytes": earlier[stem],
            "opus96_bytes": opus[stem],
            "ratio_vs_opus96": len(blob) / opus[stem],
            "ratio_vs_r7": len(blob) / earlier[stem],
            "samples_match_r7": True,
            "wav": str(paths.CURRENT / "r7" / f"{stem}_q{Q}.wav"),
        }
        songs.append({"stem": stem, "source": str(path), "seconds": left.size / rate, "row": row})
        paths.save_score("gauge_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "keeps": "r7 coefficients at Q=2048",
                "coder": "static order-0 rANS per octave band",
                "bands": "fs/16, fs/8, fs/4",
                "zoom": "zlib-9 uint8",
                "quotient": "int16",
            },
            "songs": songs,
        })
        print(
            f"  archive {len(blob)} r7 {earlier[stem]} opus {opus[stem]} "
            f"ratio {row['ratio_vs_opus96']:.3f} vs r7 {row['ratio_vs_r7']:.3f}",
            flush=True,
        )
    print("gauge done", flush=True)


if __name__ == "__main__":
    main()
