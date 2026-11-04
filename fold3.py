#!/usr/bin/env python3
"""Fold 3. Declared before this pass is scored.

Fold2's residual was the wrong fold. derived_D_eff(Acoustics) is a
compactification depth, and using it as a frequency-block parent made
the symbols larger. This pass does not try a second block length.

The basis in refine_tonal is cos(pi/N * (n+0.5+N/2) * (k+0.5)) with
N = hop = 256. One hop advances bin k by (k+0.5)/2 cycles. Two hops
advance it by k+0.5 cycles, a half cycle at every integer bin. A tone
on that center changes sign: q[t] = -q[t-2]. The residual is the
failure of that identity,

    r[0] = q[0]
    r[1] = q[1]
    r[t] = q[t] + q[t-2]    for t >= 2

and the inverse is q[t] = r[t] - q[t-2]. The characteristic factor is
r^2 + 1 = 0, fixed by the half cycle. It is not a fitted gain, not an
LPC order, and not a searched lag. A one-frame difference was already
measured and is not this contract. A four-frame step is not tried.

The tenth-pass bucket is unchanged. For t >= 2 it reads q[t-2], the
frame the half-turn reads. Zoom bytes are copied. The fold1 range coder
is reused: int32 support, counts start at 1, the 12-bit table is not
stored. SCALE stays 4096. Q stays 2048. Bruno stays at fold1.
No rung. No coarsen. No wav. Beta is not raised. Samples are not passed
to compute_scalar.
"""

from __future__ import annotations

import hashlib
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
import refine_tonal as tonal
import refine_vowel as vowel
import refine_zoom as zoom

ROOT = Path(__file__).resolve().parent
CODER = ROOT / "fold_range.exe"
Q = 2048
RATE = 44100
MAGIC = b"FS2C"
RULE = (
    "half-cycle residual q[t] + q[t-2]; two hops of the MDCT basis; "
    "frames 0 and 1 stored raw; tenth-pass bucket of q[t-2]; "
    "zoom blob copied; support stored; counts start at 1; "
    "12-bit table not stored; Bruno stays at fold1; "
    "no second block length; no rung; no coarsen; no wav"
)

SONGS = (
    ("skrillex_first_of_the_year", "r16", dial.unpack_song),
    ("vnv_nation_beloved", "r17", vowel.unpack_song),
    ("cage_the_elephant_aint_no_rest", "r34", cagequiet.unpack_song),
)


def run_coder(mode: str, blob: bytes) -> bytes:
    done = subprocess.run([str(CODER), mode], input=blob, capture_output=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"fold coder {mode} exited {done.returncode}")
    return done.stdout


def _basis_at(n: int, k: int, hop: int) -> float:
    """Same cosine as refine_tonal._basis, evaluated past the window."""
    return float(np.cos(np.pi / hop * (n + 0.5 + hop / 2.0) * (k + 0.5)))


def assert_half_cycle() -> None:
    """Two hops flip the sign of every bin center on the scored grid."""
    hop = int(tonal.HOP)
    if hop != 256 or int(tonal.WINDOW) != 512:
        raise SystemExit("half-turn law is for the scored 512/256 grid")
    if zoom.HOP != hop:
        raise SystemExit("zoom hop and tonal hop disagree")
    if tonal.BASIS.shape != (tonal.WINDOW, hop):
        raise SystemExit("tonal basis is not the scored window")
    for k in (0, 1, 2, 31, 63, 127, 255):
        if abs(_basis_at(0, k, hop) - float(tonal.BASIS[0, k])) > 1e-9:
            raise SystemExit("half-turn cosine is not the tonal basis")
        if abs(_basis_at(0, k, hop) + _basis_at(2 * hop, k, hop)) > 1e-8:
            raise SystemExit(f"bin {k} does not change sign after two hops")


def half_turn(plane: np.ndarray) -> np.ndarray:
    out = np.array(plane, dtype=np.int64, copy=True)
    if out.shape[0] > 2:
        out[2:] = out[2:] + out[:-2]
    if out.size and (int(out.min()) < -2147483648 or int(out.max()) > 2147483647):
        raise SystemExit("a half-turn symbol does not fit in int32")
    return out.astype(np.int32)


def unhalf(turned: np.ndarray) -> np.ndarray:
    out = np.array(turned, dtype=np.int64, copy=True)
    for t in range(2, out.shape[0]):
        out[t] = out[t] - out[t - 2]
    if out.size and (int(out.min()) < -2147483648 or int(out.max()) > 2147483647):
        raise SystemExit("a restored quotient does not fit in int32")
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


def pack_band(plane: np.ndarray, turned: np.ndarray) -> bytes:
    frames = plane.shape[0]
    parts = [pack_plane(turned[0])]
    if frames == 1:
        parts.append(struct.pack("<H", 0))
        return b"".join(parts)
    parts.append(pack_plane(turned[1]))
    if frames == 2:
        parts.append(struct.pack("<H", 0))
        return b"".join(parts)
    buckets = context.context_bucket(plane[:-2]).ravel()
    current = np.ascontiguousarray(turned[2:]).ravel()
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
    plane = np.zeros((frames, width), dtype=np.int64)
    plane[0] = first
    if frames == 1:
        (n_ctx,) = struct.unpack_from("<H", blob, offset)
        offset += 2
        if n_ctx != 0:
            raise SystemExit("a one-frame band carried a context")
        return plane.astype(np.int32), offset
    second, offset = unpack_plane(blob, offset)
    if second.size != width:
        raise SystemExit("second frame width does not match the band")
    plane[1] = second
    (n_ctx,) = struct.unpack_from("<H", blob, offset)
    offset += 2
    streams = {}
    cursors = {}
    for _i in range(n_ctx):
        (bucket_id,) = struct.unpack_from("<h", blob, offset)
        offset += 2
        values, offset = unpack_plane(blob, offset)
        streams[int(bucket_id)] = values.astype(np.int64)
        cursors[int(bucket_id)] = 0
    for t in range(frames - 2):
        buckets = context.context_bucket(plane[t].astype(np.int32))
        folded = np.empty(width, dtype=np.int64)
        for bucket_id in np.unique(buckets):
            key = int(bucket_id)
            if key not in streams:
                raise SystemExit("a half-turn bucket was not stored")
            idx = np.flatnonzero(buckets == bucket_id)
            cursor = cursors[key]
            count = int(idx.size)
            folded[idx] = streams[key][cursor:cursor + count]
            cursors[key] = cursor + count
        plane[t + 2] = folded - plane[t]
    for bucket_id, values in streams.items():
        if cursors[bucket_id] != values.size:
            raise SystemExit("a context stream was not consumed")
    return plane.astype(np.int32), offset


def pack_song(source: bytes, mid_qq: np.ndarray, side_qq: np.ndarray) -> bytes:
    (zoom_len,) = struct.unpack_from("<I", source, 26)
    edges = struct.unpack_from("<HHHHH", source, 16)
    parts = [MAGIC, source[4:30 + zoom_len]]
    for plane in (mid_qq, side_qq):
        for lo, hi in zip(edges, edges[1:]):
            block = plane[:, lo:hi]
            parts.append(pack_band(block, half_turn(block)))
    return b"".join(parts)


def unpack_song(blob: bytes, rate: int):
    if blob[:4] != MAGIC:
        raise SystemExit("not a fold3 archive")
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
    done = subprocess.run([str(CODER), "self"], capture_output=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"fold self exited {done.returncode}")
    raw = fspack.ENGINE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not digest.startswith(fspack.PIN_PREFIX):
        raise SystemExit("engine pin mismatch")
    assert_half_cycle()
    tone = np.array([[4, 1], [9, 2], [-4, 4], [-9, 8], [4, 3]], dtype=np.int32)
    turned = half_turn(tone)
    if int(turned[2, 0]) != 0 or int(turned[3, 0]) != 0 or int(turned[4, 0]) != 0:
        raise SystemExit("a half-cycle tone did not cancel")
    if int(turned[2, 1]) != 5 or int(turned[4, 1]) != 7:
        raise SystemExit("half-turn residual is not q[t] + q[t-2]")
    if not np.array_equal(unhalf(turned), tone):
        raise SystemExit("half-turn did not invert")
    blob = pack_band(tone, turned)
    back, offset = unpack_band(blob, 0, tone.shape[0], tone.shape[1])
    if offset != len(blob) or not np.array_equal(back, tone):
        raise SystemExit("fold3 band did not round-trip")
    print("fold3 self-check ok half-cycle", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("fold3")
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
        archive_path = out / f"{stem}_q{Q}.fs2c"
        archive_path.write_bytes(blob)
        fold1_path = paths.CURRENT / "fold1" / f"{stem}_q{Q}.fs2a"
        fold1_bytes = fold1_path.stat().st_size
        seconds = original / RATE
        opus_n = opus[stem]
        row = {
            "q": Q,
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
        paths.save_score("fold3_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "rule": RULE,
                "songs": [item[0] for item in SONGS],
                "left_out": "Bruno stays at fold1",
                "q": Q,
                "magic": MAGIC.decode("ascii"),
            },
            "songs": songs,
        })
        print(
            f"  archive {len(blob)} fold1 {fold1_bytes} opus {opus_n} "
            f"ratio {row['ratio_vs_opus96']:.6f} margin {row['margin_bytes']} "
            f"{row['kbps']:.1f}kbps",
            flush=True,
        )
    print("fold3 done", flush=True)


if __name__ == "__main__":
    main()
