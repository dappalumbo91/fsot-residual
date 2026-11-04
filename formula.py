#!/usr/bin/env python3
"""Formula pass. Declared before this pass is scored.

The MDCT on the scored grid is critically sampled: one hop brings in
256 new samples and the frame stores 256 coefficients. Fold1 already
sits within a few tens of kilobytes of the bucket entropy of those
coefficients. Skrillex and VNV are over Opus at that entropy. Another
remap of the same integers cannot clear them. Fold2 and fold3 confirmed
the remaps grow.

This pass changes the object. A frame's formula keeps every pitch and
every valley, meaning each strict local extremum of the quotient, and
both edges of the octave so the formula has boundary values. Between
two stored knots the decoder runs the unique straight chord,

    y = a * (span - step) + b * step

and rounds half away from zero. No seed is used as a blend weight.
The inverse MDCT then renders the wave. Bins the chord fixes are not
written. Keeping them would return to the same integers.

The tenth-pass bucket, the block parent, and the half-cycle residual
are not used. Zoom bytes are copied. Q stays 2048. Bruno stays at
fold1. No rung is spent. Beta is not raised. Samples are not passed
to compute_scalar. No wav is written. The score reports the render
SNR against the wave of the stored quotients, because this formula
does not claim those quotients.
"""

from __future__ import annotations

import hashlib
import math
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
import refine_cagequiet as cagequiet
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
MAGIC = b"FS2E"
EDGES = (0, 32, 64, 128, 256)
RULE = (
    "per octave, store both edges and every strict local extremum; "
    "the decoder draws the straight chord and rounds half away from zero; "
    "inverse MDCT renders the wave; zoom blob copied; "
    "Bruno stays at fold1; no rung; no wav"
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


def pack_plane(values: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(values, dtype=np.int32).ravel()
    if flat.size == 0:
        return struct.pack("<I", 0)
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
    return struct.pack("<I", len(blob)) + blob


def unpack_plane(blob: bytes, offset: int) -> tuple[np.ndarray, int]:
    (size,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    if size == 0:
        return np.zeros(0, dtype=np.int32), offset
    start = offset
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
    if offset - start != size or len(payload) != payload_len:
        raise SystemExit("truncated formula payload")
    query = struct.pack("<III", n, uniq, payload_len) + payload
    decoded = run_coder("dec", query)
    if len(decoded) != n * 2:
        raise SystemExit("decoded formula length does not match")
    inverse = np.frombuffer(decoded, dtype="<u2").astype(np.int32)
    if inverse.size and int(inverse.max()) >= uniq:
        raise SystemExit("decoded symbol left the alphabet")
    return symbols[inverse], offset


def extrema_mask(block: np.ndarray) -> np.ndarray:
    """True at a strict local pitch or valley. Edges are not included."""
    frames, width = block.shape
    keep = np.zeros((frames, width), dtype=bool)
    if width < 3:
        return keep
    mid = block[:, 1:-1]
    left = block[:, :-2]
    right = block[:, 2:]
    is_max = (mid >= left) & (mid >= right) & ((mid > left) | (mid > right))
    is_min = (mid <= left) & (mid <= right) & ((mid < left) | (mid < right))
    keep[:, 1:-1] = is_max | is_min
    return keep


def knots_of(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames, width = block.shape
    left = np.ascontiguousarray(block[:, 0], dtype=np.int32)
    right = np.ascontiguousarray(block[:, -1], dtype=np.int32)
    mask = extrema_mask(block)
    counts = mask.sum(axis=1).astype(np.int32)
    if int(counts.sum()) == 0:
        return left, right, counts, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    positions = np.empty(int(counts.sum()), dtype=np.int32)
    values = np.empty(int(counts.sum()), dtype=np.int32)
    cursor = 0
    columns = np.arange(1, width - 1, dtype=np.int32) if width > 2 else np.zeros(0, dtype=np.int32)
    for t in range(frames):
        n = int(counts[t])
        if n == 0:
            continue
        idx = columns[mask[t, 1:-1]]
        positions[cursor:cursor + n] = idx
        values[cursor:cursor + n] = block[t, idx]
        cursor += n
    return left, right, counts, positions, values


def chord(span: int, start: int, stop: int) -> np.ndarray:
    steps = np.arange(1, span, dtype=np.int64)
    num = np.int64(start) * (span - steps) + np.int64(stop) * steps
    half = span // 2
    sign = np.sign(num).astype(np.int64)
    return sign * ((np.abs(num) + half) // span)


def fill_band(width: int, left: np.ndarray, right: np.ndarray, counts: np.ndarray,
              positions: np.ndarray, values: np.ndarray) -> np.ndarray:
    frames = int(left.size)
    out = np.empty((frames, width), dtype=np.int32)
    cursor = 0
    for t in range(frames):
        n = int(counts[t])
        pos = positions[cursor:cursor + n]
        val = values[cursor:cursor + n]
        cursor += n
        row = np.empty(width, dtype=np.int64)
        row[0] = int(left[t])
        row[width - 1] = int(right[t])
        if n:
            row[pos] = val
            knots = np.concatenate((np.zeros(1, dtype=np.int32), pos, np.full(1, width - 1, dtype=np.int32)))
        else:
            knots = np.array([0, width - 1], dtype=np.int32)
        knot_values = row[knots]
        for a, b, va, vb in zip(knots[:-1], knots[1:], knot_values[:-1], knot_values[1:]):
            span = int(b - a)
            if span > 1:
                row[a + 1:b] = chord(span, int(va), int(vb))
        out[t] = row
    if cursor != int(positions.size):
        raise SystemExit("knot cursor did not consume the formula")
    return out


def pack_band(block: np.ndarray) -> bytes:
    left, right, counts, positions, values = knots_of(block)
    return b"".join((
        pack_plane(left),
        pack_plane(right),
        pack_plane(counts),
        pack_plane(positions),
        pack_plane(values),
    ))


def unpack_band(blob: bytes, offset: int, frames: int, width: int) -> tuple[np.ndarray, int, int]:
    left, offset = unpack_plane(blob, offset)
    right, offset = unpack_plane(blob, offset)
    counts, offset = unpack_plane(blob, offset)
    positions, offset = unpack_plane(blob, offset)
    values, offset = unpack_plane(blob, offset)
    if left.size != frames or right.size != frames or counts.size != frames:
        raise SystemExit("formula edge length does not match the frame count")
    if positions.size != values.size or int(counts.sum()) != int(positions.size):
        raise SystemExit("formula knots do not match their counts")
    filled = fill_band(width, left, right, counts, positions, values)
    knots = int(frames * 2 + positions.size)
    return filled, offset, knots


def pack_song(source: bytes, mid_qq: np.ndarray, side_qq: np.ndarray) -> bytes:
    (zoom_len,) = struct.unpack_from("<I", source, 26)
    parts = [MAGIC, source[4:30 + zoom_len]]
    for plane in (mid_qq, side_qq):
        for lo, hi in zip(EDGES, EDGES[1:]):
            parts.append(pack_band(plane[:, lo:hi]))
    return b"".join(parts)


def unpack_song(blob: bytes, rate: int):
    if blob[:4] != MAGIC:
        raise SystemExit("not a formula archive")
    q, original, frames = struct.unpack_from("<III", blob, 4)
    offset = 16
    edges = list(struct.unpack_from("<HHHHH", blob, offset))
    offset += 10
    if tuple(edges) != EDGES or edges != gauge.band_edges(edges[-1], rate, zoom.HOP):
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
    knot_count = 0
    bin_count = 0
    for _channel in range(2):
        cols = []
        for band_i in range(4):
            width = edges[band_i + 1] - edges[band_i]
            values, offset, knots = unpack_band(blob, offset, frames, width)
            cols.append(values)
            knot_count += knots
            bin_count += frames * width
        planes.append(np.column_stack(cols))
    if offset != len(blob):
        raise SystemExit("trailing formula bytes")
    return q, original, planes[0], mid_b, planes[1], side_b, knot_count, bin_count


def hats(qq: np.ndarray, b_band: np.ndarray) -> np.ndarray:
    hat = np.empty(qq.shape, dtype=np.float64)
    grain = np.int64(Q) * np.int64(band.GRAIN)
    for index, (lo, hi) in enumerate(zip(EDGES, EDGES[1:])):
        step = grain >> b_band[:, index].astype(np.int64)
        step = np.maximum(step, 1)
        hat[:, lo:hi] = qq[:, lo:hi] * step[:, None]
    return hat


def render_wave(qq: np.ndarray, b_band: np.ndarray, original: int) -> np.ndarray:
    return tonal.mdct_inverse(hats(qq, b_band), original)


def wave_snr(reference: np.ndarray, trial: np.ndarray) -> float:
    signal = reference.astype(np.float64)
    error = trial.astype(np.float64) - signal
    power = float(np.dot(signal, signal))
    noise = float(np.dot(error, error))
    if noise == 0.0:
        return math.inf
    if power == 0.0:
        return 0.0
    return 10.0 * math.log10(power / noise)


def pcm_snr(mid_q, mid_b, side_q, side_b, mid_f, side_f, original: int) -> float:
    mid_ref = render_wave(mid_q, mid_b, original)
    side_ref = render_wave(side_q, side_b, original)
    mid_new = render_wave(mid_f, mid_b, original)
    side_new = render_wave(side_f, side_b, original)
    left_ref, right_ref = deliver.unmid_side(mid_ref, side_ref)
    left_new, right_new = deliver.unmid_side(mid_new, side_new)
    reference = np.concatenate((left_ref, right_ref))
    trial = np.concatenate((left_new, right_new))
    return wave_snr(reference, trial)


def self_check() -> None:
    if not CODER.is_file():
        raise SystemExit("fold coder is not built")
    done = subprocess.run([str(CODER), "self"], capture_output=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"fold self exited {done.returncode}")
    raw = fspack.ENGINE_PATH.read_bytes()
    if not hashlib.sha256(raw).hexdigest().startswith(fspack.PIN_PREFIX):
        raise SystemExit("engine pin mismatch")
    if tonal.HOP != 256 or tonal.WINDOW != 512:
        raise SystemExit("formula renderer is for the scored 512/256 grid")
    ramp = np.arange(8, dtype=np.int32).reshape(1, 8)
    filled = fill_band(8, *knots_of(ramp))
    if not np.array_equal(filled, ramp):
        raise SystemExit("a straight chord did not reproduce the ramp")
    tent = np.array([[0, 1, 2, 3, 2, 1, 0]], dtype=np.int32)
    if not np.array_equal(fill_band(7, *knots_of(tent)), tent):
        raise SystemExit("a tent of pitches and valleys did not reproduce")
    broken = np.array([[0, 4, 5, 0, 0]], dtype=np.int32)
    if np.array_equal(fill_band(5, *knots_of(broken)), broken):
        raise SystemExit("the chord reproduced a row it does not determine")
    block = np.array(
        [
            [0, 1, 2, 3, 2, 1, 0, 0],
            [4, 1, 0, -2, 0, 3, 1, 4],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    blob = pack_band(block)
    back, offset, _knots = unpack_band(blob, 0, 3, 8)
    if offset != len(blob) or not np.array_equal(back, fill_band(8, *knots_of(block))):
        raise SystemExit("formula band did not round-trip")
    tone = np.zeros((2, 256), dtype=np.float64)
    tone[:, 3] = 1000.0
    rendered = tonal.mdct_inverse(tone, tonal.HOP)
    direct = tonal.mdct_inverse(tone.copy(), tonal.HOP)
    if not np.array_equal(rendered, direct):
        raise SystemExit("formula renderer drifted from the scored inverse")
    print("formula self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("formula")
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
        got_q, got_n, got_mid, got_mb, got_side, got_sb, knots, bins = unpack_song(blob, RATE)
        if got_q != Q or got_n != original:
            raise SystemExit(f"{stem} header did not round-trip")
        if not np.array_equal(got_mb, mid_b) or not np.array_equal(got_sb, side_b):
            raise SystemExit(f"{stem} zoom did not round-trip")
        if blob[30:30 + zoom_len] != zoom_blob:
            raise SystemExit(f"{stem} zoom bytes were rewritten")
        fresh_mid = fill_from_stored(mid_qq)
        fresh_side = fill_from_stored(side_qq)
        if not np.array_equal(got_mid, fresh_mid) or not np.array_equal(got_side, fresh_side):
            raise SystemExit(f"{stem} formula did not round-trip")
        snr = pcm_snr(mid_qq, mid_b, side_qq, side_b, got_mid, got_side, original)
        archive_path = out / f"{stem}_q{Q}.fs2e"
        archive_path.write_bytes(blob)
        fold1_bytes = (paths.CURRENT / "fold1" / f"{stem}_q{Q}.fs2a").stat().st_size
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
            "snr_vs_stored_wave_db": snr,
            "knot_fraction": knots / bins,
            "knots": knots,
            "bins": bins,
            "zoom_bytes_match": True,
            "archive": str(archive_path),
        }
        songs.append({"stem": stem, "seconds": seconds, "row": row})
        paths.save_score("formula_report.json", {
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
            f"{row['kbps']:.1f}kbps snr {snr:.2f} knots {row['knot_fraction']:.4f}",
            flush=True,
        )
    print("formula done", flush=True)


def fill_from_stored(plane: np.ndarray) -> np.ndarray:
    cols = []
    for lo, hi in zip(EDGES, EDGES[1:]):
        block = plane[:, lo:hi]
        cols.append(fill_band(hi - lo, *knots_of(block)))
    return np.column_stack(cols)


if __name__ == "__main__":
    main()
