#!/usr/bin/env python3
"""Thirty-first audio pass. Declared after the Cage trace and before this score.

trace_cage.py locked its rule before it ran. The full-song gaps it
printed, r16 minus Opus, are:

  0-2.8 kHz    +10.578 dB
  2.8-5.5 kHz   +0.881 dB
  5.5-11 kHz    -3.134 dB
  11-22 kHz     -1.844 dB

The candidate is the most negative full-song gap. That is 5.5-11 kHz,
band index 2. Quiet-slice gaps are not the selector. This pass raises
only that band, on both channels, by one existing ladder rung. The
quotients are retaken from the original coefficients. Every other band
stays. Q stays 2048. The tenth-pass bucket is not retuned. Cage starts
from the r16 archive. The other six songs stay.

A file that crosses its Opus 96 archive stays a size miss. The band
stays a quality miss if it is still behind Opus after this one rung.
No second band is raised from this score.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import deliver
import fspack
import paths
import refine_context as context
import refine_dial as dial
import refine_fullband as band
import refine_gauge as gauge
import refine_miss as miss
import refine_quiet as quiet
import refine_zoom as zoom
import trace_cage

Q = 2048
MAGIC = b"FSPK"
STEM = trace_cage.STEM
BAND_INDEX = 2
EDGES = dial.EDGES
NAMES = dial.NAMES


def pack_song(q: int, original: int, mid_qq, mid_b, side_qq, side_b, rate: int) -> bytes:
    edges = gauge.band_edges(mid_qq.shape[1], rate, zoom.HOP)
    parts = [MAGIC, struct.pack("<III", q, original, int(mid_qq.shape[0]))]
    parts.append(struct.pack("<" + "H" * len(edges), *edges))
    zoom_raw = np.ascontiguousarray(mid_b, dtype=np.uint8).tobytes()
    zoom_raw += np.ascontiguousarray(side_b, dtype=np.uint8).tobytes()
    zoom_blob = zlib.compress(zoom_raw, 9)
    parts.append(struct.pack("<I", len(zoom_blob)))
    parts.append(zoom_blob)
    for plane in (mid_qq, side_qq):
        for lo, hi in zip(edges, edges[1:]):
            parts.append(context.pack_band(plane[:, lo:hi]))
    return b"".join(parts)


def unpack_song(blob: bytes, rate: int):
    if blob[:4] != MAGIC:
        raise SystemExit("not a cage-rung archive")
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
            values, offset = context.unpack_band(blob, offset, frames, width)
            cols.append(values)
        planes.append(np.column_stack(cols))
    if offset != len(blob):
        raise SystemExit("trailing cage-rung bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    trace = json.loads((paths.SCORES / "cage_trace.json").read_text(encoding="utf-8"))
    if trace.get("rule") != trace_cage.RULE or trace.get("candidate_band") != BAND_INDEX:
        raise SystemExit("the locked band is not the trace candidate")
    if trace.get("candidate_name") != NAMES[BAND_INDEX]:
        raise SystemExit("the trace name does not match band 2")
    gauge.self_check()
    rng = np.random.default_rng(31)
    coeffs = rng.normal(0, 80000, (4, 256))
    qq = band.quantize(coeffs, zoom.step_at(Q, 0)).astype(np.int32)
    b_band = np.zeros((4, 4), dtype=np.int16)
    b_band[:, BAND_INDEX] = np.array([0, 1, 8, 4], dtype=np.int16)
    out_q, out_b = dial.raise_deficit(coeffs, qq, b_band, BAND_INDEX)
    if not np.array_equal(out_b[:, BAND_INDEX], np.array([1, 2, 8, 5], dtype=np.int16)):
        raise SystemExit("the cage rung did not move by one")
    if not np.array_equal(out_b[:, [0, 1, 3]], b_band[:, [0, 1, 3]]):
        raise SystemExit("a band outside 5.5-11 kHz moved")
    expect = band.quantize(coeffs[0, 64:128], zoom.step_at(Q, 1))
    if not np.array_equal(out_q[0, 64:128], expect):
        raise SystemExit("the rung was not retaken from the original coefficients")
    blob = pack_song(Q, 4 * zoom.HOP, out_q, out_b, qq, b_band, 44100)
    got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, 44100)
    if got_q != Q or got_n != 4 * zoom.HOP or not np.array_equal(got_mid, out_q):
        raise SystemExit("cage-rung archive did not round-trip")
    if not np.array_equal(got_side, qq) or not np.array_equal(got_sb, b_band):
        raise SystemExit("the untouched channel did not round-trip")
    print("cage self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("r31")
    opus, _previous = zoom.opus_and_r5()
    stem = STEM
    band_index = BAND_INDEX
    print(f"song {stem}", flush=True)
    path = band.SONGS[band.STEMS.index(stem)]
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    archive_path = paths.CURRENT / "r16" / f"{stem}_q{Q}.fspz"
    _q, original, qm, bm, qs, bs = dial.unpack_song(archive_path.read_bytes(), rate)
    if _q != Q or original != int(left.size):
        raise SystemExit(f"{stem} source archive does not match the song")
    bm = quiet.as_bands(bm)
    bs = quiet.as_bands(bs)
    cm, n_mid = band.mdct_forward(mid)
    cs, n_side = band.mdct_forward(side)
    if n_mid != original or n_side != original or cm.shape != qm.shape:
        raise SystemExit(f"{stem} cosine frame count drifted")
    source_bytes = archive_path.stat().st_size
    mid_qq, mid_b = dial.raise_deficit(cm, qm, bm, band_index)
    side_qq, side_b = dial.raise_deficit(cs, qs, bs, band_index)
    untouched = [i for i in range(4) if i != band_index]
    if not np.array_equal(mid_b[:, untouched], bm[:, untouched]):
        raise SystemExit(f"{stem} a band outside the candidate moved")
    lo, hi = EDGES[band_index], EDGES[band_index + 1]
    if np.any(mid_qq[:, :lo] != qm[:, :lo]) or np.any(mid_qq[:, hi:] != qm[:, hi:]):
        raise SystemExit(f"{stem} quotients outside the candidate were rewritten")
    pairs = miss.pairs_for(rate, mid_qq.shape[1])
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
        raise SystemExit(f"{stem} archive does not rebuild")
    back_m = miss.restore(mid_qq, mid_b, Q, original, pairs)
    back_s = miss.restore(side_qq, side_b, Q, original, pairs)
    play_l, play_r = deliver.unmid_side(back_m, back_s)
    play_l = deliver.clip_i16(play_l).astype(np.int32)
    play_r = deliver.clip_i16(play_r).astype(np.int32)
    err = np.concatenate([
        play_l.astype(np.float64) - left,
        play_r.astype(np.float64) - right,
    ])
    noise = float(np.mean(err ** 2))
    signal = float(np.mean(np.concatenate([left, right]).astype(np.float64) ** 2))
    snr = 10 * math.log10(signal / noise)
    octaves = dial.band_snr(left, right, play_l, play_r, rate)
    ours_q = quiet.quiet_snr(left, right, play_l, play_r, rate)
    opus_l, opus_r, rate_o = deliver.decode_pcm(paths.REFERENCE / "opus" / f"{stem}_opus96.opus")
    n = min(left.size, opus_l.size)
    opus_full = dial.band_snr(left[:n], right[:n], opus_l[:n], opus_r[:n], rate_o)
    opus_q = quiet.quiet_snr(left[:n], right[:n], opus_l[:n], opus_r[:n], rate_o)
    wav_path = out / f"{stem}_q{Q}.wav"
    deliver.write_wav(wav_path, play_l, play_r, rate)
    (out / f"{stem}_q{Q}.fspz").write_bytes(blob)
    opus_bytes = opus[stem]
    row = {
        "q": Q,
        "dial_band": band_index,
        "dial_name": NAMES[band_index],
        "archive_bytes": len(blob),
        "r16_bytes": source_bytes,
        "opus96_bytes": opus_bytes,
        "ratio_vs_opus96": len(blob) / opus_bytes,
        "ratio_vs_r16": len(blob) / source_bytes,
        "under_opus": len(blob) < opus_bytes,
        "mid_zoom_before": float(bm[:, band_index].mean()),
        "mid_zoom_after": float(mid_b[:, band_index].mean()),
        "side_zoom_before": float(bs[:, band_index].mean()),
        "side_zoom_after": float(side_b[:, band_index].mean()),
        "snr_db": snr,
        "band_snr_db": {name: value for name, value in zip(NAMES, octaves)},
        "opus_band_snr_db": {name: value for name, value in zip(NAMES, opus_full)},
        "quiet_snr_db": {name: value for name, value in zip(NAMES, ours_q)},
        "opus_quiet_snr_db": {name: value for name, value in zip(NAMES, opus_q)},
        "wav": str(wav_path),
    }
    paths.save_score("cage_rung_report.json", {
        "pin": fspack.PIN,
        "declared": {
            "rule": trace_cage.RULE,
            "band": NAMES[band_index],
            "kept": "every other band stays",
            "songs": [stem],
            "coder": "tenth-pass magnitude bucket, not retuned",
            "q": Q,
            "source": "r16",
        },
        "songs": [{"stem": stem, "source": str(path), "seconds": left.size / rate, "row": row}],
    })
    print(
        f"  archive {len(blob)} r16 {source_bytes} opus {opus_bytes} "
        f"ratio {row['ratio_vs_opus96']:.3f} "
        f"band {NAMES[band_index]} {octaves[band_index]:.3f} vs opus {opus_full[band_index]:.3f} "
        f"SNR {snr:.2f}",
        flush=True,
    )
    print("cage rung done", flush=True)


if __name__ == "__main__":
    main()
