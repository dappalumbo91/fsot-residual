#!/usr/bin/env python3
"""Thirty-third audio pass. Declared from the peak count, before this score.

r32 raised Cage's 11-22 kHz band from b=0 to b=1. Full-song SNR moved
from 0.047 dB to 0.528 dB against Opus 1.881. Counted on the r32
archive before this run, a frame keeps its largest coefficient when
the peak is at least half the step:

  mid  all    2452/30233 at b=1,  8353/30233 at the next rung.
             Median peak 18478. Median keeping rung b=3.
  side all   2582/30233 at b=1,  8121/30233 at the next rung.
             Median peak 14898. Median keeping rung b=3.
  mid  quiet 109/5266. side quiet 63/5266.

Another single rung stays inside that dead zone. This pass does not
issue one. On every frame, both channels, 11-22 kHz is set to the
coarsest rung in 0..8 that keeps the peak, and only when that rung is
finer than the stored one. A frame that already keeps its peak stays.
A frame still rounded away at b=8 stays. Every other band stays.
Quotients on moved frames are retaken from the original coefficients.
Q stays 2048. The bucket is not retuned. Cage starts from r32.
The other six songs stay.

The full-song meter is the selector, so the rule is not limited to
quiet frames. A file that crosses Opus 96 stays a size miss. No other
band is coarsened to pay for it. The quiet gaps on the lower octaves
are not this pass.
"""

from __future__ import annotations

import math
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import deliver
import fspack
import paths
import refine_above as above
import refine_context as context
import refine_dial as dial
import refine_fullband as band
import refine_gauge as gauge
import refine_miss as miss
import refine_peak as peak
import refine_quiet as quiet
import refine_zoom as zoom

Q = 2048
MAGIC = b"FS33"
STEM = above.STEM
BAND_INDEX = 3
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
        raise SystemExit("not a cage-peak archive")
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
        raise SystemExit("trailing cage-peak bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    rng = np.random.default_rng(33)
    coeffs = np.zeros((3, 256), dtype=np.float64)
    coeffs[:, 200] = np.array([20000.0, 100.0, 40000.0])
    qq = band.quantize(coeffs, zoom.step_at(Q, 1)).astype(np.int32)
    b_band = np.ones((3, 4), dtype=np.int16)
    keep = np.ones(3, dtype=bool)
    out_q, out_b = peak.clear_peak(coeffs, qq, b_band, BAND_INDEX, keep)
    if not np.array_equal(out_b[:, BAND_INDEX], np.array([3, 1, 2], dtype=np.int16)):
        raise SystemExit("the peak-keeping rung was not chosen")
    if not np.array_equal(out_b[:, :3], b_band[:, :3]):
        raise SystemExit("a lower band moved")
    expect = band.quantize(coeffs[0, 128:256], zoom.step_at(Q, 3))
    if not np.array_equal(out_q[0, 128:256], expect):
        raise SystemExit("the moved frame was not retaken from the original coefficients")
    if np.any(out_q[1] != qq[1]):
        raise SystemExit("a frame that still loses its peak was rewritten")
    blob = pack_song(Q, 3 * zoom.HOP, out_q, out_b, qq, b_band, 44100)
    got_q, got_n, got_mid, _mb, got_side, _sb = unpack_song(blob, 44100)
    if got_q != Q or got_n != 3 * zoom.HOP or not np.array_equal(got_mid, out_q) or not np.array_equal(got_side, qq):
        raise SystemExit("cage-peak archive did not round-trip")
    print("cage-peak self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("r33")
    opus, _previous = zoom.opus_and_r5()
    stem = STEM
    band_index = BAND_INDEX
    print(f"song {stem}", flush=True)
    path = band.SONGS[band.STEMS.index(stem)]
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    archive_path = paths.CURRENT / "r32" / f"{stem}_q{Q}.fspz"
    _q, original, qm, bm, qs, bs = above.unpack_song(archive_path.read_bytes(), rate)
    if _q != Q or original != int(left.size):
        raise SystemExit(f"{stem} source archive does not match the song")
    bm = quiet.as_bands(bm)
    bs = quiet.as_bands(bs)
    cm, n_mid = band.mdct_forward(mid)
    cs, n_side = band.mdct_forward(side)
    if n_mid != original or n_side != original or cm.shape != qm.shape:
        raise SystemExit(f"{stem} cosine frame count drifted")
    source_bytes = archive_path.stat().st_size
    keep = np.ones(qm.shape[0], dtype=bool)
    mid_qq, mid_b = peak.clear_peak(cm, qm, bm, band_index, keep)
    side_qq, side_b = peak.clear_peak(cs, qs, bs, band_index, keep)
    if not np.array_equal(mid_b[:, :3], bm[:, :3]) or not np.array_equal(side_b[:, :3], bs[:, :3]):
        raise SystemExit(f"{stem} a lower band moved")
    if np.any(mid_b[:, band_index] < bm[:, band_index]) or np.any(side_b[:, band_index] < bs[:, band_index]):
        raise SystemExit(f"{stem} the top octave was coarsened")
    lo = EDGES[band_index]
    if np.any(mid_qq[:, :lo] != qm[:, :lo]) or np.any(side_qq[:, :lo] != qs[:, :lo]):
        raise SystemExit(f"{stem} quotients below 11 kHz were rewritten")
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
        "r32_bytes": source_bytes,
        "opus96_bytes": opus_bytes,
        "ratio_vs_opus96": len(blob) / opus_bytes,
        "ratio_vs_r32": len(blob) / source_bytes,
        "under_opus": len(blob) < opus_bytes,
        "mid_zoom_before": float(bm[:, band_index].mean()),
        "mid_zoom_after": float(mid_b[:, band_index].mean()),
        "side_zoom_before": float(bs[:, band_index].mean()),
        "side_zoom_after": float(side_b[:, band_index].mean()),
        "frames_raised_mid": int(np.sum(mid_b[:, band_index] > bm[:, band_index])),
        "frames": int(mid_b.shape[0]),
        "snr_db": snr,
        "band_snr_db": {name: value for name, value in zip(NAMES, octaves)},
        "opus_band_snr_db": {name: value for name, value in zip(NAMES, opus_full)},
        "quiet_snr_db": {name: value for name, value in zip(NAMES, ours_q)},
        "opus_quiet_snr_db": {name: value for name, value in zip(NAMES, opus_q)},
        "wav": str(wav_path),
    }
    paths.save_score("cage_peak_report.json", {
        "pin": fspack.PIN,
        "declared": {
            "rule": "coarsest existing rung that keeps the 11-22 kHz peak, every frame, both channels",
            "band": NAMES[band_index],
            "kept": "every band below 11 kHz stays",
            "songs": [stem],
            "coder": "tenth-pass magnitude bucket, not retuned",
            "q": Q,
            "source": "r32",
        },
        "songs": [{"stem": stem, "source": str(path), "seconds": left.size / rate, "row": row}],
    })
    print(
        f"  archive {len(blob)} r32 {source_bytes} opus {opus_bytes} "
        f"ratio {row['ratio_vs_opus96']:.3f} "
        f"band {NAMES[band_index]} {octaves[band_index]:.3f} vs opus {opus_full[band_index]:.3f} "
        f"zoom {row['mid_zoom_before']:.2f}->{row['mid_zoom_after']:.2f} "
        f"raised {row['frames_raised_mid']}/{row['frames']} SNR {snr:.2f}",
        flush=True,
    )
    print("cage peak done", flush=True)


if __name__ == "__main__":
    main()
