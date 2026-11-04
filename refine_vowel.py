#!/usr/bin/env python3
"""Seventeenth audio pass. Declared before this pass is scored.

The wrong term is the gauge on the mid of 2.8-5.5 kHz, the sung vowel.
Traced on the sixteenth-pass playback before this run, vocal window
70 s to 170 s, and not retuned after it:

  The mid level in that band is -0.09 dB, so the vowel is present.
  Its SNR is 7.8 dB against Opus 13.0 dB.
  The waveform lines up at 0 ms (correlation 0.99). There is no
  separate delay to remove.
  The top octave was the sixteenth-pass rung and the vocal remained.

This pass raises only VNV's mid 2.8-5.5 kHz band by one existing
ladder rung. Quotients are retaken from the original coefficients.
The side stays. Every other band stays. Q stays 2048. The bucket is
not retuned. Skrillex, Cage, Sublime, Clay, Eminem, and Bruno are
not rebuilt.

The same trace put the ideal cost of this rung at 260,249 bytes.
The r16 margin under Opus 96 was 94,789 bytes. Crossing Opus is a
size miss and is not repaired by moving a different band.
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
import refine_context as context
import refine_dial as dial
import refine_fullband as band
import refine_gauge as gauge
import refine_miss as miss
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPV"
VOWEL = 1
VOCAL_LO = 70
VOCAL_HI = 170
STEM = "vnv_nation_beloved"
NAMES = ("0-2.8", "2.8-5.5", "5.5-11", "11-22")
FREQS = ((0, 2756), (2756, 5512), (5512, 11025), (11025, 22050))


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
        raise SystemExit("not a vowel archive")
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
        raise SystemExit("trailing vowel bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def octave_snr(sig: np.ndarray, err: np.ndarray, rate: int) -> list[float]:
    spec_s = np.abs(np.fft.rfft(sig)) ** 2
    spec_e = np.abs(np.fft.rfft(err)) ** 2
    freqs = np.fft.rfftfreq(sig.size, 1.0 / rate)
    out = []
    for lo, hi in FREQS:
        sl = (freqs >= lo) & (freqs < hi)
        out.append(10 * math.log10(float(spec_s[sl].sum()) / max(float(spec_e[sl].sum()), 1e-12)))
    return out


def self_check() -> None:
    gauge.self_check()
    rng = np.random.default_rng(17)
    coeffs = rng.normal(0, 80000, (4, 256))
    qq = band.quantize(coeffs, zoom.step_at(Q, 2)).astype(np.int32)
    b_band = np.zeros((4, 4), dtype=np.int16)
    b_band[:, VOWEL] = np.array([0, 2, 8, 5], dtype=np.int16)
    side_q = qq + 3
    side_b = b_band + 1
    out_q, out_b = dial.raise_deficit(coeffs, qq, b_band, VOWEL)
    if not np.array_equal(out_b[:, VOWEL], np.array([1, 3, 8, 6], dtype=np.int16)):
        raise SystemExit("the vowel rung did not move by one")
    if not np.array_equal(out_b[:, [0, 2, 3]], b_band[:, [0, 2, 3]]):
        raise SystemExit("a mid band outside the vowel moved")
    if np.any(out_q[:, :32] != qq[:, :32]) or np.any(out_q[:, 64:] != qq[:, 64:]):
        raise SystemExit("quotients outside the vowel were rewritten")
    expect = band.quantize(coeffs[1, 32:64], zoom.step_at(Q, 3))
    if not np.array_equal(out_q[1, 32:64], expect):
        raise SystemExit("the vowel was not retaken from the original coefficients")
    blob = pack_song(Q, 4 * zoom.HOP, out_q, out_b, side_q, side_b, 44100)
    got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, 44100)
    if got_q != Q or got_n != 4 * zoom.HOP:
        raise SystemExit("header did not round-trip")
    if not np.array_equal(got_mid, out_q) or not np.array_equal(got_side, side_q):
        raise SystemExit("vowel coefficients did not round-trip")
    if not np.array_equal(got_mb, out_b) or not np.array_equal(got_sb, side_b):
        raise SystemExit("vowel zoom did not round-trip")
    print("vowel self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("r17")
    opus, _previous = zoom.opus_and_r5()
    print(f"song {STEM}", flush=True)
    path = band.SONGS[band.STEMS.index(STEM)]
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    archive_path = paths.CURRENT / "r16" / f"{STEM}_q{Q}.fspz"
    _q, original, qm, bm, qs, bs = dial.unpack_song(archive_path.read_bytes(), rate)
    if _q != Q or original != int(left.size):
        raise SystemExit("source archive does not match the song")
    cm, n_mid = band.mdct_forward(mid)
    if n_mid != original or cm.shape != qm.shape:
        raise SystemExit("cosine frame count drifted")
    source_bytes = archive_path.stat().st_size
    mid_qq, mid_b = dial.raise_deficit(cm, qm, bm, VOWEL)
    if not np.array_equal(mid_b[:, [0, 2, 3]], bm[:, [0, 2, 3]]):
        raise SystemExit("a band outside the vowel moved")
    if np.any(mid_qq[:, :32] != qm[:, :32]) or np.any(mid_qq[:, 64:] != qm[:, 64:]):
        raise SystemExit("quotients outside the vowel were rewritten")
    pairs = miss.pairs_for(rate, mid_qq.shape[1])
    blob = pack_song(Q, original, mid_qq, mid_b, qs, bs, rate)
    got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, rate)
    if (
        got_q != Q
        or got_n != original
        or not np.array_equal(got_mid, mid_qq)
        or not np.array_equal(got_side, qs)
        or not np.array_equal(got_mb, mid_b)
        or not np.array_equal(got_sb, bs)
    ):
        raise SystemExit("archive does not rebuild")
    back_m = miss.restore(mid_qq, mid_b, Q, original, pairs)
    back_s = miss.restore(qs, bs, Q, original, pairs)
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
    lo = VOCAL_LO * rate
    hi = VOCAL_HI * rate
    if hi > left.size:
        raise SystemExit("vocal window is past the end of the song")
    vocal_mid = mid[lo:hi].astype(np.float64)
    vocal_err = back_m[lo:hi].astype(np.float64) - vocal_mid
    vocal = octave_snr(vocal_mid, vocal_err, rate)
    hop_m = miss.mean.hop_mean(back_m.astype(np.float64) - mid, mid_qq.shape[0])
    hop_s = miss.mean.hop_mean(back_s.astype(np.float64) - side, mid_qq.shape[0])
    wav_path = out / f"{STEM}_q{Q}.wav"
    deliver.write_wav(wav_path, play_l, play_r, rate)
    (out / f"{STEM}_q{Q}.fspz").write_bytes(blob)
    row = {
        "q": Q,
        "dial_band": VOWEL,
        "dial_name": NAMES[VOWEL],
        "channel": "mid",
        "vocal_window_s": [VOCAL_LO, VOCAL_HI],
        "archive_bytes": len(blob),
        "r16_bytes": source_bytes,
        "opus96_bytes": opus[STEM],
        "ratio_vs_opus96": len(blob) / opus[STEM],
        "ratio_vs_r16": len(blob) / source_bytes,
        "under_opus": len(blob) < opus[STEM],
        "mid_zoom_before": float(bm[:, VOWEL].mean()),
        "mid_zoom_after": float(mid_b[:, VOWEL].mean()),
        "side_zoom": float(bs[:, VOWEL].mean()),
        "mean_abs_error": float(np.mean(np.abs(err))),
        "hop_mean_mid": float(hop_m[hop_m > 0].mean()),
        "hop_mean_side": float(hop_s[hop_s > 0].mean()),
        "snr_db": snr,
        "vocal_mid_snr_db": {name: value for name, value in zip(NAMES, vocal)},
        "wav": str(wav_path),
    }
    paths.save_score("vowel_report.json", {
        "pin": fspack.PIN,
        "declared": {
            "rule": "one finer rung on VNV mid 2.8-5.5 kHz",
            "kept": "side and every other band stay",
            "vocal_window_s": [VOCAL_LO, VOCAL_HI],
            "songs": [STEM],
            "coder": "tenth-pass magnitude bucket, not retuned",
            "q": Q,
        },
        "songs": [{"stem": STEM, "source": str(path), "seconds": left.size / rate, "row": row}],
    })
    vocal_text = " ".join(f"{name} {value:.1f}" for name, value in zip(NAMES, vocal))
    print(
        f"  archive {len(blob)} r16 {source_bytes} opus {opus[STEM]} "
        f"ratio {row['ratio_vs_opus96']:.3f} vs r16 {row['ratio_vs_r16']:.3f} "
        f"mid zoom {row['mid_zoom_before']:.2f}->{row['mid_zoom_after']:.2f} "
        f"side {row['side_zoom']:.2f} "
        f"mean_abs {row['mean_abs_error']:.1f} "
        f"hop {row['hop_mean_mid']:.1f}/{row['hop_mean_side']:.1f} "
        f"SNR {snr:.2f} vocal {vocal_text}",
        flush=True,
    )
    print("vowel done", flush=True)


if __name__ == "__main__":
    main()
