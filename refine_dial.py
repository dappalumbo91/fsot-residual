#!/usr/bin/env python3
"""Sixteenth audio pass. Declared before this pass is scored.

The wrong term is the gauge index on the octave furthest behind that
song's own Opus 96 decode. Measured on the fifteenth-pass playback
before this run, and not retuned after it:

  Skrillex 5.5-11 kHz is 2.0 dB against Opus 9.2 (band index 2).
  Cage     5.5-11 kHz is 1.2 dB against Opus 9.3 (band index 2).
  VNV      11-22 kHz is 0.2 dB against Opus 7.1 (band index 3).

Those bands sit on b = 0. This pass raises only that band, on both
channels, by one existing ladder rung: b becomes min(b + 1, 8), and
the quotients are retaken from the original coefficients at the new
step. Every other band stays, including 0-2.8 kHz. The low band is
not coarsened to pay for the rung. Q stays 2048. The tenth-pass
bucket is not retuned. Sublime, Clay, Eminem, and Bruno stay at r10.
Skrillex, Cage, and VNV start from r15.

A song whose archive crosses its Opus 96 file stays a size miss.
A band that is still behind Opus after one rung stays a quality miss.
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
import refine_fill as fill
import refine_fullband as band
import refine_gauge as gauge
import refine_miss as miss
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPD"
EDGES = (0, 32, 64, 128, 256)
NAMES = ("0-2.8", "2.8-5.5", "5.5-11", "11-22")
FREQS = ((0, 2756), (2756, 5512), (5512, 11025), (11025, 22050))
# Band index locked from the pre-score gap. Not chosen from the archive size.
DIAL = (
    ("skrillex_first_of_the_year", 2),
    ("cage_the_elephant_aint_no_rest", 2),
    ("vnv_nation_beloved", 3),
)


def raise_deficit(coeffs: np.ndarray, qq: np.ndarray, b_band: np.ndarray, band_index: int):
    if coeffs.shape != qq.shape:
        raise SystemExit("coefficient frame count does not match the archive")
    if band_index not in (0, 1, 2, 3):
        raise SystemExit("deficit band is not an octave")
    out_q = np.array(qq, copy=True)
    out_b = np.array(b_band, dtype=np.int16, copy=True)
    previous = out_b[:, band_index]
    nxt = previous.copy()
    movable = previous < zoom.B_MAX
    nxt[movable] = previous[movable] + 1
    lo, hi = EDGES[band_index], EDGES[band_index + 1]
    sl = coeffs[:, lo:hi]
    taken = np.zeros((coeffs.shape[0], hi - lo), dtype=np.int32)
    for b in range(zoom.B_MAX + 1):
        pick = nxt == b
        if not np.any(pick):
            continue
        taken[pick] = band.quantize(sl[pick], zoom.step_at(Q, int(b)))
    out_q[:, lo:hi] = taken
    out_b[:, band_index] = nxt
    return out_q, out_b


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
        raise SystemExit("not a dialed archive")
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
        raise SystemExit("trailing dialed bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def band_snr(left, right, play_l, play_r, rate: int) -> list[float]:
    sig = np.stack([left.astype(np.float64), right.astype(np.float64)])
    err = np.stack([play_l.astype(np.float64) - left, play_r.astype(np.float64) - right])
    spec_s = np.abs(np.fft.rfft(sig, axis=1)) ** 2
    spec_e = np.abs(np.fft.rfft(err, axis=1)) ** 2
    freqs = np.fft.rfftfreq(sig.shape[1], 1 / rate)
    out = []
    for lo, hi in FREQS:
        sl = (freqs >= lo) & (freqs < hi)
        power = float(spec_s[:, sl].sum())
        noise = float(spec_e[:, sl].sum())
        out.append(10 * math.log10(power / max(noise, 1e-12)))
    return out


def self_check() -> None:
    gauge.self_check()
    rng = np.random.default_rng(16)
    coeffs = rng.normal(0, 80000, (5, 256))
    qq = band.quantize(coeffs, zoom.step_at(Q, 0)).astype(np.int32)
    b_band = np.zeros((5, 4), dtype=np.int16)
    b_band[:, 2] = np.array([0, 1, 8, 3, 7], dtype=np.int16)
    out_q, out_b = raise_deficit(coeffs, qq, b_band, 2)
    if not np.array_equal(out_b[:, 2], np.array([1, 2, 8, 4, 8], dtype=np.int16)):
        raise SystemExit("the deficit rung did not move by one")
    if not np.array_equal(out_b[:, [0, 1, 3]], b_band[:, [0, 1, 3]]):
        raise SystemExit("a band outside the deficit moved")
    if np.any(out_q[:, :64] != qq[:, :64]) or np.any(out_q[:, 128:] != qq[:, 128:]):
        raise SystemExit("quotients outside the deficit were rewritten")
    for frame, rung in ((0, 1), (2, 8)):
        expect = band.quantize(coeffs[frame, 64:128], zoom.step_at(Q, rung))
        if not np.array_equal(out_q[frame, 64:128], expect):
            raise SystemExit("the deficit was not retaken from the original coefficients")
    blob = pack_song(Q, 5 * zoom.HOP, out_q, out_b, out_q, out_b, 44100)
    got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, 44100)
    if got_q != Q or got_n != 5 * zoom.HOP:
        raise SystemExit("header did not round-trip")
    if not np.array_equal(got_mid, out_q) or not np.array_equal(got_side, out_q):
        raise SystemExit("dialed coefficients did not round-trip")
    if not np.array_equal(got_mb, out_b) or not np.array_equal(got_sb, out_b):
        raise SystemExit("dialed zoom did not round-trip")
    print("dial self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("r16")
    opus, _previous = zoom.opus_and_r5()
    songs = []
    for stem, band_index in DIAL:
        print(f"song {stem}", flush=True)
        path = band.SONGS[band.STEMS.index(stem)]
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        archive_path = paths.CURRENT / "r15" / f"{stem}_q{Q}.fspz"
        _q, original, qm, bm, qs, bs = fill.unpack_song(archive_path.read_bytes(), rate)
        if _q != Q or original != int(left.size):
            raise SystemExit(f"{stem} source archive does not match the song")
        cm, n_mid = band.mdct_forward(mid)
        cs, n_side = band.mdct_forward(side)
        if n_mid != original or n_side != original or cm.shape != qm.shape:
            raise SystemExit(f"{stem} cosine frame count drifted")
        source_bytes = archive_path.stat().st_size
        mid_qq, mid_b = raise_deficit(cm, qm, bm, band_index)
        side_qq, side_b = raise_deficit(cs, qs, bs, band_index)
        untouched = [i for i in range(4) if i != band_index]
        if not np.array_equal(mid_b[:, untouched], bm[:, untouched]):
            raise SystemExit(f"{stem} mid zoom moved outside the deficit")
        if not np.array_equal(side_b[:, untouched], bs[:, untouched]):
            raise SystemExit(f"{stem} side zoom moved outside the deficit")
        lo, hi = EDGES[band_index], EDGES[band_index + 1]
        if np.any(mid_qq[:, :lo] != qm[:, :lo]) or np.any(mid_qq[:, hi:] != qm[:, hi:]):
            raise SystemExit(f"{stem} mid quotients moved outside the deficit")
        if np.any(side_qq[:, :lo] != qs[:, :lo]) or np.any(side_qq[:, hi:] != qs[:, hi:]):
            raise SystemExit(f"{stem} side quotients moved outside the deficit")
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
        octaves = band_snr(left, right, play_l, play_r, rate)
        hop_m = miss.mean.hop_mean(back_m.astype(np.float64) - mid, mid_qq.shape[0])
        hop_s = miss.mean.hop_mean(back_s.astype(np.float64) - side, side_qq.shape[0])
        wav_path = out / f"{stem}_q{Q}.wav"
        deliver.write_wav(wav_path, play_l, play_r, rate)
        (out / f"{stem}_q{Q}.fspz").write_bytes(blob)
        row = {
            "q": Q,
            "dial_band": band_index,
            "dial_name": NAMES[band_index],
            "archive_bytes": len(blob),
            "r15_bytes": source_bytes,
            "opus96_bytes": opus[stem],
            "ratio_vs_opus96": len(blob) / opus[stem],
            "ratio_vs_r15": len(blob) / source_bytes,
            "under_opus": len(blob) < opus[stem],
            "mid_zoom_before": float(bm[:, band_index].mean()),
            "mid_zoom_after": float(mid_b[:, band_index].mean()),
            "side_zoom_before": float(bs[:, band_index].mean()),
            "side_zoom_after": float(side_b[:, band_index].mean()),
            "mean_abs_error": float(np.mean(np.abs(err))),
            "hop_mean_mid": float(hop_m[hop_m > 0].mean()),
            "hop_mean_side": float(hop_s[hop_s > 0].mean()),
            "snr_db": snr,
            "band_snr_db": {name: value for name, value in zip(NAMES, octaves)},
            "wav": str(wav_path),
        }
        songs.append({"stem": stem, "source": str(path), "seconds": left.size / rate, "row": row})
        paths.save_score("dial_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "rule": "one finer rung on the octave furthest behind that song's Opus 96 file",
                "bands": {stem: NAMES[index] for stem, index in DIAL},
                "kept": "every other band stays, including 0-2.8 kHz",
                "songs": [item[0] for item in DIAL],
                "coder": "tenth-pass magnitude bucket, not retuned",
                "q": Q,
            },
            "songs": songs,
        })
        octave_text = " ".join(f"{name} {value:.1f}" for name, value in zip(NAMES, octaves))
        print(
            f"  archive {len(blob)} r15 {source_bytes} opus {opus[stem]} "
            f"ratio {row['ratio_vs_opus96']:.3f} vs r15 {row['ratio_vs_r15']:.3f} "
            f"band {NAMES[band_index]} zoom {row['mid_zoom_before']:.2f}->{row['mid_zoom_after']:.2f}"
            f"/{row['side_zoom_before']:.2f}->{row['side_zoom_after']:.2f} "
            f"mean_abs {row['mean_abs_error']:.1f} "
            f"hop {row['hop_mean_mid']:.1f}/{row['hop_mean_side']:.1f} "
            f"SNR {snr:.2f} {octave_text}",
            flush=True,
        )
    print("dial done", flush=True)


if __name__ == "__main__":
    main()
