#!/usr/bin/env python3
"""Ninth audio pass. Declared before this pass is scored.

The eighth pass left four misses at Q=2048: Skrillex, Bruno Mars,
VNV Nation, and Cage the Elephant. Sublime, Clay Walker, and Eminem
stay as scored. Their archives are already under Opus 96.

A one-step difference of those residuals was measured, along time at
one hop, along time at one window, along frequency, and from mid to
side. Each one raised the entropy. That predictor is not this pass.

The wrong term is the zoom. One b still covers every band in the frame,
so a low band that needs a fine step drags the octaves above it with it.
The natural zoom is per band: the smallest b at which that band's
residual still leaves the frame inside the floor.

This pass, fixed here:
- Same Q=2048, same mean floor Q/4, same steps (Q*128)/2^b, b = 0..8.
- Start from the seventh-pass frame zoom.
- Coarsen bands in this order: above 11 kHz, then 5.5–11, then 2.8–5.5,
  then the approximation. Those are the fs/4, fs/8, and fs/16 cuts.
- A band keeps the smallest b that leaves every open frame's mean
  absolute hop error at or under Q/4. It is not allowed to go finer
  than the frame zoom.
- Each band is then order-0 rANS, as in the eighth pass.
- The three wins are not re-encoded.
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import fspack
import paths
import refine_fullband as band
import refine_gauge as gauge
import refine_mean as mean
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPB"
MISSES = (
    "skrillex_first_of_the_year",
    "bruno_mars_just_the_way_you_are",
    "vnv_nation_beloved",
    "cage_the_elephant_aint_no_rest",
)


def pairs_for(rate: int, n_bins: int) -> list[tuple[int, int]]:
    edges = gauge.band_edges(n_bins, rate, zoom.HOP)
    return list(zip(edges, edges[1:]))


def hat_from(coeffs: np.ndarray, b_band: np.ndarray, q: int, pairs) -> np.ndarray:
    hat = np.empty(coeffs.shape, dtype=np.float64)
    for bi, (lo, hi) in enumerate(pairs):
        sl = coeffs[:, lo:hi]
        for b in range(zoom.B_MAX + 1):
            pick = b_band[:, bi] == b
            if not np.any(pick):
                continue
            step = zoom.step_at(q, b)
            qq = band.quantize(sl[pick], step)
            hat[np.ix_(np.flatnonzero(pick), np.arange(lo, hi))] = qq.astype(np.float64) * step
    return hat


def quotients(coeffs: np.ndarray, b_band: np.ndarray, q: int, pairs) -> np.ndarray:
    qq = np.zeros(coeffs.shape, dtype=np.int32)
    for bi, (lo, hi) in enumerate(pairs):
        sl = coeffs[:, lo:hi]
        for b in range(zoom.B_MAX + 1):
            pick = b_band[:, bi] == b
            if not np.any(pick):
                continue
            qq[np.ix_(np.flatnonzero(pick), np.arange(lo, hi))] = band.quantize(sl[pick], zoom.step_at(q, b))
    return qq


def restore(qq: np.ndarray, b_band: np.ndarray, q: int, original: int, pairs) -> np.ndarray:
    hat = np.zeros(qq.shape, dtype=np.float64)
    for bi, (lo, hi) in enumerate(pairs):
        for b in range(zoom.B_MAX + 1):
            pick = b_band[:, bi] == b
            if not np.any(pick):
                continue
            hat[np.ix_(np.flatnonzero(pick), np.arange(lo, hi))] = qq[pick, lo:hi].astype(np.float64) * zoom.step_at(q, b)
    return band.mdct_inverse(hat, original)


def passing(samples: np.ndarray, recon: np.ndarray, frames: int, q: int) -> np.ndarray:
    err = recon.astype(np.float64) - samples.astype(np.float64)
    average = mean.hop_mean(err, frames)
    owned = np.arange(frames) * zoom.HOP < samples.size
    return owned & (average <= (q / 4.0))


def coarsen(samples: np.ndarray, coeffs: np.ndarray, q: int, base_b: np.ndarray, rate: int) -> np.ndarray:
    frames = int(coeffs.shape[0])
    pairs = pairs_for(rate, coeffs.shape[1])
    base_b = base_b.astype(np.int16, copy=False)
    b_band = np.repeat(base_b[:, None], len(pairs), axis=1)
    original = int(samples.size)
    owned = np.arange(frames) * zoom.HOP < original
    b_band[~owned, :] = 0
    # High octaves first. The approximation is last.
    # A frame that never gets under the floor on this band goes back to the
    # frame zoom. Overlap can still push a hop over the floor. That miss
    # stays a miss and is reported. It is not repaired by a second search.
    for bi in range(len(pairs) - 1, -1, -1):
        open_mask = owned.copy()
        saved = b_band[:, bi].copy()
        for b in range(zoom.B_MAX + 1):
            if not np.any(open_mask):
                break
            b_band[open_mask, bi] = np.minimum(base_b[open_mask], np.int16(b))
            hat = hat_from(coeffs, b_band, q, pairs)
            recon = band.mdct_inverse(hat, original)
            take = open_mask & passing(samples, recon, frames, q)
            open_mask[take] = False
            give_up = open_mask & (base_b <= b)
            if np.any(give_up):
                b_band[give_up, bi] = saved[give_up]
                open_mask[give_up] = False
        if np.any(open_mask):
            b_band[open_mask, bi] = saved[open_mask]
    return b_band


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
            parts.append(gauge.pack_plane(plane[:, lo:hi]))
    return b"".join(parts)


def unpack_song(blob: bytes, rate: int):
    if blob[:4] != MAGIC:
        raise SystemExit("not a per-band zoom archive")
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
            values, offset = gauge.unpack_plane(blob, offset)
            width = edges[band_i + 1] - edges[band_i]
            if values.size != frames * width:
                raise SystemExit("band length does not match the octave cut")
            cols.append(values.reshape(frames, width))
        planes.append(np.column_stack(cols))
    if offset != len(blob):
        raise SystemExit("trailing per-band zoom bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    gauge.self_check()
    rate = 44100
    tone_w = np.pi * 10.5 / zoom.HOP
    tone = (np.sin(np.arange(8192) * tone_w) * 12000).astype(np.int32)
    coeffs, _n = band.mdct_forward(tone)
    base = mean.choose_mean(tone, coeffs, Q)
    refined = coarsen(tone, coeffs, Q, base, rate)
    high = float(np.median(refined[:, 3]))
    low = float(np.median(refined[:, 0]))
    print(f"miss self-check low-band median {low:.0f} high-band median {high:.0f}", flush=True)
    if not high < low:
        raise SystemExit("the top octave did not go coarser than the approximation")
    zeros = np.zeros(4096, dtype=np.int32)
    zc, _zn = band.mdct_forward(zeros)
    zb = coarsen(zeros, zc, Q, mean.choose_mean(zeros, zc, Q), rate)
    if int(np.max(zb)) != 0:
        raise SystemExit("silence did not stay coarse")
    raw = np.random.default_rng(9).integers(-8000, 8000, 3000).astype(np.int32)
    rc, original = band.mdct_forward(raw)
    rb = mean.choose_mean(raw, rc, Q)
    same = np.repeat(rb.astype(np.int16)[:, None], 4, axis=1)
    pairs = pairs_for(rate, rc.shape[1])
    if not np.array_equal(quotients(rc, same, Q, pairs), zoom.apply_zoom(rc, rb, Q)):
        raise SystemExit("a uniform per-band zoom does not match the frame zoom")
    print("miss self-check ok", flush=True)


def r8_bytes() -> dict[str, int]:
    report = json.loads((paths.SCORES / "gauge_report.json").read_text(encoding="utf-8"))
    return {song["stem"]: int(song["row"]["archive_bytes"]) for song in report["songs"]}


def main() -> None:
    self_check()
    out = paths.begin_pass("r9")
    opus, _previous = zoom.opus_and_r5()
    earlier = r8_bytes()
    songs = []
    wanted = set(MISSES)
    for path, stem in zip(zoom.SONGS, zoom.STEMS):
        if stem not in wanted:
            continue
        print(f"song {stem}", flush=True)
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        channels = []
        for ch in (mid, side):
            coeffs, original = band.mdct_forward(ch)
            base = mean.choose_mean(ch, coeffs, Q)
            b_band = coarsen(ch, coeffs, Q, base, rate)
            pairs = pairs_for(rate, coeffs.shape[1])
            qq = quotients(coeffs, b_band, Q, pairs)
            channels.append((original, qq, b_band, pairs))
        (original, mid_qq, mid_b, pairs) = channels[0]
        (_original_s, side_qq, side_b, _pairs_s) = channels[1]
        if original != _original_s:
            raise SystemExit("channel lengths differ")
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
        play_l, play_r = deliver.unmid_side(
            restore(mid_qq, mid_b, Q, original, pairs),
            restore(side_qq, side_b, Q, original, pairs),
        )
        play_l = deliver.clip_i16(play_l).astype(np.int32)
        play_r = deliver.clip_i16(play_r).astype(np.int32)
        err = np.concatenate([
            play_l.astype(np.float64) - left,
            play_r.astype(np.float64) - right,
        ])
        noise = float(np.mean(err ** 2))
        signal = float(np.mean(np.concatenate([left, right]).astype(np.float64) ** 2))
        snr = float("inf") if noise == 0 else 10 * __import__("math").log10(signal / noise)
        wav_path = out / f"{stem}_q{Q}.wav"
        deliver.write_wav(wav_path, play_l, play_r, rate)
        (out / f"{stem}_q{Q}.fspz").write_bytes(blob)
        means = np.mean(np.concatenate([mid_b, side_b], axis=0), axis=0)
        row = {
            "q": Q,
            "archive_bytes": len(blob),
            "r8_bytes": earlier[stem],
            "opus96_bytes": opus[stem],
            "ratio_vs_opus96": len(blob) / opus[stem],
            "ratio_vs_r8": len(blob) / earlier[stem],
            "mean_abs_error": float(np.mean(np.abs(err))),
            "mean_abs_limit": Q / 4.0,
            "floor_fraction_lr": float(np.mean(np.abs(err) <= (Q / 2.0))),
            "mean_zoom_a_d2_d1_d0": [float(v) for v in means],
            "snr_db": None if snr == float("inf") else snr,
            "wav": str(wav_path),
        }
        songs.append({"stem": stem, "source": str(path), "seconds": left.size / rate, "row": row})
        paths.save_score("miss_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "songs": list(MISSES),
                "keeps": "r8 archives for sublime, clay, eminem",
                "floor": "mean absolute hop error <= Q/4",
                "order": "d0, d1, d2, approximation",
                "coder": "static order-0 rANS per octave band",
            },
            "songs": songs,
        })
        print(
            f"  archive {len(blob)} r8 {earlier[stem]} opus {opus[stem]} "
            f"ratio {row['ratio_vs_opus96']:.3f} vs r8 {row['ratio_vs_r8']:.3f} "
            f"zoom a/d2/d1/d0 {means[0]:.2f}/{means[1]:.2f}/{means[2]:.2f}/{means[3]:.2f} "
            f"mean_abs {row['mean_abs_error']:.1f} SNR {snr:.2f}",
            flush=True,
        )
    print("miss done", flush=True)


if __name__ == "__main__":
    main()
