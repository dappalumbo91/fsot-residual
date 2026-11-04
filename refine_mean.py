#!/usr/bin/env python3
"""Seventh audio pass. Declared before this pass is scored.

The sixth pass held the sample floor. Every grade put absolute error
inside Q/2. The zoom that did it sat near 5.5 on every song, and the
archives grew to between 2.4 and 13 times Opus 96, and between 2 and 12
times the fifth pass.

The wrong term is the frame test. It used the worst sample in the hop.
One sample set the zoom for all 256 bins. The direction scores a declared
distortion as an average, not as a maximum.

This pass, fixed here:
- Same cosine, same ladder, same zoom steps b = 0..8,
  step_b = (Q * 128) / 2^b.
- A frame takes the smallest b whose mean absolute hop error is at most
  Q/4. Q/4 is the mean absolute error of a uniform quantizer on
  [-Q/2, Q/2]. The uniform-b rebuild already includes both overlapping
  frames, so the overlap is not divided in a second time.
- The sixth pass stays as scored. This test is not moved after the scores.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import fspack
import paths
import refine_fullband as band
import refine_zoom as zoom

HOP = zoom.HOP


def hop_mean(error: np.ndarray, frames: int) -> np.ndarray:
    original = int(error.size)
    usable = min(frames, (original + HOP - 1) // HOP)
    out = np.zeros(frames, dtype=np.float64)
    if usable == 0:
        return out
    block = np.zeros(usable * HOP, dtype=np.float64)
    copied = min(original, block.size)
    block[:copied] = np.abs(error[:copied])
    out[:usable] = block.reshape(usable, HOP).mean(axis=1)
    if original < usable * HOP:
        start = (usable - 1) * HOP
        if original > start:
            out[usable - 1] = float(np.mean(np.abs(error[start:original])))
    return out


def choose_mean(samples: np.ndarray, coeffs: np.ndarray, q: int) -> np.ndarray:
    frames = int(coeffs.shape[0])
    original = int(samples.size)
    best = np.full(frames, zoom.B_MAX, dtype=np.int8)
    open_mask = np.ones(frames, dtype=bool)
    owned = np.arange(frames) * HOP < original
    best[~owned] = 0
    open_mask[~owned] = False
    limit = q / 4.0
    source = samples.astype(np.float64)
    for b in range(zoom.B_MAX + 1):
        if not np.any(open_mask):
            break
        step = zoom.step_at(q, b)
        qq = band.quantize(coeffs, step)
        recon = band.mdct_inverse(qq.astype(np.float64) * step, original)
        average = hop_mean(recon.astype(np.float64) - source, frames)
        take = open_mask & (average <= limit)
        best[take] = b
        open_mask[take] = False
    return best


def self_check() -> None:
    zeros = np.zeros(4096, dtype=np.int32)
    zc, _n = band.mdct_forward(zeros)
    if int(np.max(choose_mean(zeros, zc, 512))) != 0:
        raise SystemExit("silence did not stay coarse")
    tone = (np.sin(np.arange(8192) * 0.15) * 12000).astype(np.int32)
    noise = np.random.default_rng(3).integers(-12000, 12000, 8192).astype(np.int32)
    tone_b = choose_mean(tone, band.mdct_forward(tone)[0], 2048)
    noise_b = choose_mean(noise, band.mdct_forward(noise)[0], 32)
    print(
        f"mean self-check tone median {float(np.median(tone_b)):.0f} "
        f"noise median {float(np.median(noise_b)):.0f}",
        flush=True,
    )
    if float(np.median(tone_b)) > float(np.median(noise_b)):
        raise SystemExit("a tone took a finer zoom than noise")
    print("mean self-check ok", flush=True)


def encode_channels(mid, side, q: int):
    mid_c, original = band.mdct_forward(mid)
    side_c, original_s = band.mdct_forward(side)
    if original != original_s:
        raise SystemExit("channel lengths differ")
    mid_b = choose_mean(mid, mid_c, q)
    side_b = choose_mean(side, side_c, q)
    return original, zoom.apply_zoom(mid_c, mid_b, q), mid_b, zoom.apply_zoom(side_c, side_b, q), side_b


def main() -> None:
    self_check()
    out = paths.begin_pass("r7")
    opus, previous = zoom.opus_and_r5()
    songs = []
    for path, stem in zip(zoom.SONGS, zoom.STEMS):
        print(f"song {stem}", flush=True)
        started = time.perf_counter()
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        rows = []
        exact = band.pack_exact(left, right)
        rows.append({
            "q": 1,
            "archive_bytes": len(exact),
            "ratio_vs_opus96": len(exact) / opus[stem],
            "exact": True,
            "floor_fraction_lr": 1.0,
            "mean_abs_error": 0.0,
            "mean_zoom": 0.0,
            "wav": None,
            "kept_for_listening": False,
        })
        del exact
        keeper_audio = None
        for q in deliver.AUDIO_Q:
            if q <= 1:
                continue
            original, mid_qq, mid_b, side_qq, side_b = encode_channels(mid, side, q)
            blob = zoom.pack_zoom(q, original, mid_qq, mid_b, side_qq, side_b)
            got_l, got_r = deliver.unmid_side(
                zoom.restore_channel(mid_qq, mid_b, q, original),
                zoom.restore_channel(side_qq, side_b, q, original),
            )
            play_l = deliver.clip_i16(got_l).astype(np.int32)
            play_r = deliver.clip_i16(got_r).astype(np.int32)
            err = np.concatenate([
                play_l.astype(np.float64) - left,
                play_r.astype(np.float64) - right,
            ])
            noise = float(np.mean(err ** 2))
            signal = float(np.mean(np.concatenate([left, right]).astype(np.float64) ** 2))
            snr = float("inf") if noise == 0 else 10 * __import__("math").log10(signal / noise)
            covered = float(np.mean(np.abs(err) <= (q / 2.0)))
            mean_abs = float(np.mean(np.abs(err)))
            mean_b = float(np.mean(np.concatenate([mid_b, side_b]).astype(np.float64)))
            row = {
                "q": q,
                "archive_bytes": len(blob),
                "ratio_vs_opus96": len(blob) / opus[stem],
                "ratio_vs_r5": len(blob) / previous[stem][q],
                "floor_fraction_lr": covered,
                "mean_abs_error": mean_abs,
                "mean_abs_limit": q / 4.0,
                "mean_zoom": mean_b,
                "snr_db": None if snr == float("inf") else snr,
                "exact": False,
                "wav": None,
                "kept_for_listening": False,
            }
            rows.append(row)
            rank = (covered, -len(blob))
            if keeper_audio is None or rank > keeper_audio[0]:
                keeper_audio = (rank, play_l, play_r, blob)
            print(
                f"  Q={q} archive {len(blob)} vs opus {opus[stem]} "
                f"ratio {row['ratio_vs_opus96']:.3f} vs r5 {row['ratio_vs_r5']:.3f} "
                f"floor {covered:.3f} mean_abs {mean_abs:.1f} limit {q / 4:.1f} "
                f"zoom {mean_b:.2f} SNR {snr:.2f}",
                flush=True,
            )
            del mid_qq, side_qq, mid_b, side_b, blob, play_l, play_r
        keeper = max((row for row in rows if row["q"] > 1), key=lambda row: (row["floor_fraction_lr"], -row["archive_bytes"]))
        _rank, play_l, play_r, blob = keeper_audio
        wav_path = out / f"{stem}_q{keeper['q']}.wav"
        deliver.write_wav(wav_path, play_l, play_r, rate)
        (out / f"{stem}_q{keeper['q']}.fspz").write_bytes(blob)
        keeper["wav"] = str(wav_path)
        keeper["kept_for_listening"] = True
        songs.append({
            "stem": stem,
            "source": str(path),
            "opus96_bytes": opus[stem],
            "seconds": left.size / rate,
            "ladder": rows,
        })
        paths.save_score("mean_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "failure_of_r6": "hop maximum set the zoom for every bin",
                "frame_test": "mean absolute hop error <= Q/4",
                "q4": "mean absolute error of a uniform step-Q quantizer",
                "zoom": "(Q*128) / 2^b, b = 0..8",
                "q": list(deliver.AUDIO_Q),
            },
            "songs": songs,
        })
        print(f"  keep Q={keeper['q']} {time.perf_counter() - started:.1f}s", flush=True)
    print("mean done", flush=True)


if __name__ == "__main__":
    main()
