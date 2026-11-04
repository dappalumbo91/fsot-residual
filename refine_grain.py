#!/usr/bin/env python3
"""Fourth audio pass. Declared before this pass is scored.

The third pass put the time-sample step Q on the cosine coefficients.
Those coefficients are the unscaled sum over the window. The inverse
multiplies them by 2/N, and N is 256, so a coefficient count is 128
times a time count. Copying Q across that unit made the low band finer
than the declared step and the archive grew.

The coefficient step is therefore Q * (N/2) = Q * 128. The 128 is the
inverse factor already fixed by the rebuild test. It is not a listened
preference. Everything else stays as in the third pass: steps (4, 1, 1, 1),
window 512, hop 256, Q=1 stores the raw approximation, zlib level 9.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import fspack
import refine_audio as r2
import paths
import refine_tonal as tonal

GRAIN = tonal.HOP // 2  # 2/N inverse, N = 256
OUT = tonal.OUT


def encode_song(path: Path) -> dict:
    stem = path.stem
    print(f"song {stem}", flush=True)
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    mid_bands, original = deliver.haar_split(mid, 3)
    side_bands, _ = deliver.haar_split(side, 3)
    rows = []
    for q in deliver.AUDIO_Q:
        print(f"  Q={q}", flush=True)
        mid_codes, _ = r2.quantize_plain(mid_bands[:3], q)
        side_codes, _ = r2.quantize_plain(side_bands[:3], q)
        coeff_step = 1 if q <= 1 else q * GRAIN
        if q <= 1:
            mid_qq = mid_bands[3].astype(np.int32, copy=False)
            side_qq = side_bands[3].astype(np.int32, copy=False)
            mid_approx = mid_qq
            side_approx = side_qq
        else:
            mid_qq, mid_approx = tonal.quantize_mdct(mid_bands[3], coeff_step)
            side_qq, side_approx = tonal.quantize_mdct(side_bands[3], coeff_step)
        blob = tonal.pack(mid_codes, mid_qq, side_codes, side_qq, q)
        _bq, back_md, back_ma, back_sd, back_sa = tonal.unpack(blob)
        if q <= 1:
            mid_back, side_back = back_ma, back_sa
        else:
            mid_back = tonal.mdct_inverse(back_ma.astype(np.float64) * coeff_step, mid_bands[3].size)
            side_back = tonal.mdct_inverse(back_sa.astype(np.float64) * coeff_step, side_bands[3].size)
            if not np.array_equal(mid_back, mid_approx) or not np.array_equal(side_back, side_approx):
                raise SystemExit(f"{stem} Q={q} did not rebuild")
        mid_hats = []
        side_hats = []
        for codes, hats in ((back_md, mid_hats), (back_sd, side_hats)):
            for code, times in zip(codes, r2.REFINE_TIMES[:3]):
                step = 1 if q <= 1 else q * times
                hats.append(code if step <= 1 else code * step)
        mid_hats.append(mid_back)
        side_hats.append(side_back)
        got_l, got_r = deliver.unmid_side(
            deliver.haar_join(mid_hats, original),
            deliver.haar_join(side_hats, original),
        )
        got_l = deliver.clip_i16(got_l).astype(np.int32)
        got_r = deliver.clip_i16(got_r).astype(np.int32)
        if q == 1 and not (np.array_equal(got_l, left) and np.array_equal(got_r, right)):
            raise SystemExit(f"{stem} Q=1 is not sample-exact")
        err_l = got_l.astype(np.float64) - left
        err_r = got_r.astype(np.float64) - right
        noise = float(np.mean(np.concatenate([err_l, err_r]) ** 2))
        signal = float(np.mean(np.concatenate([left.astype(np.float64), right.astype(np.float64)]) ** 2))
        snr = math.inf if noise == 0 else 10 * math.log10(signal / noise)
        level = 0.0 if noise == 0 else 20 * math.log10(
            math.sqrt(float(np.mean(np.concatenate([got_l, got_r]).astype(np.float64) ** 2))) / math.sqrt(signal)
        )
        e28 = r2.band_energy(err_l, rate, 2000, 8000) + r2.band_energy(err_r, rate, 2000, 8000)
        eall = float(np.dot(err_l, err_l) + np.dot(err_r, err_r))
        wav_path = None
        if q == paths.LISTEN_Q:
            wav_path = OUT / f"{stem}_q{q}.wav"
            deliver.write_wav(wav_path, got_l, got_r, rate)
            (OUT / f"{stem}_q{q}.fspz").write_bytes(blob)
        row = {
            "q": q,
            "coeff_step": coeff_step,
            "archive_bytes": len(blob),
            "first_archive_bytes": None,
            "snr_db": None if math.isinf(snr) else snr,
            "level_delta_db": level,
            "error_2_8kHz_share": 0.0 if eall == 0 else e28 / eall,
            "error_2_8kHz_power": e28,
            "exact": bool(noise == 0),
            "wav": None if wav_path is None else str(wav_path),
            "source_bytes": path.stat().st_size,
        }
        rows.append(row)
        print(
            f"    archive {row['archive_bytes']} first {row['first_archive_bytes']} "
            f"level {level:.3f} dB share_2_8 {row['error_2_8kHz_share']:.3f} "
            f"{'exact' if noise == 0 else f'SNR {snr:.2f}'}",
            flush=True,
        )
    return {"stem": stem, "ladder": rows}


def main() -> None:
    global OUT
    OUT = paths.begin_pass("r4")
    tonal.self_check()
    songs = [encode_song(path) for path in deliver.SONGS]
    path = paths.save_score("grain_report.json", {
        "pin": fspack.PIN,
        "declared": {
            "coefficient_step": "Q * 128",
            "grain": GRAIN,
            "q": list(deliver.AUDIO_Q),
            "listen_q": paths.LISTEN_Q,
        },
        "songs": songs,
    })
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
