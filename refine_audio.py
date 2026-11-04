#!/usr/bin/env python3
"""Second audio pass. Declared before this pass is scored.

Diagnosis of the files already listened to, not a search over multipliers:
the 2-8 kHz static is largest in Haar detail d2 (about 2.8-5.5 kHz).
The top octave is not that static, and it is the cheapest band.
The approximation (below about 2.8 kHz) is the largest share of the bytes.
A one-sample delta of those coefficients got bigger under zlib, so the
gauge is not the previous sample. A tone repeats. The gauge is that lag.

Rule for this pass, fixed here:
- Viscosity multiplier 4 stays on the finest detail only (the octave above
  fs/4). The coarser three bands use step Q. The retired 2x multipliers are
  not replaced with a new number.
- The approximation stores one lag per block. The lag is the integer, from
  2 through 512, that minimizes the absolute error of that block against
  itself. Block length is 2048 approximation samples. The decoder reads the
  lag. It does not search again.
- Prediction is closed-loop: residual = sample - reconstructed[n - lag],
  then the same half-away quantizer. Before the first lag samples, predict 0.
- Mid/side, 3-level integer Haar, the Q ladder, and zlib level 9 are unchanged.
- Q=1 has to come back sample-exact.

The lag bounds are the band's own two-sample minimum and a power-of-two
window long enough to hold one cycle at the bottom of this downsampled band.
They are not moved after the scores.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import wave
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import fspack
import paths

REFINE_TIMES = (4, 1, 1, 1)  # finest detail, next, next, approximation
LAG_BLOCK = 2048
LAG_MIN = 2
LAG_MAX = 512
OUT = deliver.WATCH / "refine"
SONGS = deliver.SONGS


def choose_lags(band: np.ndarray) -> np.ndarray:
    values = band.astype(np.int64, copy=False)
    count = int(values.size)
    blocks = (count + LAG_BLOCK - 1) // LAG_BLOCK
    best = np.full(blocks, LAG_MIN, dtype=np.int32)
    best_cost = np.full(blocks, np.inf)
    block_id = np.arange(count) // LAG_BLOCK
    for lag in range(LAG_MIN, LAG_MAX + 1):
        diff = np.abs(values[lag:] - values[:-lag])
        ids = block_id[lag:]
        sums = np.bincount(ids, weights=diff.astype(np.float64), minlength=blocks)
        valid = ((np.arange(blocks) + 1) * LAG_BLOCK - 1) >= lag
        better = valid & (sums < best_cost)
        best_cost[better] = sums[better]
        best[better] = lag
    return best


def quantize_predicted(band: np.ndarray, lags: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    source = band.astype(np.int32, copy=False)
    count = int(source.size)
    qq = np.empty(count, dtype=np.int32)
    recon = np.empty(count, dtype=np.int32)
    prev = 0
    half = step // 2
    for index in range(count):
        lag = int(lags[index // LAG_BLOCK])
        pred = 0 if index < lag else int(recon[index - lag])
        resid = int(source[index]) - pred
        if step <= 1:
            hat = resid
            q = resid
        elif resid >= 0:
            q = (resid + half) // step
            hat = q * step
        else:
            q = -(((-resid) + half) // step)
            hat = q * step
        prev = pred + hat
        qq[index] = q
        recon[index] = prev
    return qq, recon


def restore_predicted(qq: np.ndarray, lags: np.ndarray, step: int) -> np.ndarray:
    count = int(qq.size)
    recon = np.empty(count, dtype=np.int32)
    for index in range(count):
        lag = int(lags[index // LAG_BLOCK])
        pred = 0 if index < lag else int(recon[index - lag])
        hat = int(qq[index]) if step <= 1 else int(qq[index]) * step
        recon[index] = pred + hat
    return recon


def quantize_plain(bands: list[np.ndarray], q: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    codes = []
    hats = []
    for band, times in zip(bands, REFINE_TIMES):
        step = 1 if q <= 1 else q * times
        qq, hat = deliver.step_of(band, step)
        codes.append(qq)
        hats.append(hat if step > 1 else band.astype(np.int32, copy=False))
    return codes, hats


def restore_plain(codes: list[np.ndarray], q: int, original: int) -> np.ndarray:
    hats = []
    for qq, times in zip(codes, REFINE_TIMES):
        step = 1 if q <= 1 else q * times
        hats.append(qq.astype(np.int32) if step <= 1 else qq.astype(np.int32) * step)
    return deliver.haar_join(hats, original)


def pack(mid_details: list[np.ndarray], mid_approx: np.ndarray, side_details: list[np.ndarray], side_approx: np.ndarray, mid_lags: np.ndarray, side_lags: np.ndarray) -> bytes:
    planes = list(mid_details) + [mid_approx] + list(side_details) + [side_approx]
    blob = bytearray()
    blob += struct.pack("<HH", int(mid_lags.size), int(side_lags.size))
    blob += np.ascontiguousarray(mid_lags, dtype="<u2").tobytes()
    blob += np.ascontiguousarray(side_lags, dtype="<u2").tobytes()
    blob += struct.pack("<B", len(planes))
    for plane in planes:
        data = np.ascontiguousarray(plane, dtype="<i4")
        blob += struct.pack("<I", int(data.size))
        blob += data.tobytes()
    return zlib.compress(bytes(blob), 9)


def unpack(blob: bytes):
    raw = zlib.decompress(blob)
    mid_n, side_n = struct.unpack_from("<HH", raw, 0)
    pos = 4
    mid_lags = np.frombuffer(raw[pos:pos + mid_n * 2], dtype="<u2").astype(np.int32).copy()
    pos += mid_n * 2
    side_lags = np.frombuffer(raw[pos:pos + side_n * 2], dtype="<u2").astype(np.int32).copy()
    pos += side_n * 2
    (count,) = struct.unpack_from("<B", raw, pos)
    pos += 1
    planes = []
    for _ in range(count):
        (size,) = struct.unpack_from("<I", raw, pos)
        pos += 4
        plane = np.frombuffer(raw[pos:pos + size * 4], dtype="<i4").astype(np.int32).copy()
        pos += size * 4
        planes.append(plane)
    if pos != len(raw) or count != 8:
        raise RuntimeError("bad refine archive")
    return planes[:3], planes[3], planes[4:7], planes[7], mid_lags, side_lags


def band_energy(samples: np.ndarray, rate: int, low: float, high: float) -> float:
    spec = np.fft.rfft(samples.astype(np.float64))
    freqs = np.fft.rfftfreq(samples.size, 1.0 / rate)
    spec[(freqs < low) | (freqs >= high)] = 0
    shaped = np.fft.irfft(spec, n=samples.size)
    return float(np.dot(shaped, shaped))


def self_check() -> None:
    tone = (np.sin(np.arange(8000) * (2 * math.pi * 6 / 64)) * 8000).astype(np.int32)
    lags = choose_lags(tone)
    if int(lags.min()) < LAG_MIN or int(lags.max()) > LAG_MAX:
        raise SystemExit("lag left the declared range")
    qq, recon = quantize_predicted(tone, lags, 1)
    back = restore_predicted(qq, lags, 1)
    if not np.array_equal(recon, tone) or not np.array_equal(back, tone):
        raise SystemExit("predicted Q=1 missed the tone")
    packed = pack(
        [np.array([1, -2], dtype=np.int32), np.array([3], dtype=np.int32), np.array([4], dtype=np.int32)],
        np.array([5], dtype=np.int32),
        [np.array([6, -7], dtype=np.int32), np.array([8], dtype=np.int32), np.array([9], dtype=np.int32)],
        np.array([10], dtype=np.int32),
        np.array([11, 12], dtype=np.int32),
        np.array([13], dtype=np.int32),
    )
    details, approx, _side_d, _side_a, lags_m, lags_s = unpack(packed)
    if int(lags_m[0]) != 11 or int(lags_s[0]) != 13 or int(approx[0]) != 5 or int(details[0][1]) != -2:
        raise SystemExit("pack round-trip failed")
    print("self-check core ok", flush=True)


def encode_channel(bands: list[np.ndarray], q: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    detail_codes, detail_hats = quantize_plain(bands[:3], q)
    approx = bands[3]
    lags = choose_lags(approx)
    step = 1 if q <= 1 else q * REFINE_TIMES[3]
    qq, recon = quantize_predicted(approx, lags, step)
    # Round-trip the stored indices, not the encoder's private recon.
    back = restore_predicted(qq, lags, step)
    if not np.array_equal(back, recon):
        raise SystemExit("approximation archive does not rebuild")
    return detail_codes, qq, lags, detail_hats, recon


def load_q1(stem: str) -> tuple[np.ndarray, np.ndarray, int]:
    path = deliver.WATCH / f"{stem}_q1.wav"
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.int32)
    return pcm[0::2].copy(), pcm[1::2].copy(), rate


def write_report(rows: list[dict]) -> None:
    path = paths.save_score("refine_report.json", {
        "pin": fspack.PIN,
        "declared": {
            "refine_times_fine_to_approx": list(REFINE_TIMES),
            "lag_block": LAG_BLOCK,
            "lag_min": LAG_MIN,
            "lag_max": LAG_MAX,
            "q": list(deliver.AUDIO_Q),
            "listen_q": paths.LISTEN_Q,
            "viscosity": "4x on the finest Haar detail only",
            "gauge": "one lag per approximation block, closed-loop residual",
        },
        "songs": rows,
    })
    print(f"wrote {path}", flush=True)


def main() -> None:
    global OUT
    OUT = paths.begin_pass("r2")
    self_check()
    rows = []
    for path in SONGS:
        stem = path.stem
        print(f"song {stem}", flush=True)
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        mid_bands, original = deliver.haar_split(mid, 3)
        side_bands, original_s = deliver.haar_split(side, 3)
        if original != original_s:
            raise SystemExit("channel lengths differ")
        song_rows = []
        for q in deliver.AUDIO_Q:
            print(f"  Q={q}", flush=True)
            mid_d, mid_a, mid_lags, _mh, _mr = encode_channel(mid_bands, q)
            side_d, side_a, side_lags, _sh, _sr = encode_channel(side_bands, q)
            blob = pack(mid_d, mid_a, side_d, side_a, mid_lags, side_lags)
            back_md, back_ma, back_sd, back_sa, back_ml, back_sl = unpack(blob)
            if not np.array_equal(back_ml, mid_lags) or not np.array_equal(back_sl, side_lags):
                raise SystemExit("lags did not round-trip")
            mid_hats = []
            for code, times in zip(back_md, REFINE_TIMES[:3]):
                step = 1 if q <= 1 else q * times
                mid_hats.append(code if step <= 1 else code * step)
            side_hats = []
            for code, times in zip(back_sd, REFINE_TIMES[:3]):
                step = 1 if q <= 1 else q * times
                side_hats.append(code if step <= 1 else code * step)
            approx_step = 1 if q <= 1 else q * REFINE_TIMES[3]
            mid_hats.append(restore_predicted(back_ma, back_ml, approx_step))
            side_hats.append(restore_predicted(back_sa, back_sl, approx_step))
            mid_r = deliver.haar_join(mid_hats, original)
            side_r = deliver.haar_join(side_hats, original)
            got_l, got_r = deliver.unmid_side(mid_r, side_r)
            got_l = deliver.clip_i16(got_l).astype(np.int32)
            got_r = deliver.clip_i16(got_r).astype(np.int32)
            if q == 1 and not (np.array_equal(got_l, left) and np.array_equal(got_r, right)):
                raise SystemExit(f"{stem} Q=1 is not sample-exact")
            err_l = got_l.astype(np.float64) - left.astype(np.float64)
            err_r = got_r.astype(np.float64) - right.astype(np.float64)
            err = np.concatenate([err_l, err_r])
            sig = np.concatenate([left.astype(np.float64), right.astype(np.float64)])
            noise = float(np.mean(err * err))
            signal = float(np.mean(sig * sig))
            snr = math.inf if noise == 0 else 10 * math.log10(signal / noise)
            level = 0.0 if noise == 0 else 20 * math.log10(
                math.sqrt(float(np.mean(np.concatenate([got_l, got_r]).astype(np.float64) ** 2)))
                / math.sqrt(signal)
            )
            e28 = band_energy(err_l, rate, 2000, 8000) + band_energy(err_r, rate, 2000, 8000)
            eall = float(np.dot(err_l, err_l) + np.dot(err_r, err_r))
            wav_path = None
            if q == paths.LISTEN_Q:
                wav_path = OUT / f"{stem}_q{q}.wav"
                deliver.write_wav(wav_path, got_l, got_r, rate)
                (OUT / f"{stem}_q{q}.fspz").write_bytes(blob)
            row = {
                "q": q,
                "archive_bytes": len(blob),
                "previous_archive_bytes": None,
                "snr_db": None if math.isinf(snr) else snr,
                "level_delta_db": level,
                "error_2_8kHz_power": e28,
                "error_2_8kHz_share": 0.0 if eall == 0 else e28 / eall,
                "max_abs": int(np.max(np.abs(err))),
                "exact": bool(noise == 0),
                "wav": None if wav_path is None else str(wav_path),
            }
            song_rows.append(row)
            print(
                f"    archive {row['archive_bytes']} was {row['previous_archive_bytes']} "
                f"level {level:.3f} dB share_2_8 {row['error_2_8kHz_share']:.3f} "
                f"{'exact' if noise == 0 else f'SNR {snr:.2f}'}",
                flush=True,
            )
        rows.append({
            "stem": stem,
            "source_bytes": path.stat().st_size,
            "ladder": song_rows,
        })
        write_report(rows)
    print("refine done", flush=True)


if __name__ == "__main__":
    main()
