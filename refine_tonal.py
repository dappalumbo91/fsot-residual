#!/usr/bin/env python3
"""Third audio pass. Declared before this pass is scored.

The second pass kept the coarse step off the 2.8-5.5 kHz band. That cut the
error by about 1 dB and raised the archive about 20 percent. One lag on the
low band did not pay for it. A chord is not one repeat. The low band is sparse
in a cosine basis and dense as time samples.

This pass keeps the second pass's steps, (4, 1, 1, 1). The approximation is a
modified discrete cosine transform. Window 512, hop 256, sine window. 512 is
the low-cycle window from the second pass. The hop is half of that so the
windows overlap and cancel. Those lengths are not moved after the scores.

Forward coefficients are the unscaled cosine sum. The inverse multiplies by
2/N, with N = 256, which is the factor that rebuilds the samples. Coefficients
then take the same half-away step Q as the approximation. Q=1 skips the
transform and stores the samples, so the exact grade stays sample-exact.
Detail bands stay memoryless. Mid/side, 3-level integer Haar, and zlib -9 stay.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import fspack
import paths
import refine_audio as r2

WINDOW = 512
HOP = 256
OUT = deliver.WATCH / "refine"


def _basis(half: int) -> np.ndarray:
    n = np.arange(half * 2)
    k = np.arange(half)
    phase = (n[:, None] + 0.5 + half / 2.0) * (k[None, :] + 0.5)
    return np.cos(np.pi / half * phase)


BASIS = _basis(HOP)
WINDOW_FN = np.sin(np.pi / WINDOW * (np.arange(WINDOW) + 0.5))


def mdct_forward(samples: np.ndarray) -> tuple[np.ndarray, int]:
    original = int(samples.size)
    pad_end = (HOP - (original % HOP)) % HOP
    body = samples.astype(np.float64)
    if pad_end:
        body = np.pad(body, (0, pad_end), mode="edge")
    data = np.pad(body, (HOP, HOP), mode="constant")
    starts = range(0, data.size - WINDOW + 1, HOP)
    frames = np.stack([data[start:start + WINDOW] for start in starts])
    return (frames * WINDOW_FN) @ BASIS, original


def mdct_inverse(coeffs: np.ndarray, original: int) -> np.ndarray:
    factor = 2.0 / HOP
    time = (factor * (coeffs @ BASIS.T)) * WINDOW_FN
    frames = time.shape[0]
    acc = np.zeros((frames - 1) * HOP + WINDOW, dtype=np.float64)
    for index in range(frames):
        start = index * HOP
        acc[start:start + WINDOW] += time[index]
    body = acc[HOP:acc.size - HOP]
    return np.rint(body[:original]).astype(np.int32)


def quantize_mdct(samples: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    coeffs, original = mdct_forward(samples)
    if step <= 1:
        qq = np.rint(coeffs).astype(np.int32)
        hat = qq.astype(np.float64)
    else:
        half = step / 2.0
        qq = np.sign(coeffs) * np.floor((np.abs(coeffs) + half) / step)
        qq = qq.astype(np.int32)
        hat = qq.astype(np.float64) * step
    recon = mdct_inverse(hat, original)
    return qq, recon


def pack(mid_details, mid_qq, side_details, side_qq, q: int) -> bytes:
    blob = bytearray()
    blob += struct.pack("<I", q)
    groups = (list(mid_details) + [mid_qq], list(side_details) + [side_qq])
    for group in groups:
        blob += struct.pack("<B", len(group))
        for plane in group:
            data = np.ascontiguousarray(plane, dtype="<i4")
            blob += struct.pack("<II", int(data.shape[0]), int(data.shape[1]) if data.ndim == 2 else 0)
            blob += data.tobytes()
    return zlib.compress(bytes(blob), 9)


def unpack(blob: bytes):
    raw = zlib.decompress(blob)
    (q,) = struct.unpack_from("<I", raw, 0)
    pos = 4
    groups = []
    for _channel in range(2):
        (count,) = struct.unpack_from("<B", raw, pos)
        pos += 1
        planes = []
        for _ in range(count):
            rows, cols = struct.unpack_from("<II", raw, pos)
            pos += 8
            size = rows if cols == 0 else rows * cols
            plane = np.frombuffer(raw[pos:pos + size * 4], dtype="<i4").astype(np.int32).copy()
            pos += size * 4
            if cols:
                plane = plane.reshape(rows, cols)
            planes.append(plane)
        groups.append(planes)
    if pos != len(raw):
        raise RuntimeError("trailing refine bytes")
    return q, groups[0][:3], groups[0][3], groups[1][:3], groups[1][3]


def self_check() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(0, 1000, size=5000)
    coeffs, original = mdct_forward(source)
    # Float rebuild, before the integer round.
    factor = 2.0 / HOP
    time = (factor * (coeffs @ BASIS.T)) * WINDOW_FN
    frames = time.shape[0]
    acc = np.zeros((frames - 1) * HOP + WINDOW, dtype=np.float64)
    for index in range(frames):
        acc[index * HOP:index * HOP + WINDOW] += time[index]
    got = acc[HOP:HOP + original]
    peak = float(np.max(np.abs(got - source)))
    if peak > 1e-6:
        raise SystemExit(f"MDCT does not rebuild, peak error {peak}")
    tone = (np.sin(np.arange(4000) * 0.2) * 5000).astype(np.int32)
    qq, recon = quantize_mdct(tone, 1)
    # Step 1 rounds cosine bins to integers, so it is not sample-exact.
    # The song path bypasses this at Q=1. Here, the archive must rebuild recon.
    back = mdct_inverse(qq.astype(np.float64), tone.size)
    if not np.array_equal(back, recon):
        raise SystemExit("MDCT archive does not rebuild its own coefficients")
    print(f"mdct self-check ok peak {peak:.3e}", flush=True)


def encode_song(path: Path) -> dict:
    stem = path.stem
    print(f"song {stem}", flush=True)
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    mid_bands, original = deliver.haar_split(mid, 3)
    side_bands, original_s = deliver.haar_split(side, 3)
    rows = []
    for q in deliver.AUDIO_Q:
        print(f"  Q={q}", flush=True)
        mid_codes, _mid_hats = r2.quantize_plain(mid_bands[:3], q)
        side_codes, _side_hats = r2.quantize_plain(side_bands[:3], q)
        if q <= 1:
            mid_qq = mid_bands[3].astype(np.int32, copy=False)
            side_qq = side_bands[3].astype(np.int32, copy=False)
            mid_approx = mid_qq
            side_approx = side_qq
        else:
            mid_qq, mid_approx = quantize_mdct(mid_bands[3], q)
            side_qq, side_approx = quantize_mdct(side_bands[3], q)
        blob = pack(mid_codes, mid_qq, side_codes, side_qq, q)
        back_q, back_md, back_ma, back_sd, back_sa = unpack(blob)
        if back_q != q:
            raise SystemExit("archive lost Q")
        step_approx = 1 if q <= 1 else q
        if q <= 1:
            mid_back = back_ma
            side_back = back_sa
        else:
            mid_back = mdct_inverse(back_ma.astype(np.float64) * step_approx, original_band_len(mid_bands[3]))
            side_back = mdct_inverse(back_sa.astype(np.float64) * step_approx, original_band_len(side_bands[3]))
            if not np.array_equal(mid_back, mid_approx) or not np.array_equal(side_back, side_approx):
                raise SystemExit(f"{stem} Q={q} approximation did not rebuild")
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
            deliver.haar_join(side_hats, original_s),
        )
        got_l = deliver.clip_i16(got_l).astype(np.int32)
        got_r = deliver.clip_i16(got_r).astype(np.int32)
        if q == 1 and not (np.array_equal(got_l, left) and np.array_equal(got_r, right)):
            raise SystemExit(f"{stem} Q=1 is not sample-exact")
        err_l = got_l.astype(np.float64) - left
        err_r = got_r.astype(np.float64) - right
        err = np.concatenate([err_l, err_r])
        sig = np.concatenate([left.astype(np.float64), right.astype(np.float64)])
        noise = float(np.mean(err * err))
        signal = float(np.mean(sig * sig))
        snr = math.inf if noise == 0 else 10 * math.log10(signal / max(noise, 1e-30))
        level = 0.0 if noise == 0 else 20 * math.log10(math.sqrt(float(np.mean(np.concatenate([got_l, got_r]).astype(np.float64) ** 2))) / math.sqrt(signal))
        e28 = r2.band_energy(err_l, rate, 2000, 8000) + r2.band_energy(err_r, rate, 2000, 8000)
        eall = float(np.dot(err_l, err_l) + np.dot(err_r, err_r))
        wav_path = None
        if q == paths.LISTEN_Q:
            wav_path = OUT / f"{stem}_q{q}.wav"
            deliver.write_wav(wav_path, got_l, got_r, rate)
            (OUT / f"{stem}_q{q}.fspz").write_bytes(blob)
        row = {
            "q": q,
            "archive_bytes": len(blob),
            "r2_archive_bytes": None,
            "first_archive_bytes": None,
            "snr_db": None if math.isinf(snr) else snr,
            "level_delta_db": level,
            "error_2_8kHz_share": 0.0 if eall == 0 else e28 / eall,
            "max_abs": int(np.max(np.abs(err))),
            "exact": bool(noise == 0),
            "wav": None if wav_path is None else str(wav_path),
        }
        rows.append(row)
        print(
            f"    archive {row['archive_bytes']} r2 {row['r2_archive_bytes']} "
            f"first {row['first_archive_bytes']} level {level:.3f} dB "
            f"share_2_8 {row['error_2_8kHz_share']:.3f} "
            f"{'exact' if noise == 0 else f'SNR {snr:.2f}'}",
            flush=True,
        )
    return {"stem": stem, "source_bytes": path.stat().st_size, "ladder": rows}


def original_band_len(band: np.ndarray) -> int:
    return int(band.size)


def main() -> None:
    global OUT
    OUT = paths.begin_pass("r3")
    self_check()
    songs = [encode_song(path) for path in deliver.SONGS]
    report = {
        "pin": fspack.PIN,
        "declared": {
            "steps": list(r2.REFINE_TIMES),
            "window": WINDOW,
            "hop": HOP,
            "approx_step": "Q on MDCT coefficients, bypass at Q=1",
            "q": list(deliver.AUDIO_Q),
            "listen_q": paths.LISTEN_Q,
        },
        "songs": songs,
    }
    path = paths.save_score("tonal_report.json", report)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
