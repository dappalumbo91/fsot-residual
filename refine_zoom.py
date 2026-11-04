#!/usr/bin/env python3
"""Sixth audio pass. Declared before this pass is scored.

Failure traced on the fifth pass:
The coefficient step was Q * 128 on every bin. That 128 is the inverse
scale of one cosine bin: the inverse multiplies by 2/N, N = 256, so one
count on one bin is 128 time counts. A sample is not one bin. It is the
sum of N bins, and the hop is half the window, so the sample sits in two
frames. The per-bin errors add. The time error is then far past Q/2, which
is the bound a step of Q was supposed to keep. Measured on the fifth pass,
Q=512 landed at 15-21 dB instead of holding that bound, and the bins under
the single-bin step were deleted. Those bins were part of the note.

The wrong term is the zoom. One zoom was used for every frame.

This pass, fixed here:
- The floor is the existing ladder step. A sample of the channel being
  quantized is inside the floor when its absolute error is at most Q/2.
- Two frames cover each sample, so each frame is allowed half of that:
  Q/4. The 2 is the overlap count, hop = window/2. It is not a new gain.
- The zoom steps are the powers of two inside the transform length:
  step_b = (Q * 128) / 2^b for b = 0, 1, ..., 8.
  b = 0 is the fifth pass. b = 8 divides by N and leaves step Q/2.
- D_eff is not this divisor. The engine nest runs from 5 to 25. That range
  cannot divide a length of 256. Using it here would name the wrong object.
- Each frame takes the smallest b whose uniform-zoom hop error is at most
  Q/4. Smallest b is the coarsest zoom that meets the floor. The decoder
  reads b. It does not search.
- The stored coefficients are the half-away indices at that frame's step.
- Q=1 still stores the samples and skips the cosine.
- Mid/side, window 512, hop 256, zlib level 9, and the Q ladder stay.
- Nothing above is moved after the scores.

The mixed-frame rebuild is the scored signal. Overlap means a frame chosen
from a uniform zoom can miss the floor once its neighbor uses another zoom.
That miss stays a miss. It is not patched with another divisor.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import fspack
import paths
import refine_fullband as band

HOP = band.HOP
GRAIN = band.GRAIN
B_MAX = 8  # 2^8 = 256 = N
SONGS = band.SONGS
STEMS = band.STEMS


def step_at(q: int, b: int) -> int:
    return max(1, (q * GRAIN) >> int(b))


def hop_max(error: np.ndarray, frames: int) -> np.ndarray:
    original = int(error.size)
    usable = min(frames, (original + HOP - 1) // HOP)
    out = np.zeros(frames, dtype=np.float64)
    if usable == 0:
        return out
    block = np.zeros(usable * HOP, dtype=np.float64)
    copied = min(original, block.size)
    block[:copied] = np.abs(error[:copied])
    out[:usable] = block.reshape(usable, HOP).max(axis=1)
    return out


def choose_zoom(samples: np.ndarray, coeffs: np.ndarray, q: int) -> np.ndarray:
    frames = int(coeffs.shape[0])
    original = int(samples.size)
    best = np.full(frames, B_MAX, dtype=np.int8)
    open_mask = np.ones(frames, dtype=bool)
    owned = np.arange(frames) * HOP < original
    best[~owned] = 0
    open_mask[~owned] = False
    limit = q / 4.0
    source = samples.astype(np.float64)
    for b in range(B_MAX + 1):
        if not np.any(open_mask):
            break
        step = step_at(q, b)
        qq = band.quantize(coeffs, step)
        recon = band.mdct_inverse(qq.astype(np.float64) * step, original)
        worst = hop_max(recon.astype(np.float64) - source, frames)
        take = open_mask & (worst <= limit)
        best[take] = b
        open_mask[take] = False
    return best


def apply_zoom(coeffs: np.ndarray, zoom: np.ndarray, q: int) -> np.ndarray:
    qq = np.zeros(coeffs.shape, dtype=np.int32)
    for b in range(B_MAX + 1):
        pick = zoom == b
        if not np.any(pick):
            continue
        qq[pick] = band.quantize(coeffs[pick], step_at(q, b))
    return qq


def restore_channel(qq: np.ndarray, zoom: np.ndarray, q: int, original: int) -> np.ndarray:
    hat = np.zeros(qq.shape, dtype=np.float64)
    for b in range(B_MAX + 1):
        pick = zoom == b
        if not np.any(pick):
            continue
        hat[pick] = qq[pick].astype(np.float64) * step_at(q, b)
    return band.mdct_inverse(hat, original)


def pack_zoom(q: int, original: int, mid_qq, mid_b, side_qq, side_b) -> bytes:
    mid = np.ascontiguousarray(mid_qq, dtype="<i4")
    side = np.ascontiguousarray(side_qq, dtype="<i4")
    blob = struct.pack("<III", q, original, int(mid.shape[0]))
    blob += np.ascontiguousarray(mid_b, dtype=np.uint8).tobytes()
    blob += np.ascontiguousarray(side_b, dtype=np.uint8).tobytes()
    blob += mid.tobytes()
    blob += side.tobytes()
    return zlib.compress(blob, 9)


def unpack_zoom(blob: bytes):
    raw = zlib.decompress(blob)
    q, original, frames = struct.unpack_from("<III", raw, 0)
    pos = 12
    mid_b = np.frombuffer(raw[pos:pos + frames], dtype=np.uint8).astype(np.int8).copy()
    pos += frames
    side_b = np.frombuffer(raw[pos:pos + frames], dtype=np.uint8).astype(np.int8).copy()
    pos += frames
    size = frames * HOP
    mid = np.frombuffer(raw[pos:pos + size * 4], dtype="<i4").astype(np.int32).copy().reshape(frames, HOP)
    pos += size * 4
    side = np.frombuffer(raw[pos:pos + size * 4], dtype="<i4").astype(np.int32).copy().reshape(frames, HOP)
    pos += size * 4
    if pos != len(raw):
        raise RuntimeError("trailing zoom bytes")
    return q, original, mid, mid_b, side, side_b


def self_check() -> None:
    band.self_check()
    zeros = np.zeros(4096, dtype=np.int32)
    zc, n = band.mdct_forward(zeros)
    if int(np.max(choose_zoom(zeros, zc, 512))) != 0:
        raise SystemExit("silence did not stay at the coarse zoom")
    tone = (np.sin(np.arange(8192) * 0.15) * 12000).astype(np.int32)
    noise = np.random.default_rng(3).integers(-12000, 12000, 8192).astype(np.int32)
    tone_b = choose_zoom(tone, band.mdct_forward(tone)[0], 2048)
    noise_b = choose_zoom(noise, band.mdct_forward(noise)[0], 32)
    print(
        f"zoom self-check tone median {float(np.median(tone_b)):.0f} "
        f"noise median {float(np.median(noise_b)):.0f}",
        flush=True,
    )
    if float(np.median(tone_b)) > float(np.median(noise_b)):
        raise SystemExit("a tone took a finer zoom than noise")
    qq = apply_zoom(zc, choose_zoom(zeros, zc, 128), 128)
    back = restore_channel(qq, np.zeros(qq.shape[0], dtype=np.int8), 128, n)
    if int(np.max(np.abs(back))) != 0:
        raise SystemExit("silence did not rebuild")
    raw = np.random.default_rng(5).integers(-4000, 4000, 3000).astype(np.int32)
    coeffs, original = band.mdct_forward(raw)
    zoom = choose_zoom(raw, coeffs, 128)
    qq = apply_zoom(coeffs, zoom, 128)
    blob = pack_zoom(128, original, qq, zoom, qq, zoom)
    q, got_n, got_qq, got_b, _s, _sb = unpack_zoom(blob)
    if q != 128 or got_n != original or not np.array_equal(got_qq, qq) or not np.array_equal(got_b, zoom):
        raise SystemExit("zoom archive does not rebuild")
    print("zoom self-check ok", flush=True)


def opus_and_r5() -> tuple[dict[str, int], dict[str, dict[int, int]]]:
    opus_rows = json.loads((paths.SCORES / "codec_sizes.json").read_text(encoding="utf-8"))["rows"]
    opus = {row["stem"]: int(row["opus96"]) for row in opus_rows}
    r5 = json.loads((paths.SCORES / "fullmdct_report.json").read_text(encoding="utf-8"))
    previous = {}
    for song in r5["songs"]:
        previous[song["stem"]] = {int(row["q"]): int(row["archive_bytes"]) for row in song["ladder"]}
    return opus, previous


def floor_fraction(error: np.ndarray, q: int) -> float:
    return float(np.mean(np.abs(error) <= (q / 2.0)))


def encode_channels(mid, side, q: int):
    mid_c, original = band.mdct_forward(mid)
    side_c, original_s = band.mdct_forward(side)
    if original != original_s:
        raise SystemExit("channel lengths differ")
    mid_b = choose_zoom(mid, mid_c, q)
    side_b = choose_zoom(side, side_c, q)
    mid_qq = apply_zoom(mid_c, mid_b, q)
    side_qq = apply_zoom(side_c, side_b, q)
    return original, mid_qq, mid_b, side_qq, side_b


def encode_one(path: Path, stem: str, out: Path, opus_bytes: int, previous: dict[int, int]) -> dict:
    print(f"song {stem}", flush=True)
    started = time.perf_counter()
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    rows = []
    exact = band.pack_exact(left, right)
    back = band.unpack(exact)
    if back[0] != 1 or not np.array_equal(back[1], left) or not np.array_equal(back[2], right):
        raise SystemExit(f"{stem} Q=1 is not sample-exact")
    rows.append({
        "q": 1,
        "archive_bytes": len(exact),
        "ratio_vs_opus96": len(exact) / opus_bytes,
        "ratio_vs_r5": len(exact) / previous[1],
        "floor_fraction_lr": 1.0,
        "mean_zoom": 0.0,
        "snr_db": None,
        "level_delta_db": 0.0,
        "exact": True,
        "wav": None,
        "kept_for_listening": False,
    })
    print(f"  Q=1 archive {len(exact)}", flush=True)
    del exact
    keeper_audio = None
    for q in deliver.AUDIO_Q:
        if q <= 1:
            continue
        original, mid_qq, mid_b, side_qq, side_b = encode_channels(mid, side, q)
        blob = pack_zoom(q, original, mid_qq, mid_b, side_qq, side_b)
        unpacked = unpack_zoom(blob)
        if unpacked[0] != q or not np.array_equal(unpacked[2], mid_qq) or not np.array_equal(unpacked[3], mid_b):
            raise SystemExit(f"{stem} Q={q} archive does not rebuild")
        got_mid = restore_channel(mid_qq, mid_b, q, original)
        got_side = restore_channel(side_qq, side_b, q, original)
        got_l, got_r = deliver.unmid_side(got_mid, got_side)
        play_l = deliver.clip_i16(got_l).astype(np.int32)
        play_r = deliver.clip_i16(got_r).astype(np.int32)
        err_l = play_l.astype(np.float64) - left
        err_r = play_r.astype(np.float64) - right
        noise = float(np.mean(np.concatenate([err_l, err_r]) ** 2))
        signal = float(np.mean(np.concatenate([left.astype(np.float64), right.astype(np.float64)]) ** 2))
        snr = math.inf if noise == 0 else 10 * math.log10(signal / noise)
        level = 0.0 if noise == 0 else 20 * math.log10(
            math.sqrt(float(np.mean(np.concatenate([play_l, play_r]).astype(np.float64) ** 2))) / math.sqrt(signal)
        )
        covered = float(np.mean(np.concatenate([
            np.abs(err_l) <= (q / 2.0),
            np.abs(err_r) <= (q / 2.0),
        ])))
        mean_b = float(np.mean(np.concatenate([mid_b.astype(np.float64), side_b.astype(np.float64)])))
        row = {
            "q": q,
            "archive_bytes": len(blob),
            "ratio_vs_opus96": len(blob) / opus_bytes,
            "ratio_vs_r5": len(blob) / previous[q],
            "floor_fraction_lr": covered,
            "mean_zoom": mean_b,
            "snr_db": None if math.isinf(snr) else snr,
            "level_delta_db": level,
            "exact": bool(noise == 0),
            "wav": None,
            "kept_for_listening": False,
        }
        rows.append(row)
        if keeper_audio is None or (covered, -len(blob)) > keeper_audio[0]:
            keeper_audio = ((covered, -len(blob)), q, play_l, play_r, blob)
        label = "exact" if noise == 0 else f"SNR {snr:.2f}"
        print(
            f"  Q={q} archive {len(blob)} vs opus {opus_bytes} "
            f"ratio {row['ratio_vs_opus96']:.3f} vs r5 {row['ratio_vs_r5']:.3f} "
            f"floor {covered:.3f} zoom {mean_b:.2f} level {level:.3f} dB {label}",
            flush=True,
        )
        del mid_qq, side_qq, mid_b, side_b, got_l, got_r
    keeper = max((row for row in rows if row["q"] > 1), key=lambda row: (row["floor_fraction_lr"], -row["archive_bytes"]))
    _rank, _q, play_l, play_r, blob = keeper_audio
    wav_path = out / f"{stem}_q{keeper['q']}.wav"
    deliver.write_wav(wav_path, play_l, play_r, rate)
    (out / f"{stem}_q{keeper['q']}.fspz").write_bytes(blob)
    keeper["wav"] = str(wav_path)
    keeper["kept_for_listening"] = True
    print(f"  keep Q={keeper['q']} {time.perf_counter() - started:.1f}s", flush=True)
    return {
        "stem": stem,
        "source": str(path),
        "source_bytes": path.stat().st_size,
        "opus96_bytes": opus_bytes,
        "seconds": left.size / rate,
        "ladder": rows,
    }


def main() -> None:
    self_check()
    out = paths.begin_pass("r6")
    opus, previous = opus_and_r5()
    songs = []
    for path, stem in zip(SONGS, STEMS):
        songs.append(encode_one(path, stem, out, opus[stem], previous[stem]))
        paths.save_score("zoom_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "failure": "one coefficient zoom Q*128 on every bin",
                "floor": "absolute channel error <= Q/2, each frame tested at Q/4",
                "zoom": "(Q*128) / 2^b, b = 0..8, smallest b that meets the frame test",
                "window": band.WINDOW,
                "hop": HOP,
                "q": list(deliver.AUDIO_Q),
                "entropy": "zlib-9",
            },
            "bar": "opus 96 kbps on the same decoded song",
            "songs": songs,
        })
    print("zoom done", flush=True)


if __name__ == "__main__":
    main()
