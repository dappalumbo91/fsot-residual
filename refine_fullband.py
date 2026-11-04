#!/usr/bin/env python3
"""Fifth audio pass. Declared before this pass is scored.

The cosine so far was applied only to the Haar approximation, the band
under about 2.8 kHz. The audible needle and the muffled hits live in the
bands that stayed in the time-sample step. The basis for this pass is the
cosine of the waveform itself.

Lengths and the coefficient unit are the ones already fixed:
- Window 512, hop 256, sine window. Overlap is half the window.
- Forward coefficients are the unscaled cosine sum.
- The inverse multiplies by 2/N with N = 256. One coefficient count is
  therefore 128 time counts. The coefficient step is Q * 128.
- Q=1 skips the transform and stores the samples. Integer cosine bins are
  not sample-exact, and this grade has to be.
- Mid/side, then the cosine on each channel. No Haar, no lag, no new
  multiplier, no masking threshold.
- The indices are zlib level 9, the same coder as the other long-form
  passes, so a size change is the basis.

The ladder is the existing one: 1, 32, 128, 512, 2048.
Nothing here is moved after the scores.
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
import refine_tonal as tonal

WINDOW = tonal.WINDOW
HOP = tonal.HOP
GRAIN = HOP // 2  # 2/N inverse, N = 256. Same unit as the fourth pass.
SONGS = [
    Path(r"G:\Music discography's\Sublime\At The House Of Blues\Sublime_At The House Of Blues__02-wrongway.mp3"),
    Path(r"G:\Music discography's\Skrillex\01 First of the Year (Equinox).mp3"),
    Path(r"G:\Music discography's\Bruno Mars - Discography\01 - Just The Way You Are.mp3"),
    Path(r"G:\Music discography's\eminem\Eminem - 8 Mile [2002]\01 - Lose Yourself - Eminem.mp3"),
    Path(r"G:\Music discography's\VNV Nation\VNV Nation - Beloved.mp3"),
    Path(r"G:\Music discography's\Clay Walker (Discography)\01- 'Fore She Was Mama.mp3"),
    Path(r"G:\Music discography's\Cage The Elephant\Cage the Elephant (2008)\03 Ain't No Rest For The Wicked.mp3"),
]
STEMS = [
    "sublime_wrong_way",
    "skrillex_first_of_the_year",
    "bruno_mars_just_the_way_you_are",
    "eminem_lose_yourself",
    "vnv_nation_beloved",
    "clay_walker_fore_she_was_mama",
    "cage_the_elephant_aint_no_rest",
]


def mdct_forward(samples: np.ndarray) -> tuple[np.ndarray, int]:
    original = int(samples.size)
    pad_end = (HOP - (original % HOP)) % HOP
    body = samples.astype(np.float64)
    if pad_end:
        body = np.pad(body, (0, pad_end), mode="edge")
    data = np.pad(body, (HOP, HOP), mode="constant")
    count = (data.size - WINDOW) // HOP + 1
    stride = data.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        data, shape=(count, WINDOW), strides=(HOP * stride, stride)
    )
    return (frames * tonal.WINDOW_FN) @ tonal.BASIS, original


def mdct_inverse(coeffs: np.ndarray, original: int) -> np.ndarray:
    factor = 2.0 / HOP
    time = (factor * (coeffs @ tonal.BASIS.T)) * tonal.WINDOW_FN
    frames = int(time.shape[0])
    # Hop is half the window, so each sample sits in two frames.
    acc = np.zeros((frames + 1) * HOP, dtype=np.float64)
    head = acc[0:frames * HOP].reshape(frames, HOP)
    tail = acc[HOP:HOP + frames * HOP].reshape(frames, HOP)
    head += time[:, :HOP]
    tail += time[:, HOP:]
    body = acc[HOP:acc.size - HOP]
    return np.rint(body[:original]).astype(np.int32)


def quantize(coeffs: np.ndarray, step: int) -> np.ndarray:
    half = step / 2.0
    qq = np.sign(coeffs) * np.floor((np.abs(coeffs) + half) / step)
    return qq.astype(np.int32)


def pack_exact(left: np.ndarray, right: np.ndarray) -> bytes:
    blob = struct.pack("<II", 1, int(left.size))
    blob += np.ascontiguousarray(left, dtype="<i2").tobytes()
    blob += np.ascontiguousarray(right, dtype="<i2").tobytes()
    return zlib.compress(blob, 9)


def pack_coeffs(q: int, original: int, mid_qq: np.ndarray, side_qq: np.ndarray) -> bytes:
    mid = np.ascontiguousarray(mid_qq, dtype="<i4")
    side = np.ascontiguousarray(side_qq, dtype="<i4")
    blob = struct.pack("<III", q, original, int(mid.shape[0]))
    blob += mid.tobytes()
    blob += side.tobytes()
    return zlib.compress(blob, 9)


def unpack(blob: bytes):
    raw = zlib.decompress(blob)
    (q,) = struct.unpack_from("<I", raw, 0)
    if q <= 1:
        (original,) = struct.unpack_from("<I", raw, 4)
        pos = 8
        count = original
        left = np.frombuffer(raw[pos:pos + count * 2], dtype="<i2").astype(np.int32).copy()
        pos += count * 2
        right = np.frombuffer(raw[pos:pos + count * 2], dtype="<i2").astype(np.int32).copy()
        pos += count * 2
        if pos != len(raw):
            raise RuntimeError("trailing exact bytes")
        return q, left, right
    _q, original, frames = struct.unpack_from("<III", raw, 0)
    pos = 12
    size = frames * HOP
    mid = np.frombuffer(raw[pos:pos + size * 4], dtype="<i4").astype(np.int32).copy().reshape(frames, HOP)
    pos += size * 4
    side = np.frombuffer(raw[pos:pos + size * 4], dtype="<i4").astype(np.int32).copy().reshape(frames, HOP)
    pos += size * 4
    if pos != len(raw):
        raise RuntimeError("trailing cosine bytes")
    return q, original, mid, side


def restore_pair(mid_qq: np.ndarray, side_qq: np.ndarray, step: int, original: int):
    mid = mdct_inverse(mid_qq.astype(np.float64) * step, original)
    side = mdct_inverse(side_qq.astype(np.float64) * step, original)
    return deliver.unmid_side(mid, side)


def self_check() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(0, 1000, size=5000)
    mine, original = mdct_forward(source)
    theirs, original_b = tonal.mdct_forward(source)
    if original != original_b or mine.shape != theirs.shape:
        raise SystemExit("frame count drifted from the scored cosine")
    if float(np.max(np.abs(mine - theirs))) > 1e-6:
        raise SystemExit("vector cosine does not match the scored cosine")
    back = tonal.mdct_inverse(mine, original)
    got = mdct_inverse(mine, original)
    if not np.array_equal(got, back):
        raise SystemExit("inverse drifted from the scored cosine")
    tone = (np.sin(np.arange(4000) * 0.2) * 5000).astype(np.int32)
    left = tone
    right = (tone // 2).astype(np.int32)
    blob = pack_exact(left, right)
    q, got_l, got_r = unpack(blob)
    if q != 1 or not np.array_equal(got_l, left) or not np.array_equal(got_r, right):
        raise SystemExit("exact grade does not round-trip")
    print("fullband self-check ok", flush=True)


def opus_table() -> dict[str, int]:
    path = paths.SCORES / "codec_sizes.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return {row["stem"]: int(row["opus96"]) for row in rows}


def metrics(left, right, got_l, got_r) -> tuple[float, float, bool]:
    noise_v = np.concatenate([
        got_l.astype(np.float64) - left,
        got_r.astype(np.float64) - right,
    ])
    signal_v = np.concatenate([left.astype(np.float64), right.astype(np.float64)])
    noise = float(np.mean(noise_v ** 2))
    signal = float(np.mean(signal_v ** 2))
    if noise == 0:
        return math.inf, 0.0, True
    snr = 10 * math.log10(signal / noise)
    level = 20 * math.log10(math.sqrt(float(np.mean(np.concatenate([got_l, got_r]).astype(np.float64) ** 2))) / math.sqrt(signal))
    return snr, level, False


def encode_one(path: Path, stem: str, out: Path, opus_bytes: int) -> dict:
    print(f"song {stem}", flush=True)
    started = time.perf_counter()
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    print(f"  cosine {time.perf_counter() - started:.1f}s setup", flush=True)
    mid_c, original = mdct_forward(mid)
    side_c, original_s = mdct_forward(side)
    if original != original_s or original != int(left.size):
        raise SystemExit(f"{stem} length mismatch")
    print(f"  forward {time.perf_counter() - started:.1f}s", flush=True)
    rows = []
    exact = pack_exact(left, right)
    back_q, back_l, back_r = unpack(exact)
    if back_q != 1 or not np.array_equal(back_l, left) or not np.array_equal(back_r, right):
        raise SystemExit(f"{stem} Q=1 is not sample-exact")
    rows.append({
        "q": 1,
        "archive_bytes": len(exact),
        "ratio_vs_opus96": len(exact) / opus_bytes,
        "snr_db": None,
        "level_delta_db": 0.0,
        "exact": True,
        "nonzero_fraction": 1.0,
        "wav": None,
    })
    print(f"  Q=1 archive {len(exact)} opus96 {opus_bytes}", flush=True)
    del exact
    for q in deliver.AUDIO_Q:
        if q <= 1:
            continue
        step = q * GRAIN
        mid_qq = quantize(mid_c, step)
        side_qq = quantize(side_c, step)
        nonzero = float(np.count_nonzero(mid_qq) + np.count_nonzero(side_qq)) / float(mid_qq.size + side_qq.size)
        blob = pack_coeffs(q, original, mid_qq, side_qq)
        unpacked = unpack(blob)
        if unpacked[0] != q or unpacked[1] != original:
            raise SystemExit(f"{stem} Q={q} header drifted")
        if not np.array_equal(unpacked[2], mid_qq) or not np.array_equal(unpacked[3], side_qq):
            raise SystemExit(f"{stem} Q={q} archive does not rebuild")
        nbytes = len(blob)
        got_l, got_r = restore_pair(unpacked[2], unpacked[3], step, original)
        play_l = deliver.clip_i16(got_l).astype(np.int32)
        play_r = deliver.clip_i16(got_r).astype(np.int32)
        snr, level, is_exact = metrics(left, right, play_l, play_r)
        del got_l, got_r, play_l, play_r, mid_qq, side_qq, blob, unpacked
        row = {
            "q": q,
            "coefficient_step": step,
            "archive_bytes": nbytes,
            "ratio_vs_opus96": nbytes / opus_bytes,
            "snr_db": None if math.isinf(snr) else snr,
            "level_delta_db": level,
            "exact": is_exact,
            "nonzero_fraction": nonzero,
            "wav": None,
        }
        rows.append(row)
        label = "exact" if is_exact else f"SNR {snr:.2f}"
        print(
            f"  Q={q} archive {nbytes} vs opus {opus_bytes} "
            f"ratio {row['ratio_vs_opus96']:.3f} nonzero {nonzero:.3f} "
            f"level {level:.3f} dB {label}",
            flush=True,
        )
    under = [row for row in rows if row["q"] > 1 and row["archive_bytes"] < opus_bytes]
    # Finest step that is already under Opus. If none are, keep Q=2048 to hear the miss.
    keeper = min(under, key=lambda row: row["q"]) if under else max(rows, key=lambda row: row["q"])
    if keeper["q"] > 1:
        step = keeper["q"] * GRAIN
        mid_qq = quantize(mid_c, step)
        side_qq = quantize(side_c, step)
        blob = pack_coeffs(keeper["q"], original, mid_qq, side_qq)
        got_l, got_r = restore_pair(mid_qq, side_qq, step, original)
        wav_path = out / f"{stem}_q{keeper['q']}.wav"
        deliver.write_wav(wav_path, deliver.clip_i16(got_l), deliver.clip_i16(got_r), rate)
        archive = out / f"{stem}_q{keeper['q']}.fspz"
        archive.write_bytes(blob)
        keeper["wav"] = str(wav_path)
        keeper["kept_for_listening"] = True
        del mid_qq, side_qq, blob, got_l, got_r
    for row in rows:
        row["kept_for_listening"] = bool(row.get("kept_for_listening", False))
    print(f"  keep Q={keeper['q']} {time.perf_counter() - started:.1f}s", flush=True)
    return {
        "stem": stem,
        "source": str(path),
        "source_bytes": path.stat().st_size,
        "opus96_bytes": opus_bytes,
        "seconds": original / rate,
        "ladder": rows,
    }


def main() -> None:
    self_check()
    out = paths.begin_pass("r5")
    opus = opus_table()
    songs = []
    for path, stem in zip(SONGS, STEMS):
        songs.append(encode_one(path, stem, out, opus[stem]))
        paths.save_score("fullmdct_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "basis": "full-band MDCT on mid and side",
                "window": WINDOW,
                "hop": HOP,
                "coefficient_step": "Q * 128",
                "grain": GRAIN,
                "q": list(deliver.AUDIO_Q),
                "entropy": "zlib-9",
                "q1": "raw int16 samples, transform bypassed",
            },
            "bar": "opus 96 kbps on the same decoded song",
            "songs": songs,
        })
    print("fullband done", flush=True)


if __name__ == "__main__":
    main()
