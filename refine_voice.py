#!/usr/bin/env python3
"""Thirteenth audio pass. Declared before this pass is scored.

Skrillex and Cage the Elephant accepted the twelfth pass. VNV Nation did
not. The vocal goes dull as soon as the singing starts.

The twelfth pass lowered the 2.8-5.5 kHz band one rung on both the mid
and the side. A sung voice is the mid. The side is the width of the
arrangement. Skrillex and Cage could spare that width and that center.
VNV's center is the vocal.

The wrong term is the rung on the mid. This pass puts VNV's mid
2.8-5.5 kHz band back on the eleventh-pass quotients and zoom. The side
keeps the twelfth-pass rung. The approximation is unchanged. Bands above
fs/8 stay zero. Q stays 2048. The context bucket is not retuned.
Skrillex, Cage, Sublime, Clay, and Eminem are not rebuilt.
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
import refine_context as context
import refine_fullband as band
import refine_gauge as gauge
import refine_high as high
import refine_miss as miss
import refine_rung as rung
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPM"
STEM = "vnv_nation_beloved"
D2 = 1


def restore_mid_d2(coarse_q, coarse_b, fine_q, fine_b):
    edges = gauge.band_edges(coarse_q.shape[1], 44100, zoom.HOP)
    if edges != [0, 32, 64, 128, 256]:
        raise SystemExit("the fs/8 cut is not bin 64")
    if coarse_q.shape != fine_q.shape or coarse_b.shape != fine_b.shape:
        raise SystemExit("mid sources do not match")
    out_q = np.array(coarse_q, copy=True)
    out_b = np.array(coarse_b, dtype=np.int16, copy=True)
    lo, hi = edges[D2], edges[D2 + 1]
    out_q[:, lo:hi] = fine_q[:, lo:hi]
    out_b[:, D2] = fine_b[:, D2]
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
        raise SystemExit("not a mid-voice archive")
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
        raise SystemExit("trailing mid-voice bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    gauge.self_check()
    rng = np.random.default_rng(13)
    coarse = rng.integers(-4, 5, (8, 256), dtype=np.int32)
    fine = coarse.copy()
    fine[:, 32:64] = rng.integers(-6, 7, (8, 32), dtype=np.int32)
    coarse_b = np.ones((8, 4), dtype=np.int16)
    fine_b = coarse_b.copy()
    fine_b[:, D2] = 3
    coarse[:, 64:] = 0
    fine[:, 64:] = 0
    coarse_b[:, 2:] = 0
    fine_b[:, 2:] = 0
    mid_q, mid_b = restore_mid_d2(coarse, coarse_b, fine, fine_b)
    if not np.array_equal(mid_q[:, 32:64], fine[:, 32:64]) or not np.array_equal(mid_b[:, D2], fine_b[:, D2]):
        raise SystemExit("the mid presence band did not return to the finer rung")
    if not np.array_equal(mid_q[:, :32], coarse[:, :32]) or np.any(mid_q[:, 64:]):
        raise SystemExit("the approximation or the top end moved")
    blob = pack_song(Q, 8 * zoom.HOP, mid_q, mid_b, coarse, coarse_b, 44100)
    got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, 44100)
    if got_q != Q or got_n != 8 * zoom.HOP:
        raise SystemExit("header did not round-trip")
    if not np.array_equal(got_mid, mid_q) or not np.array_equal(got_side, coarse):
        raise SystemExit("mid-voice coefficients did not round-trip")
    if not np.array_equal(got_mb, mid_b) or not np.array_equal(got_sb, coarse_b):
        raise SystemExit("mid-voice zoom did not round-trip")
    print("voice self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("r13")
    opus, _previous = zoom.opus_and_r5()
    earlier = json.loads((paths.SCORES / "rung_report.json").read_text(encoding="utf-8"))
    r12_bytes = {song["stem"]: int(song["row"]["archive_bytes"]) for song in earlier["songs"]}
    path = band.SONGS[band.STEMS.index(STEM)]
    print(f"song {STEM}", flush=True)
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    _q11, n11, q11_m, b11_m, q11_s, b11_s = high.unpack_song(
        (paths.CURRENT / "r11" / f"{STEM}_q{Q}.fspz").read_bytes(), rate
    )
    _q12, n12, q12_m, b12_m, q12_s, b12_s = rung.unpack_song(
        (paths.CURRENT / "r12" / f"{STEM}_q{Q}.fspz").read_bytes(), rate
    )
    if _q11 != Q or _q12 != Q or n11 != int(left.size) or n12 != n11:
        raise SystemExit("eleventh and twelfth archives do not match the source")
    if not np.array_equal(q11_m[:, :32], q12_m[:, :32]) or not np.array_equal(q11_s[:, :32], q12_s[:, :32]):
        raise SystemExit("the approximation differs between the two source passes")
    if np.any(q12_m[:, 64:]) or np.any(q12_s[:, 64:]):
        raise SystemExit("the twelfth pass is not silent above fs/8")
    mid_qq, mid_b = restore_mid_d2(q12_m, b12_m, q11_m, b11_m)
    side_qq, side_b = q12_s, b12_s
    original = n12
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
        raise SystemExit("archive does not rebuild")
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
    snr = 10 * __import__("math").log10(signal / noise)
    hop_m = miss.mean.hop_mean(back_m.astype(np.float64) - mid, mid_qq.shape[0])
    hop_s = miss.mean.hop_mean(back_s.astype(np.float64) - side, side_qq.shape[0])
    wav_path = out / f"{STEM}_q{Q}.wav"
    deliver.write_wav(wav_path, play_l, play_r, rate)
    (out / f"{STEM}_q{Q}.fspz").write_bytes(blob)
    row = {
        "q": Q,
        "archive_bytes": len(blob),
        "r12_bytes": r12_bytes[STEM],
        "opus96_bytes": opus[STEM],
        "ratio_vs_opus96": len(blob) / opus[STEM],
        "ratio_vs_r12": len(blob) / r12_bytes[STEM],
        "mean_abs_error": float(np.mean(np.abs(err))),
        "hop_mean_mid": float(hop_m[hop_m > 0].mean()),
        "hop_mean_side": float(hop_s[hop_s > 0].mean()),
        "mid_d2_zoom": float(np.mean(mid_b[:, D2])),
        "side_d2_zoom": float(np.mean(side_b[:, D2])),
        "snr_db": snr,
        "wav": str(wav_path),
    }
    paths.save_score("voice_report.json", {
        "pin": fspack.PIN,
        "declared": {
            "song": STEM,
            "keeps": "r12 for skrillex and cage; r10 for sublime, clay, eminem, bruno",
            "change": "mid 2.8-5.5 kHz returns to the eleventh-pass rung",
            "side": "twelfth-pass rung unchanged",
            "above_fs8": "remain zero",
            "coder": "tenth-pass magnitude bucket, not retuned",
            "q": Q,
        },
        "songs": [{"stem": STEM, "source": str(path), "seconds": left.size / rate, "row": row}],
    })
    print(
        f"  archive {len(blob)} r12 {r12_bytes[STEM]} opus {opus[STEM]} "
        f"ratio {row['ratio_vs_opus96']:.3f} vs r12 {row['ratio_vs_r12']:.3f} "
        f"mid d2 {row['mid_d2_zoom']:.2f} side d2 {row['side_d2_zoom']:.2f} "
        f"mean_abs {row['mean_abs_error']:.1f} "
        f"hop {row['hop_mean_mid']:.1f}/{row['hop_mean_side']:.1f} SNR {snr:.2f}",
        flush=True,
    )
    print("voice done", flush=True)


if __name__ == "__main__":
    main()
