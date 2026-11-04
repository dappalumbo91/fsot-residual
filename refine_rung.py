#!/usr/bin/env python3
"""Twelfth audio pass. Declared before this pass is scored.

The eleventh pass put VNV Nation, Cage the Elephant, and Skrillex under
Opus 96 by integrating out every band above fs/8. Sublime, Clay Walker,
Eminem, and Bruno Mars stay at the tenth-pass archives.

The residual that remains is the approximation and the 2.8-5.5 kHz band.
Those two bands are still on the ninth-pass zoom, the coarsest step that
held the old hop floor while the top octaves were still present. That
floor is not this archive's distortion. The top octaves are already gone.

The wrong term is the zoom on the higher of the two kept bands. This pass
lowers only that band, 2.8-5.5 kHz, by one rung of the existing ladder:
b becomes b-1 where b was at least 1, and the quotients are retaken from
the original coefficients at the new step. A frame already at b=0 stays
there. The approximation stays at the eleventh-pass quotients. Bands
above fs/8 stay zero. Q stays 2048. The context bucket is not retuned.
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
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPR"
MISSES = high.MISSES
D2 = 1


def coarsen_d2(coeffs: np.ndarray, qq: np.ndarray, b_band: np.ndarray, rate: int):
    edges = gauge.band_edges(qq.shape[1], rate, zoom.HOP)
    if edges != [0, 32, 64, 128, 256]:
        raise SystemExit("the fs/8 cut is not bin 64")
    if coeffs.shape != qq.shape:
        raise SystemExit("coefficient frame count does not match the archive")
    out_q = np.array(qq, copy=True)
    out_b = np.array(b_band, dtype=np.int16, copy=True)
    previous = out_b[:, D2].astype(np.int16, copy=False)
    nxt = previous.copy()
    movable = previous > 0
    nxt[movable] = previous[movable] - 1
    lo, hi = edges[D2], edges[D2 + 1]
    sl = coeffs[:, lo:hi]
    taken = np.zeros((coeffs.shape[0], hi - lo), dtype=np.int32)
    for b in range(zoom.B_MAX + 1):
        pick = nxt == b
        if not np.any(pick):
            continue
        taken[pick] = band.quantize(sl[pick], zoom.step_at(Q, b))
    out_q[:, lo:hi] = taken
    out_b[:, D2] = nxt
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
        raise SystemExit("not a rung archive")
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
        raise SystemExit("trailing rung bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    gauge.self_check()
    rate = 44100
    rng = np.random.default_rng(12)
    frames = 6
    coeffs = rng.normal(0, 20000, (frames, 256))
    b_band = np.zeros((frames, 4), dtype=np.int16)
    b_band[:, 0] = np.array([4, 4, 3, 2, 1, 0], dtype=np.int16)
    b_band[:, D2] = np.array([0, 1, 3, 4, 8, 2], dtype=np.int16)
    qq = np.zeros((frames, 256), dtype=np.int32)
    edges = [0, 32, 64, 128, 256]
    for bi, (lo, hi) in enumerate(zip(edges, edges[1:])):
        for b in range(zoom.B_MAX + 1):
            pick = b_band[:, bi] == b
            if not np.any(pick):
                continue
            qq[np.ix_(np.flatnonzero(pick), np.arange(lo, hi))] = band.quantize(
                coeffs[pick, lo:hi], zoom.step_at(Q, int(b))
            )
    qq[:, 64:] = 0
    b_band[:, 2:] = 0
    out_q, out_b = coarsen_d2(coeffs, qq, b_band, rate)
    expect = np.array([0, 0, 2, 3, 7, 1], dtype=np.int16)
    if not np.array_equal(out_b[:, D2], expect):
        raise SystemExit("the 2.8-5.5 kHz rung did not drop by one")
    if not np.array_equal(out_q[:, :32], qq[:, :32]) or not np.array_equal(out_b[:, 0], b_band[:, 0]):
        raise SystemExit("the approximation was moved")
    if np.any(out_q[:, 64:]) or np.any(out_b[:, 2:]):
        raise SystemExit("a band above fs/8 came back")
    fresh = band.quantize(coeffs[:, 32:64], zoom.step_at(Q, 2))
    if not np.array_equal(out_q[2, 32:64], fresh[2]):
        raise SystemExit("the new rung is not the original coefficient at the new step")
    blob = pack_song(Q, frames * zoom.HOP, out_q, out_b, out_q, out_b, rate)
    got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, rate)
    if got_q != Q or got_n != frames * zoom.HOP:
        raise SystemExit("header did not round-trip")
    if not np.array_equal(got_mid, out_q) or not np.array_equal(got_side, out_q):
        raise SystemExit("rung coefficients did not round-trip")
    if not np.array_equal(got_mb, out_b) or not np.array_equal(got_sb, out_b):
        raise SystemExit("rung zoom did not round-trip")
    print("rung self-check ok", flush=True)


def r11_bytes() -> dict[str, int]:
    report = json.loads((paths.SCORES / "high_report.json").read_text(encoding="utf-8"))
    return {song["stem"]: int(song["row"]["archive_bytes"]) for song in report["songs"]}


def main() -> None:
    self_check()
    out = paths.begin_pass("r12")
    opus, _previous = zoom.opus_and_r5()
    earlier = r11_bytes()
    songs = []
    wanted = set(MISSES)
    for path, stem in zip(zoom.SONGS, zoom.STEMS):
        if stem not in wanted:
            continue
        print(f"song {stem}", flush=True)
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        _q, original, mid_qq, mid_b, side_qq, side_b = high.unpack_song(
            (paths.CURRENT / "r11" / f"{stem}_q{Q}.fspz").read_bytes(), rate
        )
        if _q != Q or original != int(left.size):
            raise SystemExit(f"{stem} eleventh-pass archive does not match the source")
        mid_c, n_mid = band.mdct_forward(mid)
        side_c, n_side = band.mdct_forward(side)
        if n_mid != original or n_side != original or mid_c.shape != mid_qq.shape:
            raise SystemExit(f"{stem} cosine frame count drifted")
        before_m = float(np.mean(mid_b[:, D2]))
        before_s = float(np.mean(side_b[:, D2]))
        mid_qq, mid_b = coarsen_d2(mid_c, mid_qq, mid_b, rate)
        side_qq, side_b = coarsen_d2(side_c, side_qq, side_b, rate)
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
        snr = float("inf") if noise == 0 else 10 * __import__("math").log10(signal / noise)
        hop_m = miss.mean.hop_mean(back_m.astype(np.float64) - mid, mid_qq.shape[0])
        hop_s = miss.mean.hop_mean(back_s.astype(np.float64) - side, side_qq.shape[0])
        wav_path = out / f"{stem}_q{Q}.wav"
        deliver.write_wav(wav_path, play_l, play_r, rate)
        (out / f"{stem}_q{Q}.fspz").write_bytes(blob)
        row = {
            "q": Q,
            "archive_bytes": len(blob),
            "r11_bytes": earlier[stem],
            "opus96_bytes": opus[stem],
            "ratio_vs_opus96": len(blob) / opus[stem],
            "ratio_vs_r11": len(blob) / earlier[stem],
            "mean_abs_error": float(np.mean(np.abs(err))),
            "hop_mean_mid": float(hop_m[hop_m > 0].mean()),
            "hop_mean_side": float(hop_s[hop_s > 0].mean()),
            "d2_zoom_before_mid_side": [before_m, before_s],
            "d2_zoom_after_mid_side": [float(np.mean(mid_b[:, D2])), float(np.mean(side_b[:, D2]))],
            "snr_db": None if snr == float("inf") else snr,
            "wav": str(wav_path),
        }
        songs.append({"stem": stem, "source": str(path), "seconds": left.size / rate, "row": row})
        paths.save_score("rung_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "songs": list(MISSES),
                "keeps": "r10 archives for sublime, clay, eminem, bruno",
                "band": "2.8-5.5 kHz only, one existing ladder rung coarser",
                "approximation": "eleventh-pass quotients unchanged",
                "above_fs8": "remain zero",
                "coder": "tenth-pass magnitude bucket, not retuned",
                "q": Q,
            },
            "songs": songs,
        })
        print(
            f"  archive {len(blob)} r11 {earlier[stem]} opus {opus[stem]} "
            f"ratio {row['ratio_vs_opus96']:.3f} vs r11 {row['ratio_vs_r11']:.3f} "
            f"d2 {before_m:.2f}->{row['d2_zoom_after_mid_side'][0]:.2f} "
            f"{before_s:.2f}->{row['d2_zoom_after_mid_side'][1]:.2f} "
            f"mean_abs {row['mean_abs_error']:.1f} "
            f"hop {row['hop_mean_mid']:.1f}/{row['hop_mean_side']:.1f} SNR {snr:.2f}",
            flush=True,
        )
    print("rung done", flush=True)


if __name__ == "__main__":
    main()
