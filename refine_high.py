#!/usr/bin/env python3
"""Eleventh audio pass. Declared before this pass is scored.

The tenth pass left three misses at Q=2048: VNV Nation, Cage the
Elephant, and Skrillex. Sublime, Clay Walker, Eminem, and Bruno Mars
stay at the tenth-pass archives. Their samples are not rebuilt.

An odd frame is not a copy of the even frame before it. On these three
songs that copy stays inside the floor on under three percent of the
odd frames, and the approximation alone fails the same way. The hop
mean already sits near 480 against Q/4 = 512. Zeroing the +/-1 quotients
that slack can hold is worth under 300KB of ideal rate. The coder is
not the remaining term.

The wrong term is the residual above the fs/8 cut. The needle the ear
named sits in the 2-8 kHz region. The octave edges already in the coder
are fs/16, fs/8, and fs/4. fs/8 is the edge at the bottom of that
region. This pass integrates out every band above it: 5.5-11 kHz and
11-22 kHz. The approximation and the 2.8-5.5 kHz band keep the ninth-pass
quotients and the ninth-pass zoom. Q stays 2048. The context bucket is
the tenth pass, unchanged. One cut for all three songs.
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
import refine_miss as miss
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPI"
MISSES = (
    "vnv_nation_beloved",
    "cage_the_elephant_aint_no_rest",
    "skrillex_first_of_the_year",
)


def integrate(qq: np.ndarray, b_band: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    edges = gauge.band_edges(qq.shape[1], rate, zoom.HOP)
    if edges != [0, 32, 64, 128, 256]:
        raise SystemExit("the fs/8 cut is not bin 64")
    out_q = np.array(qq, copy=True)
    out_b = np.array(b_band, copy=True)
    out_q[:, edges[2]:] = 0
    out_b[:, 2:] = 0
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
        raise SystemExit("not an integrated-octave archive")
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
    side_b = np.frombuffer(zoom_raw[frames * 4:], dtype=np.uint8).astype(np.int16).copy().reshape(frames, 4)
    planes = []
    for _channel in range(2):
        cols = []
        for band_i in range(4):
            width = edges[band_i + 1] - edges[band_i]
            values, offset = context.unpack_band(blob, offset, frames, width)
            cols.append(values)
        planes.append(np.column_stack(cols))
    if offset != len(blob):
        raise SystemExit("trailing integrated-octave bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    gauge.self_check()
    rate = 44100
    rng = np.random.default_rng(11)
    raw = rng.integers(-6, 8, (18, 256), dtype=np.int32)
    zoom_b = rng.integers(0, 5, (18, 4), dtype=np.int16)
    qq, b_band = integrate(raw, zoom_b, rate)
    if np.any(qq[:, 64:]) or np.any(b_band[:, 2:]):
        raise SystemExit("bins above fs/8 were not integrated out")
    if not np.array_equal(qq[:, :64], raw[:, :64]) or not np.array_equal(b_band[:, :2], zoom_b[:, :2]):
        raise SystemExit("the low bands were altered")
    blob = pack_song(Q, 18 * zoom.HOP, qq, b_band, qq, b_band, rate)
    got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, rate)
    if got_q != Q or got_n != 18 * zoom.HOP:
        raise SystemExit("header did not round-trip")
    if not np.array_equal(got_mid, qq) or not np.array_equal(got_side, qq):
        raise SystemExit("integrated coefficients did not round-trip")
    if not np.array_equal(got_mb, b_band) or not np.array_equal(got_sb, b_band):
        raise SystemExit("integrated zoom did not round-trip")
    print("high self-check ok", flush=True)


def r10_bytes() -> dict[str, int]:
    report = json.loads((paths.SCORES / "context_report.json").read_text(encoding="utf-8"))
    return {song["stem"]: int(song["row"]["archive_bytes"]) for song in report["songs"]}


def main() -> None:
    self_check()
    out = paths.begin_pass("r11")
    opus, _previous = zoom.opus_and_r5()
    earlier = r10_bytes()
    songs = []
    wanted = set(MISSES)
    for path, stem in zip(zoom.SONGS, zoom.STEMS):
        if stem not in wanted:
            continue
        print(f"song {stem}", flush=True)
        left, right, rate = deliver.decode_pcm(path)
        _q, original, mid_qq, mid_b, side_qq, side_b = miss.unpack_song(
            (paths.CURRENT / "r9" / f"{stem}_q{Q}.fspz").read_bytes(), rate
        )
        if _q != Q or original != int(left.size):
            raise SystemExit(f"{stem} ninth-pass archive does not match the source")
        mid_qq, mid_b = integrate(mid_qq, mid_b, rate)
        side_qq, side_b = integrate(side_qq, side_b, rate)
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
        mid, side = deliver.mid_side(left, right)
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
            "r10_bytes": earlier[stem],
            "opus96_bytes": opus[stem],
            "ratio_vs_opus96": len(blob) / opus[stem],
            "ratio_vs_r10": len(blob) / earlier[stem],
            "mean_abs_error": float(np.mean(np.abs(err))),
            "hop_mean_mid": float(hop_m[hop_m > 0].mean()),
            "hop_mean_side": float(hop_s[hop_s > 0].mean()),
            "previous_floor": Q / 4.0,
            "snr_db": None if snr == float("inf") else snr,
            "wav": str(wav_path),
        }
        songs.append({"stem": stem, "source": str(path), "seconds": left.size / rate, "row": row})
        paths.save_score("high_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "songs": list(MISSES),
                "keeps": "r10 archives for sublime, clay, eminem, bruno",
                "cut": "integrate out every band above fs/8",
                "kept_bands": "approximation and 2.8-5.5 kHz, ninth-pass quotients",
                "coder": "tenth-pass magnitude bucket, not retuned",
                "q": Q,
            },
            "songs": songs,
        })
        print(
            f"  archive {len(blob)} r10 {earlier[stem]} opus {opus[stem]} "
            f"ratio {row['ratio_vs_opus96']:.3f} vs r10 {row['ratio_vs_r10']:.3f} "
            f"mean_abs {row['mean_abs_error']:.1f} "
            f"hop {row['hop_mean_mid']:.1f}/{row['hop_mean_side']:.1f} SNR {snr:.2f}",
            flush=True,
        )
    print("high done", flush=True)


if __name__ == "__main__":
    main()
