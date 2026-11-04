#!/usr/bin/env python3
"""Nineteenth audio pass. Declared before this pass is scored.

The eighteenth pass raised the quiet octave one rung. The quietest 15
percent of hops is still behind Opus 96 on that same octave:

  Sublime  0-2.8 kHz   14.5 dB against Opus 17.4
  Clay     2.8-5.5 kHz  8.2 dB against Opus 12.4
  Eminem   0-2.8 kHz   15.3 dB against Opus 21.0
  Bruno    2.8-5.5 kHz  6.1 dB against Opus 10.1

The wrong term is still that gauge. This pass raises the same octave,
both channels, by one more existing rung. Quotients are retaken from
the original coefficients. Every other band stays. The band is not
rechosen. Q stays 2048. The bucket is not retuned. Skrillex, Cage,
and VNV are not rebuilt.

Bruno is already over Opus at 2,791,347 bytes against 2,741,169.
Another rung makes that size miss larger. It is not pulled back.
"""

from __future__ import annotations

import math
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import deliver
import fspack
import paths
import refine_context as context
import refine_dial as dial
import refine_fullband as band
import refine_gauge as gauge
import refine_miss as miss
import refine_quiet as quiet
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPN"
EDGES = (0, 32, 64, 128, 256)
HELD = quiet.HELD


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
        raise SystemExit("not a second quiet-rung archive")
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
        raise SystemExit("trailing second quiet-rung bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    gauge.self_check()
    rng = np.random.default_rng(19)
    coeffs = rng.normal(0, 80000, (4, 256))
    first = np.zeros((4, 4), dtype=np.int16)
    first[:, 0] = np.array([3, 1, 8, 4], dtype=np.int16)
    qq = band.quantize(coeffs, zoom.step_at(Q, 3)).astype(np.int32)
    out_q, out_b = dial.raise_deficit(coeffs, qq, first, 0)
    if not np.array_equal(out_b[:, 0], np.array([4, 2, 8, 5], dtype=np.int16)):
        raise SystemExit("the second quiet rung did not move by one")
    if not np.array_equal(out_b[:, 1:], first[:, 1:]):
        raise SystemExit("an untouched band took a new zoom")
    expect = band.quantize(coeffs[0, :32], zoom.step_at(Q, 4))
    if not np.array_equal(out_q[0, :32], expect):
        raise SystemExit("the second rung was not retaken from the original coefficients")
    blob = pack_song(Q, 4 * zoom.HOP, out_q, out_b, qq, first, 44100)
    got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, 44100)
    if got_q != Q or got_n != 4 * zoom.HOP:
        raise SystemExit("header did not round-trip")
    if not np.array_equal(got_mid, out_q) or not np.array_equal(got_mb, out_b):
        raise SystemExit("second quiet rung did not round-trip")
    if not np.array_equal(got_side, qq) or not np.array_equal(got_sb, first):
        raise SystemExit("the untouched channel did not round-trip")
    print("needle self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("r19")
    opus, _previous = zoom.opus_and_r5()
    songs = []
    for stem, band_index in HELD:
        print(f"song {stem}", flush=True)
        path = band.SONGS[band.STEMS.index(stem)]
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        archive_path = paths.CURRENT / "r18" / f"{stem}_q{Q}.fspz"
        _q, original, qm, bm, qs, bs = quiet.unpack_song(archive_path.read_bytes(), rate)
        if _q != Q or original != int(left.size):
            raise SystemExit(f"{stem} source archive does not match the song")
        cm, n_mid = band.mdct_forward(mid)
        cs, n_side = band.mdct_forward(side)
        if n_mid != original or n_side != original or cm.shape != qm.shape:
            raise SystemExit(f"{stem} cosine frame count drifted")
        source_bytes = archive_path.stat().st_size
        mid_qq, mid_b = dial.raise_deficit(cm, qm, bm, band_index)
        side_qq, side_b = dial.raise_deficit(cs, qs, bs, band_index)
        untouched = [i for i in range(4) if i != band_index]
        if not np.array_equal(mid_b[:, untouched], bm[:, untouched]):
            raise SystemExit(f"{stem} mid zoom moved outside the quiet octave")
        if not np.array_equal(side_b[:, untouched], bs[:, untouched]):
            raise SystemExit(f"{stem} side zoom moved outside the quiet octave")
        lo, hi = EDGES[band_index], EDGES[band_index + 1]
        if np.any(mid_qq[:, :lo] != qm[:, :lo]) or np.any(mid_qq[:, hi:] != qm[:, hi:]):
            raise SystemExit(f"{stem} mid quotients moved outside the quiet octave")
        if np.any(side_qq[:, :lo] != qs[:, :lo]) or np.any(side_qq[:, hi:] != qs[:, hi:]):
            raise SystemExit(f"{stem} side quotients moved outside the quiet octave")
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
        snr = 10 * math.log10(signal / noise)
        quiet_db = quiet.quiet_snr(left, right, play_l, play_r, rate)
        wav_path = out / f"{stem}_q{Q}.wav"
        deliver.write_wav(wav_path, play_l, play_r, rate)
        (out / f"{stem}_q{Q}.fspz").write_bytes(blob)
        row = {
            "q": Q,
            "dial_band": band_index,
            "dial_name": quiet.NAMES[band_index],
            "archive_bytes": len(blob),
            "r18_bytes": source_bytes,
            "opus96_bytes": opus[stem],
            "ratio_vs_opus96": len(blob) / opus[stem],
            "ratio_vs_r18": len(blob) / source_bytes,
            "under_opus": len(blob) < opus[stem],
            "mid_zoom_before": float(bm[:, band_index].mean()),
            "mid_zoom_after": float(mid_b[:, band_index].mean()),
            "side_zoom_before": float(bs[:, band_index].mean()),
            "side_zoom_after": float(side_b[:, band_index].mean()),
            "mean_abs_error": float(np.mean(np.abs(err))),
            "snr_db": snr,
            "quiet15_snr_db": {name: value for name, value in zip(quiet.NAMES, quiet_db)},
            "wav": str(wav_path),
        }
        songs.append({"stem": stem, "source": str(path), "seconds": left.size / rate, "row": row})
        paths.save_score("needle_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "rule": "one more rung on the same quiet octave from the eighteenth pass",
                "bands": {item[0]: quiet.NAMES[item[1]] for item in HELD},
                "quiet_fraction": quiet.QUIET,
                "kept": "every other band stays",
                "songs": [item[0] for item in HELD],
                "coder": "tenth-pass magnitude bucket, not retuned",
                "q": Q,
            },
            "songs": songs,
        })
        quiet_text = " ".join(f"{name} {value:.1f}" for name, value in zip(quiet.NAMES, quiet_db))
        print(
            f"  archive {len(blob)} r18 {source_bytes} opus {opus[stem]} "
            f"ratio {row['ratio_vs_opus96']:.3f} vs r18 {row['ratio_vs_r18']:.3f} "
            f"band {quiet.NAMES[band_index]} zoom {row['mid_zoom_before']:.2f}->{row['mid_zoom_after']:.2f}"
            f"/{row['side_zoom_before']:.2f}->{row['side_zoom_after']:.2f} "
            f"mean_abs {row['mean_abs_error']:.1f} SNR {snr:.2f} quiet {quiet_text}",
            flush=True,
        )
    print("needle done", flush=True)


if __name__ == "__main__":
    main()
