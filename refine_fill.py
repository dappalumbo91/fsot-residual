#!/usr/bin/env python3
"""Fifteenth audio pass. Declared before this pass is scored.

The omitted bands were stored as zero. Zero is the band left out, not a
compression of it. On the current playback the 5.5-11 kHz band was 67%
of VNV's error because of that zero. The coarsest rung already in the
ladder is b = 0, step (Q * 128). A coefficient inside that step rounds
to zero. A coefficient outside it is stored.

This pass quantizes every band that is stored as all zeros at that rung,
from the original coefficients. Bands that already carry quotients stay
as they are. Q stays 2048. The context bucket is not retuned. Sublime,
Clay, Eminem, and Bruno are not rebuilt. Skrillex and Cage start from
the twelfth pass. VNV starts from the fourteenth.
"""

from __future__ import annotations

import json
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import deliver
import fspack
import paths
import refine_clear as clear
import refine_context as context
import refine_fullband as band
import refine_gauge as gauge
import refine_miss as miss
import refine_rung as rung
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPF"
EDGES = (0, 32, 64, 128, 256)
SONGS = (
    ("skrillex_first_of_the_year", "r12", rung.unpack_song),
    ("cage_the_elephant_aint_no_rest", "r12", rung.unpack_song),
    ("vnv_nation_beloved", "r14", clear.unpack_song),
)


def fill_empty(coeffs: np.ndarray, qq: np.ndarray, b_band: np.ndarray):
    if coeffs.shape != qq.shape:
        raise SystemExit("coefficient frame count does not match the archive")
    out_q = np.array(qq, copy=True)
    out_b = np.array(b_band, dtype=np.int16, copy=True)
    step = zoom.step_at(Q, 0)
    filled = []
    for bi, (lo, hi) in enumerate(zip(EDGES, EDGES[1:])):
        if np.any(out_q[:, lo:hi]):
            continue
        out_q[:, lo:hi] = band.quantize(coeffs[:, lo:hi], step)
        out_b[:, bi] = 0
        filled.append(bi)
    return out_q, out_b, filled


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
        raise SystemExit("not a filled-band archive")
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
        raise SystemExit("trailing filled-band bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    gauge.self_check()
    rng = np.random.default_rng(15)
    coeffs = rng.normal(0, 50000, (5, 256))
    qq = np.zeros((5, 256), dtype=np.int32)
    qq[:, :64] = rng.integers(-2, 3, (5, 64), dtype=np.int32)
    kept = qq[:, :64].copy()
    b_band = np.zeros((5, 4), dtype=np.int16)
    b_band[:, 0] = 4
    b_band[:, 1] = 2
    out_q, out_b, filled = fill_empty(coeffs, qq, b_band)
    if filled != [2, 3]:
        raise SystemExit("the empty bands were not the ones above fs/8")
    if not np.array_equal(out_q[:, :64], kept) or not np.array_equal(out_b[:, :2], b_band[:, :2]):
        raise SystemExit("a stored band was recompressed")
    expect = band.quantize(coeffs[:, 64:128], zoom.step_at(Q, 0))
    if not np.array_equal(out_q[:, 64:128], expect) or np.any(out_b[:, 2:] != 0):
        raise SystemExit("the empty band is not the coarsest rung")
    blob = pack_song(Q, 5 * zoom.HOP, out_q, out_b, out_q, out_b, 44100)
    got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, 44100)
    if got_q != Q or got_n != 5 * zoom.HOP:
        raise SystemExit("header did not round-trip")
    if not np.array_equal(got_mid, out_q) or not np.array_equal(got_side, out_q):
        raise SystemExit("filled coefficients did not round-trip")
    if not np.array_equal(got_mb, out_b) or not np.array_equal(got_sb, out_b):
        raise SystemExit("filled zoom did not round-trip")
    print("fill self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("r15")
    opus, _previous = zoom.opus_and_r5()
    songs = []
    for stem, folder, unpack in SONGS:
        print(f"song {stem}", flush=True)
        path = band.SONGS[band.STEMS.index(stem)]
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        _q, original, qm, bm, qs, bs = unpack((paths.CURRENT / folder / f"{stem}_q{Q}.fspz").read_bytes(), rate)
        if _q != Q or original != int(left.size):
            raise SystemExit(f"{stem} source archive does not match the song")
        cm, n_mid = band.mdct_forward(mid)
        cs, n_side = band.mdct_forward(side)
        if n_mid != original or n_side != original or cm.shape != qm.shape:
            raise SystemExit(f"{stem} cosine frame count drifted")
        source_bytes = (paths.CURRENT / folder / f"{stem}_q{Q}.fspz").stat().st_size
        mid_qq, mid_b, mid_filled = fill_empty(cm, qm, bm)
        side_qq, side_b, side_filled = fill_empty(cs, qs, bs)
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
        snr = 10 * __import__("math").log10(signal / noise)
        hop_m = miss.mean.hop_mean(back_m.astype(np.float64) - mid, mid_qq.shape[0])
        hop_s = miss.mean.hop_mean(back_s.astype(np.float64) - side, side_qq.shape[0])
        wav_path = out / f"{stem}_q{Q}.wav"
        deliver.write_wav(wav_path, play_l, play_r, rate)
        (out / f"{stem}_q{Q}.fspz").write_bytes(blob)
        row = {
            "q": Q,
            "archive_bytes": len(blob),
            "source_pass": folder,
            "source_bytes": source_bytes,
            "opus96_bytes": opus[stem],
            "ratio_vs_opus96": len(blob) / opus[stem],
            "ratio_vs_source": len(blob) / source_bytes,
            "filled_bands_mid_side": [mid_filled, side_filled],
            "mean_abs_error": float(np.mean(np.abs(err))),
            "hop_mean_mid": float(hop_m[hop_m > 0].mean()),
            "hop_mean_side": float(hop_s[hop_s > 0].mean()),
            "snr_db": snr,
            "wav": str(wav_path),
        }
        songs.append({"stem": stem, "source": str(path), "seconds": left.size / rate, "row": row})
        paths.save_score("fill_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "rule": "a band stored as all zeros is quantized at b=0",
                "kept": "bands that already have quotients stay",
                "songs": [item[0] for item in SONGS],
                "coder": "tenth-pass magnitude bucket, not retuned",
                "q": Q,
            },
            "songs": songs,
        })
        print(
            f"  archive {len(blob)} {folder} {source_bytes} opus {opus[stem]} "
            f"ratio {row['ratio_vs_opus96']:.3f} vs source {row['ratio_vs_source']:.3f} "
            f"filled {mid_filled}/{side_filled} "
            f"mean_abs {row['mean_abs_error']:.1f} "
            f"hop {row['hop_mean_mid']:.1f}/{row['hop_mean_side']:.1f} SNR {snr:.2f}",
            flush=True,
        )
    print("fill done", flush=True)


if __name__ == "__main__":
    main()
