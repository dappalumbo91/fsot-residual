#!/usr/bin/env python3
"""Twenty-first audio pass. Declared before this pass is scored.

The quiet octave already raised is no longer the largest miss.
Measured on the current playback, quietest 15 percent of hops,
ours against Opus 96:

  Sublime r19   2.8-5.5 kHz   4.7 against 7.6.   0-2.8 is already ahead.
  Clay r19      0-2.8 kHz    13.1 against 18.8.
  Eminem r20    2.8-5.5 kHz   4.7 against 11.1.
  Bruno r20     0-2.8 kHz    15.5 against 19.9.

The wrong term is the gauge on that octave inside the quiet hops.
A frame is raised one existing rung only when its window covers a
quiet hop. Loud frames stay. Every other band stays. Quotients for
the raised frames are retaken from the original coefficients. The
quiet fraction is not moved. Q stays 2048. The bucket is not retuned.
Skrillex, Cage, and VNV are not rebuilt.

A song that crosses Opus stays a size miss. The band is not swapped
after the archive size is known.
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
import refine_fullband as band
import refine_gauge as gauge
import refine_miss as miss
import refine_needle as needle
import refine_quiet as quiet
import refine_trim as trim
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPA"
EDGES = (0, 32, 64, 128, 256)
# Octave still furthest behind Opus in the quiet slice. Not rechosen from size.
GAPS = (
    ("sublime_wrong_way", 1, "r19", needle.unpack_song),
    ("clay_walker_fore_she_was_mama", 0, "r19", needle.unpack_song),
    ("eminem_lose_yourself", 1, "r20", trim.unpack_song),
    ("bruno_mars_just_the_way_you_are", 0, "r20", trim.unpack_song),
)


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
        raise SystemExit("not a gap archive")
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
        raise SystemExit("trailing gap bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def raise_quiet(coeffs, qq, b_band, band_index: int, keep: np.ndarray):
    if keep.shape != (qq.shape[0],):
        raise SystemExit("the quiet frame mask does not match")
    out_q = np.array(qq, copy=True)
    out_b = np.array(b_band, dtype=np.int16, copy=True)
    nxt = out_b[:, band_index].copy()
    movable = keep & (nxt < zoom.B_MAX)
    nxt[movable] = nxt[movable] + 1
    lo, hi = EDGES[band_index], EDGES[band_index + 1]
    sl = coeffs[:, lo:hi]
    for b in range(zoom.B_MAX + 1):
        pick = movable & (nxt == b)
        if not np.any(pick):
            continue
        out_q[pick, lo:hi] = band.quantize(sl[pick], zoom.step_at(Q, int(b)))
    out_b[:, band_index] = nxt
    return out_q, out_b


def self_check() -> None:
    gauge.self_check()
    rng = np.random.default_rng(21)
    coeffs = rng.normal(0, 80000, (4, 256))
    qq = band.quantize(coeffs, zoom.step_at(Q, 2)).astype(np.int32)
    b_band = np.full((4, 4), 2, dtype=np.int16)
    keep = np.array([True, False, True, False])
    out_q, out_b = raise_quiet(coeffs, qq, b_band, 1, keep)
    if not np.array_equal(out_b[:, 1], np.array([3, 2, 3, 2], dtype=np.int16)):
        raise SystemExit("the quiet frames did not take one rung")
    if not np.array_equal(out_b[:, [0, 2, 3]], b_band[:, [0, 2, 3]]):
        raise SystemExit("a band outside the gap moved")
    if np.any(out_q[1] != qq[1]) or np.any(out_q[3] != qq[3]):
        raise SystemExit("a loud frame was rewritten")
    expect = band.quantize(coeffs[0, 32:64], zoom.step_at(Q, 3))
    if not np.array_equal(out_q[0, 32:64], expect):
        raise SystemExit("the quiet frame was not retaken from the original coefficients")
    if np.any(out_q[:, :32] != qq[:, :32]) or np.any(out_q[:, 64:] != qq[:, 64:]):
        raise SystemExit("quotients outside the gap were rewritten")
    blob = pack_song(Q, 4 * zoom.HOP, out_q, out_b, out_q, out_b, 44100)
    got_q, got_n, got_mid, got_mb, _side, _sb = unpack_song(blob, 44100)
    if got_q != Q or got_n != 4 * zoom.HOP or not np.array_equal(got_mid, out_q) or not np.array_equal(got_mb, out_b):
        raise SystemExit("gap archive did not round-trip")
    print("gap self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("r21")
    opus_bytes, _previous = zoom.opus_and_r5()
    songs = []
    for stem, band_index, folder, unpack in GAPS:
        print(f"song {stem}", flush=True)
        path = band.SONGS[band.STEMS.index(stem)]
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        archive_path = paths.CURRENT / folder / f"{stem}_q{Q}.fspz"
        _q, original, qm, bm, qs, bs = unpack(archive_path.read_bytes(), rate)
        if _q != Q or original != int(left.size):
            raise SystemExit(f"{stem} source archive does not match the song")
        bm = quiet.as_bands(bm)
        bs = quiet.as_bands(bs)
        cm, n_mid = band.mdct_forward(mid)
        cs, n_side = band.mdct_forward(side)
        if n_mid != original or n_side != original or cm.shape != qm.shape:
            raise SystemExit(f"{stem} cosine frame count drifted")
        keep = trim.frames_covering(qm.shape[0], trim.quiet_hops(left, right))
        source_bytes = archive_path.stat().st_size
        mid_qq, mid_b = raise_quiet(cm, qm, bm, band_index, keep)
        side_qq, side_b = raise_quiet(cs, qs, bs, band_index, keep)
        lo, hi = EDGES[band_index], EDGES[band_index + 1]
        if np.any(mid_qq[~keep, lo:hi] != qm[~keep, lo:hi]):
            raise SystemExit(f"{stem} a loud frame moved")
        if np.any(mid_b[keep, band_index] < bm[keep, band_index]):
            raise SystemExit(f"{stem} a quiet frame lost a rung")
        untouched = [i for i in range(4) if i != band_index]
        if not np.array_equal(mid_b[:, untouched], bm[:, untouched]):
            raise SystemExit(f"{stem} a band outside the gap moved")
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
        ours_q = quiet.quiet_snr(left, right, play_l, play_r, rate)
        opus_path = paths.REFERENCE / "opus" / f"{stem}_opus96.opus"
        opus_l, opus_r, rate_o = deliver.decode_pcm(opus_path)
        n = min(left.size, opus_l.size)
        opus_q = quiet.quiet_snr(left[:n], right[:n], opus_l[:n], opus_r[:n], rate_o)
        wav_path = out / f"{stem}_q{Q}.wav"
        deliver.write_wav(wav_path, play_l, play_r, rate)
        (out / f"{stem}_q{Q}.fspz").write_bytes(blob)
        row = {
            "q": Q,
            "dial_band": band_index,
            "dial_name": quiet.NAMES[band_index],
            "source_pass": folder,
            "frames_raised": int(keep.sum()),
            "frames": int(keep.size),
            "archive_bytes": len(blob),
            "source_bytes": source_bytes,
            "opus96_bytes": opus_bytes[stem],
            "ratio_vs_opus96": len(blob) / opus_bytes[stem],
            "ratio_vs_source": len(blob) / source_bytes,
            "under_opus": len(blob) < opus_bytes[stem],
            "mean_abs_error": float(np.mean(np.abs(err))),
            "snr_db": snr,
            "quiet15_snr_db": {name: value for name, value in zip(quiet.NAMES, ours_q)},
            "quiet15_opus_db": {name: value for name, value in zip(quiet.NAMES, opus_q)},
            "wav": str(wav_path),
        }
        songs.append({"stem": stem, "source": str(path), "seconds": left.size / rate, "row": row})
        paths.save_score("gap_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "rule": "one finer rung on the remaining worst quiet octave, only on frames that cover a quiet hop",
                "bands": {item[0]: quiet.NAMES[item[1]] for item in GAPS},
                "quiet_fraction": quiet.QUIET,
                "kept": "loud frames and every other band stay",
                "songs": [item[0] for item in GAPS],
                "coder": "tenth-pass magnitude bucket, not retuned",
                "q": Q,
            },
            "songs": songs,
        })
        name = quiet.NAMES[band_index]
        print(
            f"  archive {len(blob)} {folder} {source_bytes} opus {opus_bytes[stem]} "
            f"ratio {row['ratio_vs_opus96']:.3f} vs source {row['ratio_vs_source']:.3f} "
            f"raised {int(keep.sum())}/{int(keep.size)} "
            f"mean_abs {row['mean_abs_error']:.1f} SNR {snr:.2f} "
            f"quiet {name} {ours_q[band_index]:.1f} opus {opus_q[band_index]:.1f}",
            flush=True,
        )
    print("gap done", flush=True)


if __name__ == "__main__":
    main()
