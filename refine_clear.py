#!/usr/bin/env python3
"""Fourteenth audio pass. Declared before this pass is scored.

VNV's twelfth-pass listen was dull. The thirteenth pass put the center
of 2.8-5.5 kHz back and the vocal got clearer. Distortion is still in
the voice, and the calculating tone came back with it.

Traced on the thirteenth-pass playback against the source, before this
pass is scored. The 0-2.8 kHz band is at 27 dB. The 2.8-5.5 kHz band is
at 13 dB. The 5.5-11 kHz band is at 0 dB: it holds 2.7% of the song and
67% of the error, because that band is stored as zero. 11-22 kHz is the
same kind of deletion and is 24% of the error.

The wrong term is the missing 5.5-11 kHz mid. The vocal is the mid.
This pass puts that band back from the ninth-pass quotients and zoom,
mid only. The side of that band stays zero. 11-22 kHz stays zero. The
approximation and the 2.8-5.5 kHz split stay as the thirteenth pass left
them. Q stays 2048. The context bucket is not retuned. Skrillex, Cage,
and the four earlier wins are not rebuilt.
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
import refine_context as context
import refine_fullband as band
import refine_gauge as gauge
import refine_miss as miss
import refine_voice as voice
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPL"
STEM = voice.STEM
D1 = 2


def restore_mid_d1(current_q, current_b, fine_q, fine_b):
    edges = gauge.band_edges(current_q.shape[1], 44100, zoom.HOP)
    if edges != [0, 32, 64, 128, 256]:
        raise SystemExit("the fs/8 cut is not bin 64")
    if current_q.shape != fine_q.shape or current_b.shape != fine_b.shape:
        raise SystemExit("mid sources do not match")
    out_q = np.array(current_q, copy=True)
    out_b = np.array(current_b, dtype=np.int16, copy=True)
    lo, hi = edges[D1], edges[D1 + 1]
    out_q[:, lo:hi] = fine_q[:, lo:hi]
    out_b[:, D1] = fine_b[:, D1]
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
        raise SystemExit("not a clarity archive")
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
        raise SystemExit("trailing clarity bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    gauge.self_check()
    rng = np.random.default_rng(14)
    current = np.zeros((6, 256), dtype=np.int32)
    current[:, :64] = rng.integers(-3, 4, (6, 64), dtype=np.int32)
    fine = current.copy()
    fine[:, 64:128] = rng.integers(-5, 6, (6, 64), dtype=np.int32)
    current_b = np.zeros((6, 4), dtype=np.int16)
    current_b[:, 0] = 3
    current_b[:, 1] = 2
    fine_b = current_b.copy()
    fine_b[:, D1] = 4
    mid_q, mid_b = restore_mid_d1(current, current_b, fine, fine_b)
    if not np.array_equal(mid_q[:, 64:128], fine[:, 64:128]) or not np.array_equal(mid_b[:, D1], fine_b[:, D1]):
        raise SystemExit("the mid 5.5-11 kHz band did not return")
    if not np.array_equal(mid_q[:, :64], current[:, :64]) or np.any(mid_q[:, 128:]):
        raise SystemExit("a kept band or the top octave moved")
    blob = pack_song(Q, 6 * zoom.HOP, mid_q, mid_b, current, current_b, 44100)
    got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, 44100)
    if got_q != Q or got_n != 6 * zoom.HOP:
        raise SystemExit("header did not round-trip")
    if not np.array_equal(got_mid, mid_q) or not np.array_equal(got_side, current):
        raise SystemExit("clarity coefficients did not round-trip")
    if not np.array_equal(got_mb, mid_b) or not np.array_equal(got_sb, current_b):
        raise SystemExit("clarity zoom did not round-trip")
    print("clear self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("r14")
    opus, _previous = zoom.opus_and_r5()
    earlier = json.loads((paths.SCORES / "voice_report.json").read_text(encoding="utf-8"))
    r13_bytes = int(earlier["songs"][0]["row"]["archive_bytes"])
    path = band.SONGS[band.STEMS.index(STEM)]
    print(f"song {STEM}", flush=True)
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    _q13, n13, q13_m, b13_m, q13_s, b13_s = voice.unpack_song(
        (paths.CURRENT / "r13" / f"{STEM}_q{Q}.fspz").read_bytes(), rate
    )
    _q9, n9, q9_m, b9_m, _q9_s, _b9_s = miss.unpack_song(
        (paths.CURRENT / "r9" / f"{STEM}_q{Q}.fspz").read_bytes(), rate
    )
    if _q13 != Q or _q9 != Q or n13 != int(left.size) or n9 != n13:
        raise SystemExit("source archives do not match the song")
    if np.any(q13_m[:, 64:]) or np.any(q13_s[:, 64:]):
        raise SystemExit("the thirteenth pass is not silent above fs/8")
    mid_qq, mid_b = restore_mid_d1(q13_m, b13_m, q9_m, b9_m)
    side_qq, side_b = q13_s, b13_s
    if np.any(side_qq[:, 64:]) or np.any(mid_qq[:, 128:]):
        raise SystemExit("the side brightness or the top octave was filled in")
    original = n13
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
        "r13_bytes": r13_bytes,
        "opus96_bytes": opus[STEM],
        "ratio_vs_opus96": len(blob) / opus[STEM],
        "ratio_vs_r13": len(blob) / r13_bytes,
        "mean_abs_error": float(np.mean(np.abs(err))),
        "hop_mean_mid": float(hop_m[hop_m > 0].mean()),
        "hop_mean_side": float(hop_s[hop_s > 0].mean()),
        "mid_d1_zoom": float(np.mean(mid_b[:, D1])),
        "snr_db": snr,
        "wav": str(wav_path),
    }
    paths.save_score("clear_report.json", {
        "pin": fspack.PIN,
        "declared": {
            "song": STEM,
            "keeps": "r12 skrillex and cage; r10 sublime, clay, eminem, bruno",
            "change": "mid 5.5-11 kHz returns from the ninth pass",
            "side_above_fs8": "stays zero",
            "top_octave": "stays zero",
            "coder": "tenth-pass magnitude bucket, not retuned",
            "q": Q,
        },
        "songs": [{"stem": STEM, "source": str(path), "seconds": left.size / rate, "row": row}],
    })
    print(
        f"  archive {len(blob)} r13 {r13_bytes} opus {opus[STEM]} "
        f"ratio {row['ratio_vs_opus96']:.3f} vs r13 {row['ratio_vs_r13']:.3f} "
        f"mid d1 {row['mid_d1_zoom']:.2f} "
        f"mean_abs {row['mean_abs_error']:.1f} "
        f"hop {row['hop_mean_mid']:.1f}/{row['hop_mean_side']:.1f} SNR {snr:.2f}",
        flush=True,
    )
    print("clear done", flush=True)


if __name__ == "__main__":
    main()
