#!/usr/bin/env python3
"""Thirty-fourth audio pass. Declared from the quiet trace, before this score.

On the r33 playback the quiet gaps versus Opus are -4.390, -3.233,
-3.249, and +0.611 dB. The most negative is 2.8-5.5 kHz. Counted on
the r33 archive before this run, quiet frames of that band:

  mid  kept 2727/5266 at the stored rung, 3182/5266 at the next rung.
  side kept 3041/5266 at the stored rung, 3531/5266 at the next rung.

The stored rung is b=2. The next rung already keeps the peak on most
of those frames, so this is not the dead-zone rule. This pass raises
only that band, both channels, by one existing rung, and only on
frames whose window covers a quiet hop. Loud frames and every other
band stay. Quotients are retaken from the original coefficients.
Q stays 2048. The bucket is not retuned. Cage starts from r33.
The other six songs stay.

A file that crosses Opus 96 stays a size miss. No band is coarsened
to pay for the rung. The 11-22 kHz peak rule is not run again.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import deliver
import fspack
import paths
import refine_cagepeak as peaked
import refine_context as context
import refine_dial as dial
import refine_fullband as band
import refine_gap as gap
import refine_gauge as gauge
import refine_miss as miss
import refine_quiet as quiet
import refine_trim as trim
import refine_zoom as zoom

Q = 2048
MAGIC = b"FS34"
STEM = peaked.STEM
BAND_INDEX = 1
NAMES = dial.NAMES


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
        raise SystemExit("not a cage-quiet archive")
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
        raise SystemExit("trailing cage-quiet bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    trace = json.loads((paths.SCORES / "cage_quiet_trace.json").read_text(encoding="utf-8"))
    if trace.get("candidate_band") != BAND_INDEX or trace.get("candidate_name") != NAMES[BAND_INDEX]:
        raise SystemExit("the locked band is not the quiet-trace candidate")
    quiet_mid = trace["mid"]["quiet"]
    if quiet_mid["kept_at_next"] != 3182 or quiet_mid["frames"] != 5266:
        raise SystemExit("the quiet peak count does not match the locked trace")
    rng = np.random.default_rng(34)
    coeffs = rng.normal(0, 80000, (4, 256))
    qq = band.quantize(coeffs, zoom.step_at(Q, 2)).astype(np.int32)
    b_band = np.full((4, 4), 2, dtype=np.int16)
    keep = np.array([True, False, True, False])
    out_q, out_b = gap.raise_quiet(coeffs, qq, b_band, BAND_INDEX, keep)
    if not np.array_equal(out_b[:, BAND_INDEX], np.array([3, 2, 3, 2], dtype=np.int16)):
        raise SystemExit("the quiet presence band did not take one rung")
    if np.any(out_q[1] != qq[1]):
        raise SystemExit("a loud frame was rewritten")
    blob = pack_song(Q, 4 * zoom.HOP, out_q, out_b, qq, b_band, 44100)
    got_q, got_n, got_mid, _mb, got_side, _sb = unpack_song(blob, 44100)
    if got_q != Q or got_n != 4 * zoom.HOP or not np.array_equal(got_mid, out_q) or not np.array_equal(got_side, qq):
        raise SystemExit("cage-quiet archive did not round-trip")
    print("cage-quiet self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("r34")
    opus, _previous = zoom.opus_and_r5()
    stem = STEM
    band_index = BAND_INDEX
    print(f"song {stem}", flush=True)
    path = band.SONGS[band.STEMS.index(stem)]
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    archive_path = paths.CURRENT / "r33" / f"{stem}_q{Q}.fspz"
    _q, original, qm, bm, qs, bs = peaked.unpack_song(archive_path.read_bytes(), rate)
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
    mid_qq, mid_b = gap.raise_quiet(cm, qm, bm, band_index, keep)
    side_qq, side_b = gap.raise_quiet(cs, qs, bs, band_index, keep)
    lo, hi = gap.EDGES[band_index], gap.EDGES[band_index + 1]
    if np.any(mid_qq[~keep, lo:hi] != qm[~keep, lo:hi]):
        raise SystemExit(f"{stem} a loud frame moved")
    untouched = [i for i in range(4) if i != band_index]
    if not np.array_equal(mid_b[:, untouched], bm[:, untouched]):
        raise SystemExit(f"{stem} a band outside the quiet gap moved")
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
    octaves = dial.band_snr(left, right, play_l, play_r, rate)
    ours_q = quiet.quiet_snr(left, right, play_l, play_r, rate)
    opus_l, opus_r, rate_o = deliver.decode_pcm(paths.REFERENCE / "opus" / f"{stem}_opus96.opus")
    n = min(left.size, opus_l.size)
    opus_full = dial.band_snr(left[:n], right[:n], opus_l[:n], opus_r[:n], rate_o)
    opus_q = quiet.quiet_snr(left[:n], right[:n], opus_l[:n], opus_r[:n], rate_o)
    wav_path = out / f"{stem}_q{Q}.wav"
    deliver.write_wav(wav_path, play_l, play_r, rate)
    (out / f"{stem}_q{Q}.fspz").write_bytes(blob)
    opus_bytes = opus[stem]
    row = {
        "q": Q,
        "dial_band": band_index,
        "dial_name": NAMES[band_index],
        "archive_bytes": len(blob),
        "r33_bytes": source_bytes,
        "opus96_bytes": opus_bytes,
        "ratio_vs_opus96": len(blob) / opus_bytes,
        "under_opus": len(blob) < opus_bytes,
        "raised_frames": int(keep.sum()),
        "frames": int(keep.size),
        "snr_db": snr,
        "band_snr_db": {name: value for name, value in zip(NAMES, octaves)},
        "opus_band_snr_db": {name: value for name, value in zip(NAMES, opus_full)},
        "quiet_snr_db": {name: value for name, value in zip(NAMES, ours_q)},
        "opus_quiet_snr_db": {name: value for name, value in zip(NAMES, opus_q)},
        "wav": str(wav_path),
    }
    paths.save_score("cage_quiet_report.json", {
        "pin": fspack.PIN,
        "declared": {
            "rule": "quiet octave furthest behind on r33, one rung, quiet frames only",
            "band": NAMES[band_index],
            "kept": "loud frames and every other band stay",
            "songs": [stem],
            "coder": "tenth-pass magnitude bucket, not retuned",
            "q": Q,
            "source": "r33",
        },
        "songs": [{"stem": stem, "source": str(path), "seconds": left.size / rate, "row": row}],
    })
    print(
        f"  archive {len(blob)} r33 {source_bytes} opus {opus_bytes} "
        f"ratio {row['ratio_vs_opus96']:.3f} "
        f"quiet {NAMES[band_index]} {ours_q[band_index]:.3f} vs opus {opus_q[band_index]:.3f} "
        f"raised {int(keep.sum())}/{int(keep.size)} SNR {snr:.2f}",
        flush=True,
    )
    print("cage quiet done", flush=True)


if __name__ == "__main__":
    main()
