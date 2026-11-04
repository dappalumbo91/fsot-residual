#!/usr/bin/env python3
"""Twenty-eighth audio pass. Declared before this pass is scored.

On the twenty-seventh pass, Lose Yourself's quiet slice against Opus 96 was:

  0-2.8 kHz    19.26 against 21.03. Gap -1.77 dB. This is the worst.
  2.8-5.5 kHz   9.76 against 11.14. Gap -1.38 dB.
  5.5-11 kHz   10.29 against 8.66. Already ahead. Not raised.
  11-22 kHz     9.58 against 8.35. Already ahead. Not raised.

The wrong term is the gauge on the quiet bass. One existing rung, both
channels, only on frames whose window covers a quiet hop. Loud frames
stay. Every other band stays. Quotients are retaken from the original
coefficients. The quiet fraction is not moved. Q stays 2048. The bucket
is not retuned.

Every other song stays. The margin under Opus is 490,458 bytes. A song
that crosses stays a size miss. No other band is coarsened to pay for
the rung.
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
import refine_gap as gap
import refine_gauge as gauge
import refine_miss as miss
import refine_quiet as quiet
import refine_top as top
import refine_trim as trim
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPO"
EDGES = (0, 32, 64, 128, 256)
GAPS = (("eminem_lose_yourself", 0),)


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
        raise SystemExit("not a bass archive")
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
        raise SystemExit("trailing bass bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    gauge.self_check()
    rng = np.random.default_rng(28)
    coeffs = rng.normal(0, 80000, (4, 256))
    qq = band.quantize(coeffs, zoom.step_at(Q, 2)).astype(np.int32)
    b_band = np.full((4, 4), 2, dtype=np.int16)
    keep = np.array([False, True, False, True])
    out_q, out_b = gap.raise_quiet(coeffs, qq, b_band, 0, keep)
    if not np.array_equal(out_b[:, 0], np.array([2, 3, 2, 3], dtype=np.int16)):
        raise SystemExit("the quiet bass octave did not take one rung")
    if np.any(out_q[0] != qq[0]) or np.any(out_q[2] != qq[2]):
        raise SystemExit("a loud frame was rewritten")
    expect = band.quantize(coeffs[1, 0:32], zoom.step_at(Q, 3))
    if not np.array_equal(out_q[1, 0:32], expect):
        raise SystemExit("the quiet frame was not retaken from the original coefficients")
    blob = pack_song(Q, 4 * zoom.HOP, out_q, out_b, qq, b_band, 44100)
    got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, 44100)
    if got_q != Q or got_n != 4 * zoom.HOP or not np.array_equal(got_mid, out_q) or not np.array_equal(got_mb, out_b):
        raise SystemExit("bass archive did not round-trip")
    if not np.array_equal(got_side, qq) or not np.array_equal(got_sb, b_band):
        raise SystemExit("the untouched channel did not round-trip")
    print("bass self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("r28")
    opus_bytes, _previous = zoom.opus_and_r5()
    songs = []
    for stem, band_index in GAPS:
        print(f"song {stem}", flush=True)
        path = band.SONGS[band.STEMS.index(stem)]
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        archive_path = paths.CURRENT / "r27" / f"{stem}_q{Q}.fspz"
        _q, original, qm, bm, qs, bs = top.unpack_song(archive_path.read_bytes(), rate)
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
        lo, hi = EDGES[band_index], EDGES[band_index + 1]
        if np.any(mid_qq[~keep, lo:hi] != qm[~keep, lo:hi]):
            raise SystemExit(f"{stem} a loud frame moved")
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
        opus_l, opus_r, rate_o = deliver.decode_pcm(paths.REFERENCE / "opus" / f"{stem}_opus96.opus")
        n = min(left.size, opus_l.size)
        opus_q = quiet.quiet_snr(left[:n], right[:n], opus_l[:n], opus_r[:n], rate_o)
        wav_path = out / f"{stem}_q{Q}.wav"
        deliver.write_wav(wav_path, play_l, play_r, rate)
        (out / f"{stem}_q{Q}.fspz").write_bytes(blob)
        seconds = left.size / rate
        row = {
            "q": Q,
            "dial_band": band_index,
            "dial_name": quiet.NAMES[band_index],
            "frames_raised": int(keep.sum()),
            "frames": int(keep.size),
            "archive_bytes": len(blob),
            "r27_bytes": source_bytes,
            "opus96_bytes": opus_bytes[stem],
            "ratio_vs_opus96": len(blob) / opus_bytes[stem],
            "ratio_vs_r27": len(blob) / source_bytes,
            "under_opus": len(blob) < opus_bytes[stem],
            "kbps": len(blob) * 8 / seconds / 1000,
            "opus_kbps": opus_bytes[stem] * 8 / seconds / 1000,
            "mean_abs_error": float(np.mean(np.abs(err))),
            "snr_db": snr,
            "quiet15_snr_db": {name: value for name, value in zip(quiet.NAMES, ours_q)},
            "quiet15_opus_db": {name: value for name, value in zip(quiet.NAMES, opus_q)},
            "wav": str(wav_path),
        }
        songs.append({"stem": stem, "source": str(path), "seconds": seconds, "row": row})
        paths.save_score("bass_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "rule": "one finer rung on Lose Yourself's quiet 0-2.8 kHz octave",
                "bands": {item[0]: quiet.NAMES[item[1]] for item in GAPS},
                "quiet_fraction": quiet.QUIET,
                "kept": "loud frames and every other band stay",
                "songs": [item[0] for item in GAPS],
                "coder": "tenth-pass magnitude bucket, not retuned",
                "q": Q,
            },
            "songs": songs,
        })
        ours_text = " ".join(f"{name} {value:.1f}" for name, value in zip(quiet.NAMES, ours_q))
        opus_text = " ".join(f"{value:.1f}" for value in opus_q)
        print(
            f"  archive {len(blob)} r27 {source_bytes} opus {opus_bytes[stem]} "
            f"ratio {row['ratio_vs_opus96']:.3f} {row['kbps']:.1f}kbps "
            f"raised {int(keep.sum())}/{int(keep.size)} "
            f"SNR {snr:.2f} quiet {ours_text} opus {opus_text}",
            flush=True,
        )
    print("bass done", flush=True)


if __name__ == "__main__":
    main()
