#!/usr/bin/env python3
"""Twentieth audio pass. Declared before this pass is scored.

Eminem and Bruno are over Opus 96. Sublime and Clay are not, and they
stay at the nineteenth pass. The overage is the quiet-octave repair
from the eighteenth and nineteenth passes, stored on every frame.
The needle those rungs were for sits in the quietest 15 percent of
hops. A loud frame does not hold that needle.

A transform frame covers two hops. A frame keeps the nineteenth-pass
rung when either hop in its window is inside that quiet set. Every
other frame of the dialed octave returns to the tenth-pass quotients
and zoom. The quiet fraction is not moved. No other band is coarsened.
Q stays 2048. The bucket is not retuned.

  Eminem  0-2.8 kHz
  Bruno   2.8-5.5 kHz

If an archive is still over Opus after this, that size miss stays.
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
import refine_zoom as zoom

Q = 2048
MAGIC = b"FSPT"
EDGES = (0, 32, 64, 128, 256)
# Same octaves as r18 and r19. Not rechosen from the archive size.
OVER = (
    ("eminem_lose_yourself", 0),
    ("bruno_mars_just_the_way_you_are", 1),
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
        raise SystemExit("not a trimmed archive")
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
        raise SystemExit("trailing trimmed bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def quiet_hops(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    hop = zoom.HOP
    count = left.size // hop
    energy = left[: count * hop].reshape(count, hop).astype(np.float64) ** 2
    energy = energy + right[: count * hop].reshape(count, hop).astype(np.float64) ** 2
    hop_energy = energy.mean(axis=1)
    cut = float(np.quantile(hop_energy, quiet.QUIET))
    return hop_energy <= cut


def frames_covering(n_frames: int, hops: np.ndarray) -> np.ndarray:
    """A frame stays fine when either hop in its window is quiet."""
    keep = np.zeros(n_frames, dtype=bool)
    idx = np.flatnonzero(hops)
    for shifted in (idx, idx + 1):
        valid = shifted < n_frames
        keep[shifted[valid]] = True
    return keep


def restore_loud(fine_q, fine_b, base_q, base_b, band_index: int, keep: np.ndarray):
    if fine_q.shape != base_q.shape or fine_b.shape != base_b.shape:
        raise SystemExit("the tenth-pass frame count does not match")
    if keep.shape != (fine_q.shape[0],):
        raise SystemExit("the quiet frame mask does not match")
    out_q = np.array(fine_q, copy=True)
    out_b = np.array(fine_b, dtype=np.int16, copy=True)
    lo, hi = EDGES[band_index], EDGES[band_index + 1]
    release = ~keep
    out_q[release, lo:hi] = base_q[release, lo:hi]
    out_b[release, band_index] = base_b[release, band_index]
    return out_q, out_b


def self_check() -> None:
    gauge.self_check()
    hops = np.array([True, False, True, False], dtype=bool)
    keep = frames_covering(5, hops)
    if not np.array_equal(keep, np.array([True, True, True, True, False])):
        raise SystemExit("a quiet hop did not keep both frames in its window")
    fine_q = np.arange(5 * 256, dtype=np.int32).reshape(5, 256)
    base_q = fine_q + 7
    fine_b = np.zeros((5, 4), dtype=np.int16)
    base_b = np.ones((5, 4), dtype=np.int16)
    fine_b[:, 0] = 4
    base_b[:, 0] = 2
    out_q, out_b = restore_loud(fine_q, fine_b, base_q, base_b, 0, keep)
    if not np.array_equal(out_q[:4], fine_q[:4]) or not np.array_equal(out_q[4, :32], base_q[4, :32]):
        raise SystemExit("the loud frame did not return to the earlier quotients")
    if np.any(out_q[:, 32:] != fine_q[:, 32:]):
        raise SystemExit("a band outside the dialed octave was rewritten")
    if not np.array_equal(out_b[:4, 0], fine_b[:4, 0]) or int(out_b[4, 0]) != 2:
        raise SystemExit("the loud zoom did not return")
    if not np.array_equal(out_b[:, 1:], fine_b[:, 1:]):
        raise SystemExit("an untouched zoom moved")
    blob = pack_song(Q, 5 * zoom.HOP, out_q, out_b, out_q, out_b, 44100)
    got_q, got_n, got_mid, got_mb, _side, _sb = unpack_song(blob, 44100)
    if got_q != Q or got_n != 5 * zoom.HOP or not np.array_equal(got_mid, out_q) or not np.array_equal(got_mb, out_b):
        raise SystemExit("trimmed archive did not round-trip")
    print("trim self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("r20")
    opus, _previous = zoom.opus_and_r5()
    songs = []
    for stem, band_index in OVER:
        print(f"song {stem}", flush=True)
        path = band.SONGS[band.STEMS.index(stem)]
        left, right, rate = deliver.decode_pcm(path)
        mid, side = deliver.mid_side(left, right)
        fine_path = paths.CURRENT / "r19" / f"{stem}_q{Q}.fspz"
        base_path = paths.CURRENT / "r10" / f"{stem}_q{Q}.fspz"
        _q, original, qm, bm, qs, bs = needle.unpack_song(fine_path.read_bytes(), rate)
        bq, boriginal, bqm, bbm, bqs, bbs = context.unpack_song(base_path.read_bytes(), rate)
        if _q != Q or bq != Q or original != int(left.size) or boriginal != original:
            raise SystemExit(f"{stem} archives do not match the song")
        bm = quiet.as_bands(bm)
        bs = quiet.as_bands(bs)
        bbm = quiet.as_bands(bbm)
        bbs = quiet.as_bands(bbs)
        hops = quiet_hops(left, right)
        keep = frames_covering(qm.shape[0], hops)
        source_bytes = fine_path.stat().st_size
        mid_qq, mid_b = restore_loud(qm, bm, bqm, bbm, band_index, keep)
        side_qq, side_b = restore_loud(qs, bs, bqs, bbs, band_index, keep)
        lo, hi = EDGES[band_index], EDGES[band_index + 1]
        if np.any(mid_qq[keep, lo:hi] != qm[keep, lo:hi]) or np.any(mid_b[keep, band_index] != bm[keep, band_index]):
            raise SystemExit(f"{stem} a quiet frame lost the fine rung")
        if np.any(mid_qq[~keep, lo:hi] != bqm[~keep, lo:hi]):
            raise SystemExit(f"{stem} a loud frame did not return")
        untouched = [i for i in range(4) if i != band_index]
        if not np.array_equal(mid_qq[:, :lo], qm[:, :lo]) or not np.array_equal(mid_b[:, untouched], bm[:, untouched]):
            raise SystemExit(f"{stem} a band outside the dialed octave moved")
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
            "frames_kept_fine": int(keep.sum()),
            "frames": int(keep.size),
            "archive_bytes": len(blob),
            "r19_bytes": source_bytes,
            "opus96_bytes": opus[stem],
            "ratio_vs_opus96": len(blob) / opus[stem],
            "ratio_vs_r19": len(blob) / source_bytes,
            "under_opus": len(blob) < opus[stem],
            "mean_abs_error": float(np.mean(np.abs(err))),
            "snr_db": snr,
            "quiet15_snr_db": {name: value for name, value in zip(quiet.NAMES, quiet_db)},
            "wav": str(wav_path),
        }
        songs.append({"stem": stem, "source": str(path), "seconds": left.size / rate, "row": row})
        paths.save_score("trim_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "rule": "loud frames of the dialed octave return to the tenth pass; quiet-covering frames keep the nineteenth",
                "bands": {item[0]: quiet.NAMES[item[1]] for item in OVER},
                "quiet_fraction": quiet.QUIET,
                "kept": "Sublime and Clay stay at r19; no other band is coarsened",
                "songs": [item[0] for item in OVER],
                "coder": "tenth-pass magnitude bucket, not retuned",
                "q": Q,
            },
            "songs": songs,
        })
        quiet_text = " ".join(f"{name} {value:.1f}" for name, value in zip(quiet.NAMES, quiet_db))
        print(
            f"  archive {len(blob)} r19 {source_bytes} opus {opus[stem]} "
            f"ratio {row['ratio_vs_opus96']:.3f} vs r19 {row['ratio_vs_r19']:.3f} "
            f"fine frames {int(keep.sum())}/{int(keep.size)} "
            f"mean_abs {row['mean_abs_error']:.1f} SNR {snr:.2f} quiet {quiet_text}",
            flush=True,
        )
    print("trim done", flush=True)


if __name__ == "__main__":
    main()
