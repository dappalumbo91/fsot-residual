#!/usr/bin/env python3
"""Cage top-octave dead-zone trace. Not a compressor pass.

Declared before it is run. r32 raised 11-22 kHz from b=0 to b=1 on
every frame. The full-song SNR moved from 0.047 dB to 0.528 dB against
Opus 1.881. This script counts whether that rung, and the next one,
keep the largest coefficient. It does not pack an archive and it does
not choose a rung.

For each channel, peak is the largest absolute coefficient in bins
128:256. A rung b keeps the peak when peak >= step(b) / 2. step is
the existing zoom.step_at ladder. Quiet frames are the r20 set:
a frame whose window covers a hop in the quietest 15 percent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import paths
import refine_above as above
import refine_dial as dial
import refine_fullband as band
import refine_fullband as songs
import refine_quiet as quiet
import refine_trim as trim
import refine_zoom as zoom

STEM = above.STEM
Q = above.Q
BAND = 3
LO, HI = 128, 256


def count(coeffs: np.ndarray, stored: np.ndarray, keep: np.ndarray) -> dict:
    peak = np.max(np.abs(coeffs[:, LO:HI]), axis=1)
    holds = []
    for b in range(zoom.B_MAX + 1):
        holds.append(peak >= (zoom.step_at(Q, b) / 2.0))
    held = np.column_stack(holds)
    at_stored = held[np.arange(peak.size), stored]
    nxt = np.minimum(stored.astype(np.int16) + 1, zoom.B_MAX)
    at_next = held[np.arange(peak.size), nxt]
    coarsest = np.full(peak.size, -1, dtype=np.int16)
    for b in range(zoom.B_MAX + 1):
        pick = held[:, b] & (coarsest < 0)
        coarsest[pick] = b

    def pack(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        return {
            "frames": n,
            "kept_at_stored": int(at_stored[mask].sum()),
            "kept_at_next": int(at_next[mask].sum()),
            "median_peak": float(np.median(peak[mask])) if n else None,
            "median_stored_b": float(np.median(stored[mask])) if n else None,
            "median_peak_keeping_b": float(np.median(coarsest[mask])) if n else None,
        }

    return {"all": pack(np.ones(peak.size, dtype=bool)), "quiet": pack(keep)}


def main() -> None:
    path = songs.SONGS[songs.STEMS.index(STEM)]
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    archive = paths.CURRENT / "r32" / f"{STEM}_q{Q}.fspz"
    _q, original, qm, bm, qs, bs = above.unpack_song(archive.read_bytes(), rate)
    if _q != Q or original != int(left.size):
        raise SystemExit("r32 archive does not match the song")
    cm, n_mid = band.mdct_forward(mid)
    cs, n_side = band.mdct_forward(side)
    if n_mid != original or cm.shape != qm.shape:
        raise SystemExit("cosine frame count drifted")
    bm = quiet.as_bands(bm)
    bs = quiet.as_bands(bs)
    keep = trim.frames_covering(qm.shape[0], trim.quiet_hops(left, right))
    payload = {
        "stem": STEM,
        "band": dial.NAMES[BAND],
        "rule": "count peaks kept at the stored rung and at the next rung; do not pack",
        "mid": count(cm, bm[:, BAND], keep),
        "side": count(cs, bs[:, BAND], keep),
    }
    out = paths.save_score("cage_peak_trace.json", payload)
    for channel in ("mid", "side"):
        for slice_name in ("all", "quiet"):
            row = payload[channel][slice_name]
            print(
                f"{channel} {slice_name} kept {row['kept_at_stored']}/{row['frames']} "
                f"next {row['kept_at_next']}/{row['frames']} "
                f"median peak {row['median_peak']:.1f} "
                f"stored b {row['median_stored_b']:.1f} "
                f"keeping b {row['median_peak_keeping_b']:.1f}",
                flush=True,
            )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
