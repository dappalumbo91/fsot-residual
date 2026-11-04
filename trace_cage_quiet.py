#!/usr/bin/env python3
"""Cage quiet-band trace on the r33 playback. Not a compressor pass.

Declared before it is run. The selector is the quiet octave whose
(r33 SNR minus Opus SNR) is the most negative in cage_peak_report.json.
A tie takes the lower band index. Full-song gaps are not the selector.
Archive size is not the selector.

On that one band, count how many frames keep the largest coefficient
at the stored rung and at the next rung. A rung keeps the peak when
peak >= step(b) / 2. Quiet frames are the r20 set. This script does
not pack an archive and does not raise a second band.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import paths
import refine_cagepeak as peakpass
import refine_dial as dial
import refine_fullband as band
import refine_quiet as quiet
import refine_trim as trim
import refine_zoom as zoom

STEM = peakpass.STEM
Q = peakpass.Q
EDGES = (0, 32, 64, 128, 256)


def main() -> None:
    payload = json.loads((paths.SCORES / "cage_peak_report.json").read_text(encoding="utf-8"))
    row = payload["songs"][0]["row"]
    if payload["songs"][0]["stem"] != STEM:
        raise SystemExit("r33 report is not Cage")
    gaps = [row["quiet_snr_db"][name] - row["opus_quiet_snr_db"][name] for name in dial.NAMES]
    worst = min(range(4), key=lambda index: (gaps[index], index))
    if gaps[worst] >= 0:
        print("candidate none", flush=True)
        paths.save_score("cage_quiet_trace.json", {"candidate_band": None, "gaps": gaps})
        return
    lo, hi = EDGES[worst], EDGES[worst + 1]
    path = band.SONGS[band.STEMS.index(STEM)]
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    archive = paths.CURRENT / "r33" / f"{STEM}_q{Q}.fspz"
    _q, original, qm, bm, qs, bs = peakpass.unpack_song(archive.read_bytes(), rate)
    if _q != Q or original != int(left.size):
        raise SystemExit("r33 archive does not match the song")
    cm, n_mid = band.mdct_forward(mid)
    cs, _n_side = band.mdct_forward(side)
    if n_mid != original or cm.shape != qm.shape:
        raise SystemExit("cosine frame count drifted")
    bm = quiet.as_bands(bm)
    bs = quiet.as_bands(bs)
    keep = trim.frames_covering(qm.shape[0], trim.quiet_hops(left, right))

    def count(coeffs: np.ndarray, stored: np.ndarray) -> dict:
        peak = np.max(np.abs(coeffs[:, lo:hi]), axis=1)
        held_now = peak >= (zoom.step_at(Q, 0) / 2.0)
        holds = np.zeros((peak.size, zoom.B_MAX + 1), dtype=bool)
        for b in range(zoom.B_MAX + 1):
            holds[:, b] = peak >= (zoom.step_at(Q, b) / 2.0)
        at_stored = holds[np.arange(peak.size), stored]
        nxt = np.minimum(stored.astype(np.int16) + 1, zoom.B_MAX)
        at_next = holds[np.arange(peak.size), nxt]
        out = {}
        for name, mask in (("all", np.ones(peak.size, dtype=bool)), ("quiet", keep)):
            n = int(mask.sum())
            out[name] = {
                "frames": n,
                "kept_at_stored": int(at_stored[mask].sum()),
                "kept_at_next": int(at_next[mask].sum()),
                "median_b": float(np.median(stored[mask])) if n else None,
                "median_peak": float(np.median(peak[mask])) if n else None,
            }
        del held_now
        return out

    result = {
        "stem": STEM,
        "rule": "quiet octave with the most negative r33-minus-Opus gap; tie takes the lower index",
        "gaps": {name: gaps[index] for index, name in enumerate(dial.NAMES)},
        "candidate_band": worst,
        "candidate_name": dial.NAMES[worst],
        "mid": count(cm, bm[:, worst]),
        "side": count(cs, bs[:, worst]),
    }
    out = paths.save_score("cage_quiet_trace.json", result)
    print(f"candidate {worst} {dial.NAMES[worst]} gap {gaps[worst]:+.3f}", flush=True)
    for channel in ("mid", "side"):
        for slice_name in ("all", "quiet"):
            item = result[channel][slice_name]
            print(
                f"{channel} {slice_name} kept {item['kept_at_stored']}/{item['frames']} "
                f"next {item['kept_at_next']}/{item['frames']} "
                f"median b {item['median_b']:.1f} peak {item['median_peak']:.1f}",
                flush=True,
            )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
