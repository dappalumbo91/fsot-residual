#!/usr/bin/env python3
"""Cage band trace. Not a compressor pass. Declared before it is run.

Reads the r16 playback and the Opus 96 decode against the same source PCM.
Prints full-song band SNR and quiet-15% band SNR on the four octaves.

The candidate is the full-song octave whose (r16 SNR minus Opus SNR) is
the most negative. A tie takes the lower band index. If that gap is not
negative, the candidate is none. Quiet-slice numbers are recorded and
are not the selector. Archive size is not the selector. This script
does not pack an archive and does not choose a second band.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import paths
import refine_dial as dial
import refine_fullband as band
import refine_quiet as quiet

STEM = "cage_the_elephant_aint_no_rest"
NAMES = dial.NAMES
RULE = (
    "full-song octave with the most negative r16-minus-Opus gap; "
    "tie takes the lower band index; quiet slice is not the selector"
)


def read_wav(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 2 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path.name} is not stereo 16-bit")
        rate = handle.getframerate()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.int32)
    if pcm.size % 2:
        raise SystemExit(f"{path.name} has a dangling sample")
    return pcm[0::2].copy(), pcm[1::2].copy(), rate


def main() -> None:
    source = band.SONGS[band.STEMS.index(STEM)]
    playback = paths.CURRENT / "r16" / f"{STEM}_q2048.wav"
    opus_path = paths.REFERENCE / "opus" / f"{STEM}_opus96.opus"
    if not playback.is_file():
        raise SystemExit(f"missing {playback}")
    if not opus_path.is_file():
        raise SystemExit(f"missing {opus_path}")
    left, right, rate = deliver.decode_pcm(source)
    play_l, play_r, play_rate = read_wav(playback)
    opus_l, opus_r, opus_rate = deliver.decode_pcm(opus_path)
    if play_rate != rate or opus_rate != rate:
        raise SystemExit("sample rate drifted")
    n = min(left.size, right.size, play_l.size, play_r.size, opus_l.size, opus_r.size)
    left, right = left[:n], right[:n]
    play_l, play_r = play_l[:n], play_r[:n]
    opus_l, opus_r = opus_l[:n], opus_r[:n]
    full_ours = dial.band_snr(left, right, play_l, play_r, rate)
    full_opus = dial.band_snr(left, right, opus_l, opus_r, rate)
    quiet_ours = quiet.quiet_snr(left, right, play_l, play_r, rate)
    quiet_opus = quiet.quiet_snr(left, right, opus_l, opus_r, rate)
    full_gap = [ours - ref for ours, ref in zip(full_ours, full_opus)]
    quiet_gap = [ours - ref for ours, ref in zip(quiet_ours, quiet_opus)]
    worst = min(range(4), key=lambda index: (full_gap[index], index))
    candidate = None if full_gap[worst] >= 0 else worst
    rows = []
    for index, name in enumerate(NAMES):
        row = {
            "band": index,
            "name": name,
            "full_r16": full_ours[index],
            "full_opus": full_opus[index],
            "full_gap": full_gap[index],
            "quiet_r16": quiet_ours[index],
            "quiet_opus": quiet_opus[index],
            "quiet_gap": quiet_gap[index],
        }
        rows.append(row)
        print(
            f"{name} full {full_ours[index]:.3f} vs {full_opus[index]:.3f} gap {full_gap[index]:+.3f} "
            f"quiet {quiet_ours[index]:.3f} vs {quiet_opus[index]:.3f} gap {quiet_gap[index]:+.3f}",
            flush=True,
        )
    payload = {
        "stem": STEM,
        "rule": RULE,
        "samples": n,
        "candidate_band": candidate,
        "candidate_name": None if candidate is None else NAMES[candidate],
        "bands": rows,
    }
    path = paths.save_score("cage_trace.json", payload)
    if candidate is None:
        print("candidate none", flush=True)
    else:
        print(f"candidate {candidate} {NAMES[candidate]} gap {full_gap[candidate]:+.3f}", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
