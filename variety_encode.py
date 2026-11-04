#!/usr/bin/env python3
"""Same refine rule as refine_audio.py, on a wider set of songs.

No step, lag bound, or multiplier changes. The listening note on
cries_in_the_beat at Q=2048 stays a note: the lag can impose a high
pitched residue. These songs are here so that residue can be heard on
more than one kind of music. Scores go to results/scores. A playback file
is kept only at the listening grade, under results/current/variety.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deliver
import fspack
import paths
import refine_audio as r2
SONGS = [
    Path(r"G:\Music discography's\Sublime\At The House Of Blues\Sublime_At The House Of Blues__02-wrongway.mp3"),
    Path(r"G:\Music discography's\Skrillex\01 First of the Year (Equinox).mp3"),
    Path(r"G:\Music discography's\Bruno Mars - Discography\01 - Just The Way You Are.mp3"),
    Path(r"G:\Music discography's\eminem\Eminem - 8 Mile [2002]\01 - Lose Yourself - Eminem.mp3"),
    Path(r"G:\Music discography's\VNV Nation\VNV Nation - Beloved.mp3"),
    Path(r"G:\Music discography's\Clay Walker (Discography)\01- 'Fore She Was Mama.mp3"),
    Path(r"G:\Music discography's\Cage The Elephant\Cage the Elephant (2008)\03 Ain't No Rest For The Wicked.mp3"),
]
STEMS = [
    "sublime_wrong_way",
    "skrillex_first_of_the_year",
    "bruno_mars_just_the_way_you_are",
    "eminem_lose_yourself",
    "vnv_nation_beloved",
    "clay_walker_fore_she_was_mama",
    "cage_the_elephant_aint_no_rest",
]


def encode_one(path: Path, stem: str) -> dict:
    print(f"song {stem}", flush=True)
    if not path.exists():
        raise SystemExit(f"missing {path}")
    left, right, rate = deliver.decode_pcm(path)
    mid, side = deliver.mid_side(left, right)
    mid_bands, original = deliver.haar_split(mid, 3)
    side_bands, original_s = deliver.haar_split(side, 3)
    if original != original_s:
        raise SystemExit(f"{stem} channel lengths differ")
    rows = []
    for q in deliver.AUDIO_Q:
        print(f"  Q={q}", flush=True)
        mid_d, mid_a, mid_lags, _mh, _mr = r2.encode_channel(mid_bands, q)
        side_d, side_a, side_lags, _sh, _sr = r2.encode_channel(side_bands, q)
        blob = r2.pack(mid_d, mid_a, side_d, side_a, mid_lags, side_lags)
        back_md, back_ma, back_sd, back_sa, back_ml, back_sl = r2.unpack(blob)
        mid_hats = []
        side_hats = []
        for codes, hats in ((back_md, mid_hats), (back_sd, side_hats)):
            for code, times in zip(codes, r2.REFINE_TIMES[:3]):
                step = 1 if q <= 1 else q * times
                hats.append(code if step <= 1 else code * step)
        approx_step = 1 if q <= 1 else q * r2.REFINE_TIMES[3]
        mid_hats.append(r2.restore_predicted(back_ma, back_ml, approx_step))
        side_hats.append(r2.restore_predicted(back_sa, back_sl, approx_step))
        got_l, got_r = deliver.unmid_side(
            deliver.haar_join(mid_hats, original),
            deliver.haar_join(side_hats, original),
        )
        got_l = deliver.clip_i16(got_l).astype(np.int32)
        got_r = deliver.clip_i16(got_r).astype(np.int32)
        if q == 1 and not (np.array_equal(got_l, left) and np.array_equal(got_r, right)):
            raise SystemExit(f"{stem} Q=1 is not sample-exact")
        err_l = got_l.astype(np.float64) - left
        err_r = got_r.astype(np.float64) - right
        noise = float(np.mean(np.concatenate([err_l, err_r]) ** 2))
        signal = float(np.mean(np.concatenate([left.astype(np.float64), right.astype(np.float64)]) ** 2))
        snr = math.inf if noise == 0 else 10 * math.log10(signal / noise)
        level = 0.0 if noise == 0 else 20 * math.log10(
            math.sqrt(float(np.mean(np.concatenate([got_l, got_r]).astype(np.float64) ** 2))) / math.sqrt(signal)
        )
        kept = None
        if q == paths.LISTEN_Q:
            kept = OUT / f"{stem}_q{q}.wav"
            deliver.write_wav(kept, got_l, got_r, rate)
            (OUT / f"{stem}_q{q}.fspz").write_bytes(blob)
        row = {
            "q": q,
            "archive_bytes": len(blob),
            "ratio_vs_source": len(blob) / path.stat().st_size,
            "snr_db": None if math.isinf(snr) else snr,
            "level_delta_db": level,
            "exact": bool(noise == 0),
            "wav": None if kept is None else str(kept),
        }
        rows.append(row)
        print(
            f"    archive {row['archive_bytes']} vs source {path.stat().st_size} "
            f"ratio {row['ratio_vs_source']:.3f} level {level:.3f} dB "
            f"{'exact' if noise == 0 else f'SNR {snr:.2f}'}",
            flush=True,
        )
    return {
        "stem": stem,
        "artist_note": stem.split("_", 1)[0],
        "source": str(path),
        "source_bytes": path.stat().st_size,
        "seconds": original / rate,
        "ladder": rows,
    }


def main() -> None:
    global OUT
    OUT = paths.begin_pass("variety")
    songs = []
    for path, stem in zip(SONGS, STEMS):
        songs.append(encode_one(path, stem))
        report = {
            "pin": fspack.PIN,
            "rule": "refine_audio r2, unchanged",
            "listening_note": "Q=2048 on cries_in_the_beat added a high pitched residue. Same rule, new songs.",
            "listen_q": paths.LISTEN_Q,
            "songs": songs,
        }
        paths.save_score("variety_report.json", report)
    print("variety done", flush=True)


if __name__ == "__main__":
    main()
