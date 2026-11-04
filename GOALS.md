# Compression goals

Live ledger for this folder. A row changes when a pass is scored. The archive byte count is the compressed state. A playback wav is the listening copy, not the archive.

Pin AEB2AD. Updated 2026-11-04. Repository: https://github.com/dappalumbo91/fsot-residual

A row is passed only when both tests on that row are true. The table in each section is the standing. A pass paragraph keeps the status written on the day that pass was scored. A smaller file with a rejected listen stays open. A good listen that is larger than the incumbent stays open. A number that was not scored does not get written in as a win.

## Audio

These rows are the current job. The system has to beat the codec that owns that kind of sound.

### Lossy music — Opus at 96 kbps

Incumbent: Opus (RFC 6716), stereo, about 96 kbps, same decoded song. Opus at this rate is the music bar. The quality test is the ear. The size test is the `.fspz` against `results/reference/opus/{stem}_opus96.opus`.

| Song | Current file | Archive | Opus 96 | Ratio | Rate | Status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Sublime, Wrong Way | r24 | 1,213,106 | 1,327,328 | 0.914 | 86.7 vs 94.9 kbps | Size under. Ear passed the r22 render. Current file not heard. |
| Clay, 'Fore She Was Mama | r30 | 2,702,448 | 2,796,589 | 0.966 | 96.6 vs 100.0 kbps | Size under. Ear passed r22. Current file not heard. All four quiet octaves are ahead of Opus. |
| Eminem, Lose Yourself | r29 | 3,541,726 | 3,954,866 | 0.896 | 88.4 vs 98.8 kbps | Size under. Ear passed r22. Current file not heard. All four quiet octaves are ahead of Opus. |
| Bruno, Just the Way You Are | fold4 | 2,710,140 | 2,741,169 | 0.989 | 99.1 vs 100.3 kbps | Size under by 31,029. Same quotients as r26. Do not spend that margin on a rung. Ear has not heard r26. |
| Cage, Ain't No Rest | fold4 | 2,175,043 | 2,182,354 | 0.997 | 99.2 vs 99.5 kbps | Size under by 7,311. Same quotients as r34. Do not spend that margin on a rung. Quiet 2.8–5.5 kHz is still behind. Do not raise it. Do not coarsen. Ear has not heard r31–r34. |
| Skrillex, First of the Year | fold4 | 3,244,313 | 3,056,747 | 1.061 | 99.2 kbps | Size miss by 187,566. Same quotients as r16. Earlier ear had r16 in good standing. Do not coarsen. |
| VNV Nation, Beloved | fold4 | 5,144,164 | 5,058,636 | 1.017 | 93.7 kbps | Size miss by 85,528. Same quotients as r17. Vocal was still muffled and underwater. Not heard after r17. Do not coarsen. |

Quiet-slice notes are the quietest 15% of hops. Full-song averages hide that slice. Do not retune a scored rung to close a row.

The four size misses were measured on 2026-11-04 as coder floors, not as a new pass. The quotients stayed the unpacked values. Zoom stayed the stored blob. No rung was spent and no band was coarsened. Skrillex's bucket entropy plus the stored header and zoom is 3,233,571 bytes, 176,824 over Opus, before any frequency table. VNV's is 5,134,274, 75,638 over. Cage's bucket entropy plus that header is 2,167,516, 14,838 under, and the tables already stored are 29,210 bytes, so entropy plus those tables is 2,196,726, still 14,372 over. One order-0 model per octave is 2,269,463 bytes of quotient entropy alone, already over Cage's Opus file. Bruno's bucket entropy plus the stored tables is 2,740,425, 744 under Opus. The 12-bit frequency tables in that file have cross-entropy 24,513 bytes above the bucket entropy, and the reachable size with those tables kept is 2,764,938, 23,769 over. The rANS payloads sit 325 to 418 bytes above the cross-entropy of the tables they already use. SCALE stays 4096. Scores: `results/scores/lossy_floor_trace.json`, `results/scores/lossy_model_trace.json`.

Pass fold1 (`fold1.py`, magic FS2A, scored 2026-11-04) keeps those quotients, those buckets, and the zoom bytes. The 12-bit frequency table is not stored. Each symbol in the stored support starts at count 1 and steps by 1. The quotients restored exactly, so the PCM is the source archive's PCM. No wav was written. No rung was spent. Bruno 2,727,420 / 2,741,169 = 0.995× at 99.8 kbps, under by 13,749 bytes. That margin is not a rung. Cage 2,187,321 / 2,182,354 = 1.002×, still over by 4,967. Skrillex 3,263,857 / 3,056,747 = 1.068×, still over by 207,110. VNV 5,159,733 / 5,058,636 = 1.020×, still over by 101,097. Skrillex and VNV are still over the bucket-entropy floor of those same quotients. The next fold has to change the symbols. Do not coarsen. Do not raise Cage 2.8–5.5 kHz. Score: `results/scores/fold1_report.json`.

Pass fold2 (`fold2.py`, magic FS2B, scored 2026-11-04) changed that residual. The block length is `derived_D_eff("Acoustics")` from the pinned engine, which printed 8. The first bin of each block stays the quotient. Every other bin is that bin minus the parent. The tenth-pass bucket, the zoom bytes, and the fold1 range coder stayed. Quotients and zoom restored on Skrillex, VNV, and Cage. Bruno stayed at fold1. No wav. No rung. No coarsen. The archives grew. Skrillex 4,099,300 / fold1 3,263,857 / Opus 3,056,747 = 1.341× at 125.4 kbps, over by 1,042,553. VNV 6,399,910 / 5,159,733 / 5,058,636 = 1.265× at 116.5 kbps, over by 1,341,274. Cage 2,733,245 / 2,187,321 / 2,182,354 = 1.252× at 124.6 kbps, over by 550,891. The current files stay the fold1 rows above. The wrong term is this residual. Do not adopt the larger files. Do not try a second block length. Do not raise Cage 2.8–5.5 kHz. Do not coarsen. Score: `results/scores/fold2_report.json`.

Pass fold3 (`fold3.py`, magic FS2C, scored 2026-11-04) keeps the quotients and changes the residual to the half-cycle of the scored MDCT grid. Hop equals the transform length, 256, so two hops advance every bin center by k+0.5 cycles. The stored symbol is q[t] + q[t-2], with frames 0 and 1 raw. The tenth-pass bucket reads q[t-2]. Zoom bytes are copied. Quotients restored on Skrillex, VNV, and Cage. Bruno stayed at fold1. No wav. No rung. No coarsen. The archives still grew. Skrillex 3,319,516 / fold1 3,263,857 / Opus 3,056,747 = 1.086× at 101.5 kbps, over by 262,769. VNV 5,249,242 / 5,159,733 / 5,058,636 = 1.038× at 95.6 kbps, over by 190,606. Cage 2,229,675 / 2,187,321 / 2,182,354 = 1.022× at 101.6 kbps, over by 47,321. The current files stay fold1. The wrong term is this half-cycle residual: the quotients are not locked to the bin center. Do not adopt the larger files. Do not take a four-frame step from this miss. Do not try a second block length. Do not raise Cage 2.8–5.5 kHz. Do not coarsen. Score: `results/scores/fold3_report.json`.

Pass formula (`formula.py`, magic FS2E, scored 2026-11-04) is a different object. Each octave keeps both edges and every strict local extremum. The decoder draws the straight chord between those knots, rounds half away from zero, and renders with the inverse MDCT. Zoom bytes are copied. Bruno stayed at fold1. No wav. No rung. The chord does not reproduce the stored wave. Skrillex 6,107,742 / fold1 3,263,857 / Opus 3,056,747 = 1.998× at 186.8 kbps, over by 3,050,995, SNR 10.92 dB against the stored wave, knots 0.263. VNV 9,090,901 / 5,159,733 / 5,058,636 = 1.797× at 165.6 kbps, over by 4,032,265, SNR 11.17 dB, knots 0.228. Cage 4,306,015 / 2,187,321 / 2,182,354 = 1.973× at 196.3 kbps, over by 2,123,661, SNR 11.13 dB, knots 0.282. The current files stay fold1. The straight chord is not the law of these spectra. Do not adopt the larger files. Do not retune the chord. Do not raise Cage 2.8–5.5 kHz. Do not coarsen. Score: `results/scores/formula_report.json`.

Pass fold4 (`fold4.py`, magic FS2F, scored 2026-11-04) keeps the fold1 quotients, payloads, buckets, and zoom bytes. The stored alphabet was a sorted int16 list. It is now the zigzag of the first symbol and the positive gaps, each as an Elias gamma code. One code, no parameter. Quotients restored, so the PCM is unchanged. No wav. No rung. Bruno 2,710,140 / 2,741,169 = 0.989× at 99.1 kbps, under by 31,029. That margin is not a rung. Cage 2,175,043 / 2,182,354 = 0.997× at 99.2 kbps, under by 7,311. That margin is not a rung. Quiet 2.8–5.5 kHz is still behind. Do not raise it. Skrillex 3,244,313 / 3,056,747 = 1.061× at 99.2 kbps, still over by 187,566. VNV 5,144,164 / 5,058,636 = 1.017× at 93.7 kbps, still over by 85,528. Skrillex and VNV remain over the bucket-entropy floor of these quotients. Do not coarsen. Score: `results/scores/fold4_report.json`.

### Lossy speech — Opus SILK

Incumbent: Opus in its speech configuration. RFC 6716 sweet spots at a 20 ms frame are 8–12 kbps narrowband, 16–20 kbps wideband, and 28–40 kbps fullband speech. SILK is the linear-prediction layer Opus uses for voice. Hybrid mode puts SILK under 8 kHz and the music transform above it.

Status: not started.

A listing one folder deep on G: found no speech, podcast, or audiobook directory. A song is not a speech test. The first pass needs a real voice recording, an Opus encode at one of those sweet spots on that same recording, and a contract written before the score. The music MDCT ladder is not that contract.

### Lossless music — FLAC

Incumbent: FLAC level 8 on the same decoded song (`results/scores/codec_sizes.json`). The test is byte-exact restore and a smaller archive than FLAC.

The scored exact grade is the Q=1 archive in that file. It is larger than FLAC on every song.

| Song | FLAC level 8 | Exact archive | Exact / FLAC |
| --- | ---: | ---: | ---: |
| Sublime, Wrong Way | 10,659,386 | 20,631,098 | 1.94 |
| Clay, 'Fore She Was Mama | 23,262,265 | 39,512,150 | 1.70 |
| Eminem, Lose Yourself | 29,262,719 | 58,717,804 | 2.01 |
| Bruno, Just the Way You Are | 25,589,412 | 42,026,921 | 1.64 |
| Cage, Ain't No Rest | 22,268,030 | 35,316,646 | 1.59 |
| Skrillex, First of the Year | 36,081,830 | 54,243,191 | 1.50 |
| VNV Nation, Beloved | 49,466,148 | 86,270,876 | 1.74 |

The Haar table above is the older exact grade. Pass flac1 (`flac1.py`, magic FSPJ, scored 2026-11-04) named one predictor before the run: reversible mid/side, one first difference, zigzag into uint32, four little-endian byte planes, each plane the existing static order-0 rANS. No second difference and no LPC. Restore is the source PCM on all seven songs. Every archive is still larger than FLAC. The low byte of the difference is essentially uncompressed, and the top two bytes collapse. That miss stays. Do not fit a coefficient from it.

| Song | FLAC level 8 | flac1 | flac1 / FLAC | flac1 / Haar Q=1 |
| --- | ---: | ---: | ---: | ---: |
| Sublime, Wrong Way | 10,659,386 | 14,419,861 | 1.353 | 0.699 |
| Skrillex, First of the Year | 36,081,830 | 39,629,316 | 1.098 | 0.731 |
| Bruno, Just the Way You Are | 25,589,412 | 30,022,195 | 1.173 | 0.714 |
| Eminem, Lose Yourself | 29,262,719 | 42,630,675 | 1.457 | 0.726 |
| VNV Nation, Beloved | 49,466,148 | 62,508,804 | 1.264 | 0.725 |
| Clay, 'Fore She Was Mama | 23,262,265 | 28,702,776 | 1.234 | 0.726 |
| Cage, Ain't No Rest | 22,268,030 | 25,109,204 | 1.128 | 0.711 |

The byte trace before flac2 showed that ideal order-1 coding of the flac1 residual is still over FLAC on every song (1.067× to 1.344×). A context coder of that residual cannot reach FLAC. Pass flac2 (`flac2.py`, magic FS2D, scored 2026-11-04) replaced the first difference with one fixed second difference, `x[n] - 2*x[n-1] + x[n-2]`, and no fitted coefficient. Restore is the source PCM on all seven songs. Every archive is still larger than FLAC. The low byte is still essentially uncompressed. Skrillex's flac2 file is larger than its flac1 file. The other six are smaller than flac1 and still over FLAC. Do not score a third difference or an LPC fit from this miss.

| Song | FLAC level 8 | flac2 | flac2 / FLAC | flac1 / FLAC |
| --- | ---: | ---: | ---: | ---: |
| Sublime, Wrong Way | 10,659,386 | 13,353,846 | 1.253 | 1.353 |
| Skrillex, First of the Year | 36,081,830 | 40,006,807 | 1.109 | 1.098 |
| Bruno, Just the Way You Are | 25,589,412 | 29,186,547 | 1.141 | 1.173 |
| Eminem, Lose Yourself | 29,262,719 | 40,923,743 | 1.398 | 1.457 |
| VNV Nation, Beloved | 49,466,148 | 62,246,055 | 1.258 | 1.264 |
| Clay, 'Fore She Was Mama | 23,262,265 | 27,998,202 | 1.204 | 1.234 |
| Cage, Ain't No Rest | 22,268,030 | 24,809,255 | 1.114 | 1.128 |

The integer trace (`trace_flac_int.py`, 2026-11-04, no archive) measured the order-0 entropy of the zigzag uint32 itself. That is the joint entropy of the four byte planes. Ideal bytes versus FLAC level 8: Sublime 13,334,235 = 1.251, Skrillex 39,770,415 = 1.102, Bruno 29,067,457 = 1.136, Eminem 40,524,378 = 1.385, VNV 61,908,874 = 1.252, Clay 27,718,188 = 1.192, Cage 24,685,869 = 1.109. The byte-plane sum is only slightly higher. Every ideal is over FLAC. No coder of this second-difference residual can pass. Do not pack one. Do not score a third difference or an LPC fit from this measurement. Score: `results/scores/flac_int_trace.json`.

Status: behind FLAC on every song. Ahead of the Haar exact grade on every song. The lossy listening grade is not a FLAC result.

### Other lossy music codecs

Opus 128 references exist beside the 96 kbps files. Opus 96 is the bar. A file that is under Opus 96 is also under the larger Opus 128 file. That is not a separate win.

The commercial source mp3 is the file on the disc. Its bitrate is not 96 kbps. Landing under that mp3 is not beating a codec. Bytes are in `codec_sizes.json` under `source_mp3`.

AAC and Vorbis were measured on 2026-11-03 at a nominal 96 kbps, audio stream only. The encoder is ffmpeg's native AAC and libvorbis. Files are in `results/reference/aac/` and `results/reference/vorbis/`. The score is `results/scores/codec_match.json`. This is size only. The ear has not compared these encodes.

Vorbis at that setting spent less than 96 kbps on several songs. Lose Yourself's Vorbis file is 83.6 kbps. A larger archive than that file is a size miss against Vorbis even when the same archive is under Opus and AAC.

| Song | Archive at this measurement | AAC 96 | Vorbis 96 | Against AAC | Against Vorbis |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sublime, Wrong Way | 1,213,106 | 1,371,257 | 1,293,243 | 0.885 | 0.938 |
| Clay, 'Fore She Was Mama | 2,702,448 | 2,730,558 | 2,489,438 | 0.990 | 1.085 |
| Eminem, Lose Yourself | 3,541,726 | 3,981,310 | 3,347,650 | 0.890 | 1.058 |
| Cage, Ain't No Rest | 2,204,742 | 2,175,066 | 2,176,585 | 1.014 | 1.013 |
| Bruno, Just the Way You Are | 2,765,356 | 2,698,825 | 2,540,818 | 1.025 | 1.088 |
| Skrillex, First of the Year | 3,296,457 | 3,228,551 | 3,102,569 | 1.021 | 1.063 |
| VNV Nation, Beloved | 5,202,179 | 5,375,507 | 4,962,989 | 0.968 | 1.048 |

This comparison was measured on 2026-11-03, before fold4. Bruno's row is the r26 archive. Cage's row is the r34 archive. fold4 later changed those two byte counts and was not run against AAC or Vorbis again. Bruno's last under-Opus archive, r24 at 2,681,526, is also under this AAC file (0.994×) and over this Vorbis file. Cage r34 is over Opus, AAC, and Vorbis. The last Cage file under Opus is r33 at 2,181,324. The last Cage file under both AAC and Vorbis is r32 at 1,986,213 (0.913×). Quality against AAC and Vorbis is not signed.

## Other information

These are the goals outside sound. None of them is passed. The stored state stays phase, gauge, and residual. The residual changes with the kind of information. One music transform does not carry these rows.

### Programs, source, logs, and tables

Incumbent: zstd. The first bench compressed project prose from 22,608 bytes to 19,238. zlib level 9 did the same prose in 9,405. Already-coded mp3 and avi containers barely moved.

Status: behind zlib. zstd has not been run as the official bar on that prose yet. The byte coder in this folder is static order-0 rANS. It has no repeated-string match.

Pass test: a smaller archive than zstd on text, source, and tables, with byte-exact restore. Random bytes and files that are already compressed stay at their entropy. A result under that entropy is not a goal.

### Still images

Incumbent: JPEG XL and AVIF. The probe on disk is 24 frames of Getaway, raw YUV 6,266,880 bytes. The exact Q=1 archive was 2,030,877 bytes. Q=16 showed visible blocks. That probe is not a still-image codec.

Status: probe only. The first real pass needs one still image, a JPEG XL or AVIF file at a declared quality, and a contract written before the score.

### Video

Incumbent: AV1, x265, and x264. The 8 second South Park probe at Q=8 is 4,164,451 bytes against an x264 crf 18 video-only file of 611,547 bytes, about 6.8×. The full episode encode was stopped and the archives were deleted. `deliver.py full` stays stopped.

Status: behind. The missing term is motion. Do not restart the episode to improve this row. A motion contract is written before any new episode run.

### Model weights

Incumbent: the quantization-plus-entropy tools used on real weight files. The test is the model's own loss on a held-out task, plus the archive size. An audio signal-to-noise number is the wrong score.

Status: not started. No weight file is in this project.

## Rules that stay in force

- The engine pin stays AEB2AD. Do not edit `vendor/fsot_compute.py`.
- No new free parameter after a score. A failed pass stays failed. Name the term, then write the next contract.
- The Opus margin is not a budget for choosing which band to spend.
- Shannon's limit still binds anything the file does not repeat.
- Do not publish the media.

## Listening copies

| Song | Wav |
| --- | --- |
| Sublime, Wrong Way | `results/current/r24/sublime_wrong_way_q2048.wav` |
| Clay, 'Fore She Was Mama | `results/current/r30/clay_walker_fore_she_was_mama_q2048.wav` |
| Eminem, Lose Yourself | `results/current/r29/eminem_lose_yourself_q2048.wav` |
| Bruno, Just the Way You Are | `results/current/r26/bruno_mars_just_the_way_you_are_q2048.wav` |
| Cage, Ain't No Rest | `results/current/r34/cage_the_elephant_aint_no_rest_q2048.wav` |
| Skrillex, First of the Year | `results/current/r16/skrillex_first_of_the_year_q2048.wav` |
| VNV Nation, Beloved | `results/current/r17/vnv_nation_beloved_q2048.wav` |

## Log

- 2026-11-03. Ledger opened. Audio rows filled from scored passes r16, r17, r24, r25, r26, and r27. FLAC row filled from `codec_sizes.json`. Video row filled from the 8 second probe file sizes. Speech, AAC, Vorbis, zstd-as-bar, stills, and weights are open.
- 2026-11-03. r28 raised Lose Yourself's quiet bass one rung. Archive 3,512,485 / 3,954,866 = 0.888× at 87.7 kbps. Quiet 0–2.8 kHz moved to 23.2 dB against Opus 21.0. Do not raise that band again from this score.
- 2026-11-03. r29 raised the quiet 2.8–5.5 kHz octave on Lose Yourself and Clay. Lose Yourself 3,541,726 / 3,954,866 = 0.896× at 88.4 kbps. That quiet band moved to 12.6 dB against Opus 11.1. All four quiet octaves on Lose Yourself are ahead. Clay in that pass was 2,684,004 bytes. Do not raise Lose Yourself's 2.8–5.5 kHz band again from this score.
- 2026-11-03. r30 raised Clay's quiet 11–22 kHz octave. Archive 2,702,448 / 2,796,589 = 0.966× at 96.6 kbps. That band moved to 4.1 dB against Opus 3.0. All four quiet octaves on Clay are ahead. Do not raise that band again from this score.
- 2026-11-03. AAC and Vorbis nominal 96 kbps measured. Size only. No speech folder one level down on G:.
- 2026-11-04. flac1 stored a reversible mid/side first difference as zigzag uint32 byte planes in order-0 rANS. All seven songs restored the source PCM and all seven are larger than FLAC level 8: 1.353, 1.098, 1.173, 1.457, 1.264, 1.234, and 1.128. Each is smaller than the Haar exact grade. The low byte of the difference did not compress. Do not fit LPC from this miss. Score: `results/scores/flac1_report.json`.
- 2026-11-04. Cage was traced before any new rung. The full-song octave furthest behind Opus was 5.5–11 kHz, −3.134 dB. r31 raised that band. Archive 1,970,312 / 2,182,354 = 0.903×. The band moved to 7.51 against Opus 6.80 and is ahead. r32 then raised the only remaining full-song miss, 11–22 kHz, by one rung. Archive 1,986,213 = 0.910×. That band moved from 0.05 dB to 0.53 against Opus 1.88 and stayed behind. A peak count on that archive showed 2,452/30,233 mid frames and 2,582/30,233 side frames keeping the peak at the stored rung. r33 set every frame of 11–22 kHz to the coarsest rung that keeps the peak. Archive 2,181,324 / 2,182,354 = 0.9995× at 99.4 kbps, under by 1,030 bytes. The band is 1.25 against Opus 1.88, still behind. Raised 26,945/30,233 frames. Do not run that peak rule again, and do not coarsen another band. Scores: `cage_trace.json`, `cage_rung_report.json`, `above_report.json`, `cage_peak_trace.json`, `cage_peak_report.json`.
- 2026-11-04. The flac1 low byte was traced. Ideal order-1 size of that residual is 1.067× to 1.344× FLAC, so a context coder cannot pass. flac2 stored one fixed second difference. All seven songs restored the PCM and all seven lost to FLAC: 1.253, 1.109, 1.141, 1.398, 1.258, 1.204, 1.114. The low byte stayed near 8 bits. Skrillex grew versus flac1. Do not fit LPC and do not score a third difference from this miss. Score: `results/scores/flac2_report.json`.
- 2026-11-04. Cage's worst quiet octave on r33 was 2.8–5.5 kHz, −4.390 dB. The next rung already kept the peak on 3,182/5,266 quiet mid frames, so r34 raised that band one rung on quiet frames only. Archive 2,204,742 / 2,182,354 = 1.010×, a size miss by 22,388 bytes. Quiet 2.8–5.5 moved from 5.17 dB to 8.56 against Opus 9.56 and is still behind. Do not raise that band again from this score and do not coarsen another band. The last under-Opus Cage file is r33. Score: `results/scores/cage_quiet_report.json`. No speech folder turned up under the music, text, linguistics, or dictionary directories. A song is not a SILK test. Bruno, Skrillex, and VNV were not coarsened and were not given another rung.
- 2026-11-04. The flac2 zigzag integer was traced. Its order-0 ideal is over FLAC on every song: 1.251, 1.102, 1.136, 1.385, 1.252, 1.192, 1.109. No coder of that residual can pass. Do not score a third difference or an LPC fit from this measurement. Score: `results/scores/flac_int_trace.json`.
- 2026-11-04. The four Opus size misses were traced as coder floors. Skrillex and VNV are over Opus at the bucket entropy of the stored quotients. Cage's bucket entropy is 14,838 under and the stored tables put it back over. Bruno's bucket entropy plus those tables is 744 under, and the 12-bit frequency tables raise the reachable size to 23,769 over. The rANS payloads are 325 to 418 bytes above the cross-entropy of the tables already stored. SCALE stays 4096. No archive was written. Scores: `results/scores/lossy_floor_trace.json`, `results/scores/lossy_model_trace.json`.
- 2026-11-04. fold1 dropped the 12-bit frequency table and coded the same quotients with counts that start at 1. Quotients and zoom bytes matched. Bruno 2,727,420 is under Opus by 13,749. Cage is still over by 4,967. Skrillex is still over by 207,110. VNV is still over by 101,097. Do not spend Bruno's margin on a rung. Do not coarsen. Score: `results/scores/fold1_report.json`.
- 2026-11-04. fold2 stored each child bin as the integer error against the block parent. Block length is derived_D_eff(Acoustics) = 8. Quotients restored. Skrillex 4,099,300, VNV 6,399,910, and Cage 2,733,245 are larger than fold1 and over Opus by 1,042,553, 1,341,274, and 550,891. The current files stay fold1. Do not adopt these archives. Do not try a second block length. Score: `results/scores/fold2_report.json`.
- 2026-11-04. fold3 stored the half-cycle residual q[t] + q[t-2] on the 256-hop MDCT grid. Quotients restored. Skrillex 3,319,516, VNV 5,249,242, and Cage 2,229,675 are larger than fold1 and over Opus by 262,769, 190,606, and 47,321. The current files stay fold1. Do not adopt these archives. Do not take a four-frame step. Score: `results/scores/fold3_report.json`.
- 2026-11-04. The formula pass stored each octave's edges and strict local extrema and rendered the straight chord through the inverse MDCT. Skrillex 6,107,742, VNV 9,090,901, and Cage 4,306,015 are about twice Opus. SNR against the stored wave is 10.92, 11.17, and 11.13 dB. Knots are 0.263, 0.228, and 0.282 of the bins. The current files stay fold1. Do not adopt these archives. Score: `results/scores/formula_report.json`.
- 2026-11-04. Entropy-metric trace. The stored object was one float64, the byte entropy of the fold1 archive, and not the archive. Bruno, Cage, Skrillex, and VNV measure 7.998693, 7.998850, 7.998921, and 7.999645 bits per byte. The metric is 8 bytes. The ideal length is the file length. A permutation keeps the metric and is a different file. The preimage is the size of the archive. Opus 96 sits at about 7.997 bits per byte on the same four songs. No archive. Score: `results/scores/entropy_metric_trace.json`.
- 2026-11-04. fold4 stored each sorted alphabet as Elias gamma gaps and copied the fold1 payloads. Quotients matched. Bruno 2,710,140 is under Opus by 31,029. Cage 2,175,043 is under by 7,311. Skrillex is still over by 187,566. VNV is still over by 85,528. Do not spend either new margin on a rung. Do not coarsen. Score: `results/scores/fold4_report.json`.
