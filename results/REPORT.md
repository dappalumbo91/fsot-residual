# FSPACK bench

Engine pin AEB2AD, sha256 `aeb2adad6e80f487772c5df90a2e3dda71624ab831a6a83b94ab471ac9aac170`.
K = 0.420108763650. POOF/(POOF+SUCTION) = 0.510728588774.
That valve is the pump fraction from the engine. It is not the compression ratio of these files.

Q was fixed as 1, 4, 8, 16 for pictures and 1, 16, 64, 256 for audio before the files were scored.
Byte archives were checked by sha256. Q=1 picture and audio restores were checked against the decoded samples.

## Containers

These inputs are already encoded, or they are prose. The ratio is packed size divided by original size.

| File | Bytes | FSPACK | zlib -9 | FSPACK ratio | zlib ratio | D_eff | Phase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| direction_md | 22608 | 19238 | 9405 | 0.851 | 0.416 | 14 | free |
| sample_avi_container | 4937728 | 4929661 | 4889063 | 0.998 | 0.990 | 25 | free |
| cries_mp3_container | 4227964 | 4224173 | 4218768 | 0.999 | 0.998 | 25 | free |

## Video

Source `G:\Cinema and movies and shows\movies\Getaway [2013] HDRip XViD-ETRG\sample.avi`.
Decoded 24 frames, 640x272, mpeg4, 23.97602 fps.
Raw yuv420p 6266880 bytes. zlib -9 on that raw stream: 2122626 bytes.
PSNR and SSIM compare the FSPACK restore with those decoded frames. The 8x8 SSIM is the mean of non-overlapping blocks on Y.

| Q | FSPACK bytes | Ratio vs raw | PSNR Y | PSNR U | PSNR V | SSIM Y | Max abs Y |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2030877 | 0.324 | exact | exact | exact | 1.0000 | 0 |
| 4 | 1419122 | 0.226 | 46.31 | 46.43 | 45.90 | 0.9835 | 2 |
| 8 | 922670 | 0.147 | 40.77 | 40.92 | 39.98 | 0.9578 | 4 |
| 16 | 547574 | 0.087 | 34.89 | 35.73 | 36.29 | 0.9078 | 8 |

Contact sheet, source then Q=1, 4, 8, 16: `C:\Users\damia\Desktop\Information compressor fsot\results\getaway_contact.png`.
Q=8 viewing file, stored with x264 qp 0 so the wrap does not add a second lossy step: `C:\Users\damia\Desktop\Information compressor fsot\results\getaway_q8.mp4`.

## Audio

Source `G:\Music discography's\music i made\cries_in_the_beat.mp3`.
5 seconds, 16000 Hz, mono, s16le. zlib -9 on that pcm: 156456 bytes.

| Q | FSPACK bytes | Ratio vs raw | SNR dB | Max abs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 143202 | 0.895 | exact | 0 |
| 16 | 96721 | 0.605 | 66.24 | 8 |
| 64 | 65563 | 0.410 | 54.25 | 32 |
| 256 | 45294 | 0.283 | 42.17 | 128 |

Exact restore: `C:\Users\damia\Desktop\Information compressor fsot\results\cries_q1.wav`.
Q=64 restore: `C:\Users\damia\Desktop\Information compressor fsot\results\cries_q64.wav`.
