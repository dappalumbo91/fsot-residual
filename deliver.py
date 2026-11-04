#!/usr/bin/env python3
"""Long-form FSPACK delivery.

The first bench predicted each pixel from the pixel to its left and packed the
residual with an order-0 coder. The decoder of frame N already holds frame N-1.
This file uses that frame as the prediction, which is the library in the
Slepian-Wolf sense. Audio uses a reversible mid/side split and a 3-level
integer Haar split. Fine ripples take a larger declared step than the coarse
band. Steps are fixed here, before any file is scored.

Chroma step is 2Q when Q > 1, and 1 when Q is 1.
Haar multipliers, finest detail first, then the approximation: 4, 2, 2, 1.

The residual bytes are packed with zlib -9. The order-0 rANS coder from
fspack.py already lost to zlib on prose, and it is too slow for a full episode
of residual. The comparison against FLAC and x264 is about this residual.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
import time
import wave
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fspack
import paths

ROOT = Path(__file__).resolve().parent
WATCH = ROOT / "results" / "watch"
SONGS = [
    Path(r"G:\Music discography's\music i made\cries_in_the_beat.mp3"),
    Path(r"G:\Music discography's\music i made\human_in_slaved_buy_robots.mp3"),
]
SHOW_ROOT = Path(r"G:\Cinema and movies and shows\showes\South Park\South Park Season 1")
AUDIO_Q = (1, 32, 128, 512, 2048)
VIDEO_Q = (1, 4, 8, 16)
HAAR_TIMES = (4, 2, 2, 1)  # finest detail, next, next, approximation
CHROMA_TIMES = 2
CHUNK_FRAMES = 15
HEADER_ROOM = 8192


class ArchiveStream:
    """FSPZ file. The header occupies a fixed room so chunks can be appended once.

    Layout: magic FSPZ, uint32 header length, header padded with spaces, then
    uint32 chunk count and repeated uint32 length + zlib bytes. The payload
    hash covers the count and the chunks. It does not cover the header.
    """

    def __init__(self, path: Path):
        self.path = path
        self.fh = path.open("w+b")
        self.payload_at = 8 + HEADER_ROOM
        self.fh.write(b"\x00" * (self.payload_at + 4))
        self.count = 0
        self.chunk_bytes = 0

    def add(self, blob: bytes) -> None:
        self.fh.write(struct.pack("<I", len(blob)))
        self.fh.write(blob)
        self.count += 1
        self.chunk_bytes += len(blob)

    def finish(self, meta: dict) -> None:
        end = self.fh.tell()
        self.fh.seek(self.payload_at)
        self.fh.write(struct.pack("<I", self.count))
        self.fh.flush()
        hasher = hashlib.sha256()
        self.fh.seek(self.payload_at)
        remaining = end - self.payload_at
        while remaining:
            block = self.fh.read(min(1 << 20, remaining))
            if not block:
                raise RuntimeError("archive payload ended early")
            hasher.update(block)
            remaining -= len(block)
        stored = dict(meta)
        stored["payload_sha256"] = hasher.hexdigest()
        stored["chunk_count"] = self.count
        header = json.dumps(stored, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(header) > HEADER_ROOM:
            raise RuntimeError("archive header exceeds the reserved room")
        padded = header + (b" " * (HEADER_ROOM - len(header)))
        self.fh.seek(0)
        self.fh.write(b"FSPZ")
        self.fh.write(struct.pack("<I", HEADER_ROOM))
        self.fh.write(padded)
        self.fh.flush()
        self.fh.close()


def payload_sha256(chunks: list[bytes]) -> str:
    hasher = hashlib.sha256()
    hasher.update(struct.pack("<I", len(chunks)))
    for blob in chunks:
        hasher.update(struct.pack("<I", len(blob)))
        hasher.update(blob)
    return hasher.hexdigest()


def read_first_chunk(path: Path) -> bytes:
    with path.open("rb") as handle:
        handle.seek(8 + HEADER_ROOM)
        handle.read(4)
        (size,) = struct.unpack("<I", handle.read(4))
        blob = handle.read(size)
    if len(blob) != size:
        raise RuntimeError("short chunk on disk")
    return blob


def read_fspz_header(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic != b"FSPZ":
            raise RuntimeError(f"bad archive magic {magic!r}")
        (size,) = struct.unpack("<I", handle.read(4))
        return json.loads(handle.read(size))


def stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.kill()
    proc.wait()


def show_path() -> Path:
    matches = sorted(SHOW_ROOT.glob("S01E01*/*.avi"))
    if not matches:
        raise SystemExit(f"no S01E01 avi under {SHOW_ROOT}")
    return matches[0]


def step_of(resid: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    resid = resid.astype(np.int32, copy=False)
    if q <= 1:
        qq = resid
        hat = resid
    else:
        qq = np.sign(resid) * ((np.abs(resid) + (q // 2)) // q)
        hat = qq * q
    return qq.astype(np.int32), hat.astype(np.int32)


def haar_split(samples: np.ndarray, levels: int = 3) -> tuple[list[np.ndarray], int]:
    original = int(samples.size)
    width = 1 << levels
    pad = (width - (original % width)) % width
    data = samples.astype(np.int32, copy=True)
    if pad:
        data = np.pad(data, (0, pad), mode="edge")
    bands: list[np.ndarray] = []
    approx = data
    for _ in range(levels):
        even = approx[0::2]
        odd = approx[1::2]
        detail = odd - even
        approx = even + (detail // 2)
        bands.append(detail)
    bands.append(approx)
    return bands, original


def haar_join(bands: list[np.ndarray], original: int) -> np.ndarray:
    approx = bands[-1].astype(np.int32, copy=False)
    for detail in reversed(bands[:-1]):
        detail = detail.astype(np.int32, copy=False)
        even = approx - (detail // 2)
        odd = even + detail
        mixed = np.empty(even.size * 2, dtype=np.int32)
        mixed[0::2] = even
        mixed[1::2] = odd
        approx = mixed
    return approx[:original]


def mid_side(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    side = left.astype(np.int32) - right.astype(np.int32)
    mid = right.astype(np.int32) + (side // 2)
    return mid, side


def unmid_side(mid: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = mid.astype(np.int32) - (side.astype(np.int32) // 2)
    left = right + side.astype(np.int32)
    return left, right


def quantize_bands(bands: list[np.ndarray], q: int) -> list[np.ndarray]:
    if len(bands) != len(HAAR_TIMES):
        raise RuntimeError("Haar band count does not match the declared multipliers")
    out = []
    for band, times in zip(bands, HAAR_TIMES):
        step = 1 if q <= 1 else q * times
        qq, _hat = step_of(band, step)
        out.append(qq)
    return out


def restore_bands(quants: list[np.ndarray], q: int, original: int) -> np.ndarray:
    restored = []
    for qq, times in zip(quants, HAAR_TIMES):
        step = 1 if q <= 1 else q * times
        restored.append(qq.astype(np.int32) if step == 1 else qq.astype(np.int32) * step)
    return haar_join(restored, original)


def clip_i16(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -32768, 32767).astype(np.int16)


def pack_planes(planes: list[np.ndarray], dtype: str) -> bytes:
    width = {"<i2": 2, "<i4": 4}[dtype]
    np_dtype = np.dtype(dtype)
    info = np.iinfo(np_dtype)
    blob = bytearray()
    blob += struct.pack("<HB", len(planes), width)
    for plane in planes:
        data = np.ascontiguousarray(plane)
        if int(data.min()) < info.min or int(data.max()) > info.max:
            raise RuntimeError(f"residual {int(data.min())}..{int(data.max())} does not fit {dtype}")
        raw = data.astype(np_dtype).tobytes()
        blob += struct.pack("<I", plane.size)
        blob += raw
    return zlib.compress(bytes(blob), 9)


def unpack_planes(blob: bytes) -> list[np.ndarray]:
    raw = zlib.decompress(blob)
    count, width = struct.unpack_from("<HB", raw, 0)
    pos = 3
    dtype = "<i2" if width == 2 else "<i4"
    planes = []
    for _ in range(count):
        (size,) = struct.unpack_from("<I", raw, pos)
        pos += 4
        byte_len = size * width
        plane = np.frombuffer(raw[pos:pos + byte_len], dtype=dtype).astype(np.int32)
        pos += byte_len
        planes.append(plane.copy())
    if pos != len(raw):
        raise ValueError("trailing plane bytes")
    return planes


def self_check() -> None:
    rng = np.random.default_rng(7)
    samples = rng.integers(-20000, 20000, size=8000, dtype=np.int32)
    bands, n = haar_split(samples, 3)
    back = haar_join(bands, n)
    if not np.array_equal(back, samples):
        raise SystemExit("Haar split is not reversible")
    left = rng.integers(-30000, 30000, size=1000, dtype=np.int32)
    right = rng.integers(-30000, 30000, size=1000, dtype=np.int32)
    mid, side = mid_side(left, right)
    l2, r2 = unmid_side(mid, side)
    if not (np.array_equal(l2, left) and np.array_equal(r2, right)):
        raise SystemExit("mid/side is not reversible")
    for q in (1, 32):
        quants = quantize_bands(bands, q)
        packed = pack_planes(quants, "<i4")
        restored = restore_bands(unpack_planes(packed), q, n)
        again = restore_bands(quants, q, n)
        if not np.array_equal(restored, again):
            raise SystemExit(f"audio pack mismatch at Q={q}")
        if q == 1 and not np.array_equal(restored, samples):
            raise SystemExit("audio Q=1 is not exact")
    # Temporal picture, exact at Q=1.
    prev = np.zeros(16, dtype=np.uint8)
    curr = rng.integers(0, 256, size=16, dtype=np.uint8)
    for q in (1, 8):
        resid = curr.astype(np.int32) - prev.astype(np.int32)
        qq, hat = step_of(resid, q)
        recon = np.clip(prev.astype(np.int32) + hat, 0, 255).astype(np.uint8)
        resid2 = curr.astype(np.int32) - prev.astype(np.int32)
        qq2, hat2 = step_of(resid2, q)
        recon2 = np.clip(prev.astype(np.int32) + hat2, 0, 255).astype(np.uint8)
        if not np.array_equal(recon, recon2) or not np.array_equal(qq, qq2):
            raise SystemExit("temporal quant is unstable")
        if q == 1 and not np.array_equal(recon, curr):
            raise SystemExit("temporal Q=1 is not exact")
    side = np.array([40000, -40000], dtype=np.int32)
    try:
        pack_planes([side], "<i2")
    except RuntimeError:
        pass
    else:
        raise SystemExit("int16 pack accepted a value outside int16")
    packed = pack_planes([side], "<i4")
    back = unpack_planes(packed)
    if not np.array_equal(back[0], side):
        raise SystemExit("int32 plane did not round-trip")
    scratch = WATCH / "_selfcheck.fspz"
    stream = ArchiveStream(scratch)
    stream.add(packed)
    stream.finish({"q": 1, "kind": "self-check"})
    header = read_fspz_header(scratch)
    if header.get("q") != 1 or header.get("chunk_count") != 1:
        raise SystemExit("archive header did not round-trip")
    if header.get("payload_sha256") != hashlib.sha256(struct.pack("<I", 1) + struct.pack("<I", len(packed)) + packed).hexdigest():
        raise SystemExit("archive payload hash does not match")
    scratch.unlink()
    print("self-check ok", flush=True)


def decode_pcm(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "2", "-ar", "44100", "pipe:1"],
        capture_output=True, check=True,
    )
    pcm = np.frombuffer(proc.stdout, dtype="<i2").astype(np.int32)
    if pcm.size % 2:
        pcm = pcm[:-1]
    return pcm[0::2].copy(), pcm[1::2].copy(), 44100


def write_wav(path: Path, left: np.ndarray, right: np.ndarray, rate: int) -> None:
    inter = np.empty(left.size * 2, dtype="<i2")
    inter[0::2] = clip_i16(left)
    inter[1::2] = clip_i16(right)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(inter.tobytes())


def flac_size(pcm_path_wav: Path) -> int:
    dest = pcm_path_wav.with_suffix(".flac")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(pcm_path_wav), "-c:a", "flac", str(dest)],
        check=True,
    )
    return dest.stat().st_size


def encode_song(path: Path, engine: dict) -> dict:
    print(f"song {path.name}", flush=True)
    left, right, rate = decode_pcm(path)
    mid, side = mid_side(left, right)
    mid_bands, n = haar_split(mid, 3)
    side_bands, n2 = haar_split(side, 3)
    if n != n2:
        raise RuntimeError("channel lengths differ")
    rows = []
    stem = path.stem
    for q in AUDIO_Q:
        print(f"  Q={q}", flush=True)
        mq = quantize_bands(mid_bands, q)
        sq = quantize_bands(side_bands, q)
        packed = pack_planes(mq + sq, "<i4")
        back = unpack_planes(packed)
        if len(back) != 8:
            raise RuntimeError("expected 8 Haar planes")
        mid_r = restore_bands(back[:4], q, n)
        side_r = restore_bands(back[4:], q, n)
        l2, r2 = unmid_side(mid_r, side_r)
        l2 = clip_i16(l2).astype(np.int32)
        r2 = clip_i16(r2).astype(np.int32)
        if q == 1:
            if not (np.array_equal(l2, left) and np.array_equal(r2, right)):
                raise SystemExit(f"{path.name} Q=1 did not restore the samples")
        err = np.concatenate([l2 - left, r2 - right]).astype(np.float64)
        noise = float(np.mean(err * err))
        signal = float(np.mean(np.concatenate([left, right]).astype(np.float64) ** 2))
        snr = math.inf if noise == 0 else 10 * math.log10(signal / noise)
        wav_path = None
        archive_path = None
        if q == paths.LISTEN_Q:
            listen = paths.CURRENT / "songs"
            listen.mkdir(parents=True, exist_ok=True)
            wav_path = listen / f"{stem}_q{q}.wav"
            archive_path = listen / f"{stem}_q{q}.fspz"
            write_wav(wav_path, l2, r2, rate)
            archive_path.write_bytes(packed)
        rows.append({
            "q": q,
            "archive_bytes": len(packed),
            "wav_bytes": None if wav_path is None else wav_path.stat().st_size,
            "snr_db": None if math.isinf(snr) else snr,
            "exact": bool(noise == 0),
            "max_abs": int(np.max(np.abs(err))),
            "wav": None if wav_path is None else str(wav_path),
            "archive": None if archive_path is None else str(archive_path),
        })
        label = "exact" if noise == 0 else f"SNR {snr:.2f} dB"
        print(f"    archive {len(packed)}  {label}", flush=True)
    exact_wav = WATCH / f"_flac_{stem}.wav"
    write_wav(exact_wav, left, right, rate)
    flac = flac_size(exact_wav)
    exact_wav.unlink(missing_ok=True)
    exact_wav.with_suffix(".flac").unlink(missing_ok=True)
    return {
        "path": str(path),
        "seconds": n / rate,
        "rate": rate,
        "channels": 2,
        "source_bytes": path.stat().st_size,
        "pcm_bytes": int(n * 2 * 2),
        "flac_bytes": flac,
        "ladder": rows,
        "engine_sha256": engine["sha256"],
    }


def plane_shapes(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    return (width, height), (width // 2, height // 2), (width // 2, height // 2)


def open_decoder(path: Path, seconds: float | None) -> subprocess.Popen:
    cmd = ["ffmpeg", "-v", "error"]
    if seconds is not None:
        cmd += ["-t", f"{seconds}"]
    cmd += ["-i", str(path), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-an", "pipe:1"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def read_exact(stream, size: int) -> bytes | None:
    chunks = []
    got = 0
    while got < size:
        part = stream.read(size - got)
        if not part:
            if got == 0:
                return None
            raise RuntimeError("truncated yuv frame")
        chunks.append(part)
        got += len(part)
    return b"".join(chunks)


def split_planes(frame: bytes, width: int, height: int) -> list[np.ndarray]:
    y_n = width * height
    c_n = y_n // 4
    y = np.frombuffer(frame[:y_n], dtype=np.uint8)
    u = np.frombuffer(frame[y_n:y_n + c_n], dtype=np.uint8)
    v = np.frombuffer(frame[y_n + c_n:], dtype=np.uint8)
    return [y, u, v]


def encode_temporal(qq_planes: list[np.ndarray]) -> bytes:
    return pack_planes(qq_planes, "<i2")


def apply_temporal(curr: np.ndarray, prev: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    qq, hat = step_of(curr.astype(np.int32) - prev.astype(np.int32), q)
    recon = np.clip(prev.astype(np.int32) + hat, 0, 255).astype(np.uint8)
    return qq, recon


def video_pass(path: Path, meta: dict, q: int, seconds: float | None, write_view: bool) -> dict:
    width, height, fps = meta["width"], meta["height"], meta["fps"]
    if width % 2 or height % 2:
        raise RuntimeError("yuv420p needs even dimensions")
    shapes = plane_shapes(width, height)
    frame_bytes = width * height * 3 // 2
    chroma_q = 1 if q <= 1 else q * CHROMA_TIMES
    prev = [np.zeros(w * h, dtype=np.uint8) for w, h in shapes]
    sse = [0.0, 0.0, 0.0]
    max_abs = [0, 0, 0]
    frames = 0
    pending: list[np.ndarray] = []
    # The short probe keeps every chunk so the whole clip can be checked.
    # The episode keeps only the first chunk. Later chunks are on disk.
    keep_all = seconds is not None and seconds <= 12
    kept: list[bytes] = []
    tag = "probe" if seconds is not None else "full"
    archive_path = WATCH / f"southpark_s01e01_{tag}_q{q}.fspz"
    stream = ArchiveStream(archive_path)
    view = None
    view_path = None
    err_handle = None
    err_path = None
    if write_view:
        view_path = WATCH / f"southpark_s01e01_{tag}_q{q}.mp4"
        err_path = WATCH / f"southpark_s01e01_{tag}_q{q}.log"
        err_handle = err_path.open("wb")
        view = subprocess.Popen(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}",
                "-r", f"{fps:.8f}", "-i", "pipe:0",
                "-i", str(path),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
                "-c:a", "copy", "-shortest",
                str(view_path),
            ],
            stdin=subprocess.PIPE, stderr=err_handle, bufsize=1 << 20,
        )
    dec = open_decoder(path, seconds)
    assert dec.stdout is not None
    started = time.perf_counter()
    try:
        while True:
            raw = read_exact(dec.stdout, frame_bytes)
            if raw is None:
                break
            planes = split_planes(raw, width, height)
            recon_planes = []
            for index, plane in enumerate(planes):
                step = q if index == 0 else chroma_q
                qq, recon = apply_temporal(plane, prev[index], step)
                prev[index] = recon
                pending.append(qq)
                diff = plane.astype(np.int16) - recon.astype(np.int16)
                sse[index] += float(np.mean(diff.astype(np.float64) ** 2))
                max_abs[index] = max(max_abs[index], int(np.max(np.abs(diff))))
                recon_planes.append(recon)
            if view is not None and view.stdin is not None:
                view.stdin.write(b"".join(p.tobytes() for p in recon_planes))
            frames += 1
            if len(pending) >= CHUNK_FRAMES * 3:
                blob = encode_temporal(pending)
                stream.add(blob)
                if keep_all or not kept:
                    kept.append(blob)
                pending = []
            if frames % 300 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"    Q={q} frames {frames} archive {stream.chunk_bytes} "
                    f"elapsed {elapsed:.0f}s",
                    flush=True,
                )
        if pending:
            blob = encode_temporal(pending)
            stream.add(blob)
            if keep_all or not kept:
                kept.append(blob)
    finally:
        if view is not None and view.stdin is not None:
            view.stdin.close()
        # Kill a decoder that is still writing into the pipe. Waiting on it
        # deadlocks once the pipe fills.
        stop_proc(dec)
        if view is not None:
            view.wait()
        if err_handle is not None:
            err_handle.close()
    if view is not None and view.returncode != 0:
        detail = ""
        if err_path is not None and err_path.exists():
            detail = err_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"view encode failed for Q={q}: {detail}")
    if frames == 0:
        raise RuntimeError("no frames decoded")
    # Round-trip the stored chunks against the running reconstruction rule.
    # A second decode confirms the archive, on a short prefix when the pass is long.
    check_frames = frames if keep_all else min(frames, CHUNK_FRAMES)
    rebuilt = decode_chunks(kept, width, height, q, chroma_q, check_frames)
    dec2 = open_decoder(path, seconds if keep_all else None)
    assert dec2.stdout is not None
    try:
        prev = [np.zeros(w * h, dtype=np.uint8) for w, h in shapes]
        for index in range(check_frames):
            raw = read_exact(dec2.stdout, frame_bytes)
            if raw is None:
                raise RuntimeError("source ended during check")
            planes = split_planes(raw, width, height)
            for plane_index, plane in enumerate(planes):
                step = q if plane_index == 0 else chroma_q
                _qq, recon = apply_temporal(plane, prev[plane_index], step)
                prev[plane_index] = recon
            got = rebuilt[index * frame_bytes:(index + 1) * frame_bytes]
            if got != b"".join(p.tobytes() for p in prev):
                raise SystemExit(f"Q={q} archive does not rebuild frame {index}")
            if q == 1 and got != raw:
                raise SystemExit(f"Q=1 archive is not exact at frame {index}")
    finally:
        stop_proc(dec2)
    stream.finish({
        "q": q,
        "chroma_q": chroma_q,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "source": str(path),
        "seconds_limit": seconds,
        "predictor": "previous reconstructed frame, black before the first",
    })
    header = read_fspz_header(archive_path)
    if header.get("frames") != frames or header.get("chunk_count") != stream.count:
        raise SystemExit(f"Q={q} archive header does not match the pass")
    if not kept:
        raise SystemExit(f"Q={q} stored no chunks")
    if keep_all:
        if payload_sha256(kept) != header["payload_sha256"]:
            raise SystemExit(f"Q={q} disk payload does not match the chunks just encoded")
    elif read_first_chunk(archive_path) != kept[0]:
        raise SystemExit(f"Q={q} first chunk on disk does not match the encoder")
    def mean_psnr(sse_value: float) -> float | None:
        mse = sse_value / frames
        if mse == 0:
            return None
        return 10 * math.log10((255 * 255) / mse)
    return {
        "q": q,
        "chroma_q": chroma_q,
        "frames": frames,
        "seconds": frames / fps,
        "archive_bytes": archive_path.stat().st_size,
        "payload_bytes": stream.chunk_bytes,
        "exact": all(v == 0 for v in max_abs),
        "psnr_y_db": mean_psnr(sse[0]),
        "psnr_u_db": mean_psnr(sse[1]),
        "psnr_v_db": mean_psnr(sse[2]),
        "max_abs_y": max_abs[0],
        "max_abs_u": max_abs[1],
        "max_abs_v": max_abs[2],
        "elapsed_s": time.perf_counter() - started,
        "view": str(view_path) if view_path is not None else None,
        "archive": str(archive_path),
    }


def decode_chunks(chunks: list[bytes], width: int, height: int, q: int, chroma_q: int, frame_limit: int) -> bytes:
    shapes = plane_shapes(width, height)
    prev = [np.zeros(w * h, dtype=np.uint8) for w, h in shapes]
    out = bytearray()
    produced = 0
    for chunk in chunks:
        planes = unpack_planes(chunk)
        if len(planes) % 3:
            raise RuntimeError("chunk plane count is not a multiple of 3")
        for offset in range(0, len(planes), 3):
            if produced >= frame_limit:
                return bytes(out)
            recon_bytes = []
            for plane_index in range(3):
                qq = planes[offset + plane_index]
                step = q if plane_index == 0 else chroma_q
                hat = qq if step == 1 else qq * step
                recon = np.clip(prev[plane_index].astype(np.int32) + hat, 0, 255).astype(np.uint8)
                if recon.size != prev[plane_index].size:
                    raise RuntimeError("plane size mismatch")
                prev[plane_index] = recon
                recon_bytes.append(recon.tobytes())
            out += b"".join(recon_bytes)
            produced += 1
    return bytes(out)


def x264_size(path: Path, seconds: float | None, crf: int) -> int:
    dest = WATCH / f"x264_crf{crf}_{'probe' if seconds else 'full'}.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if seconds is not None:
        cmd += ["-t", f"{seconds}"]
    cmd += ["-i", str(path), "-an", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", str(dest)]
    print(f"x264 crf {crf} seconds={seconds}", flush=True)
    subprocess.run(cmd, check=True)
    return dest.stat().st_size


def probe_video(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name",
            "-show_entries", "format=duration,bit_rate,size",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(proc.stdout)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(num) / float(den),
        "codec": stream["codec_name"],
        "duration": float(data["format"]["duration"]),
        "source_bytes": int(data["format"]["size"]),
        "source_bit_rate": int(data["format"]["bit_rate"]),
    }


def declared_block() -> dict:
    return {
        "audio_q": list(AUDIO_Q),
        "video_q": list(VIDEO_Q),
        "haar_times_fine_to_coarse_then_approx": list(HAAR_TIMES),
        "chroma_times_when_q_gt_1": CHROMA_TIMES,
        "entropy": "zlib-9 on the integer residual",
        "video_predictor": "previous reconstructed frame, black before the first",
        "audio_transform": "reversible mid/side, 3-level integer Haar",
    }


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage not in ("all", "probe", "full", "full-keep"):
        raise SystemExit("stage is all, probe, full, or full-keep")
    run_full = stage == "full-keep"
    if stage == "full":
        raise SystemExit(
            "Full-episode archives are multi-gigabyte and were removed. "
            "The 8 second probe in results/video_probe is the video comparison. "
            "Run deliver.py full-keep only when those archives are intended."
        )
    if stage == "full-keep":
        stage = "full"
    WATCH.mkdir(parents=True, exist_ok=True)
    self_check()
    engine = fspack.load_engine()
    songs_path = WATCH / "songs_partial.json"
    if stage == "full" and songs_path.exists():
        songs = json.loads(songs_path.read_text(encoding="utf-8"))
    else:
        songs = []
        for path in SONGS:
            songs.append(encode_song(path, engine))
            songs_path.write_text(json.dumps(songs, indent=2), encoding="utf-8")
        print("SONGS_DONE", flush=True)
    episode = show_path()
    print(f"show {episode}", flush=True)
    vmeta = probe_video(episode)
    probe_path = WATCH / "probe_partial.json"
    if stage == "full" and probe_path.exists():
        probe_saved = json.loads(probe_path.read_text(encoding="utf-8"))
        probe_rows = probe_saved["probe_8s"]
        competitors = probe_saved["probe_x264_bytes"]
    else:
        print("probe 8 seconds", flush=True)
        probe_rows = []
        for q in VIDEO_Q:
            print(f"  video probe Q={q}", flush=True)
            probe_rows.append(video_pass(episode, vmeta, q, 8.0, write_view=(q in (4, 8, 16))))
            print(f"    {probe_rows[-1]}", flush=True)
        print("x264 probe", flush=True)
        competitors = {
            "crf18": x264_size(episode, 8.0, 18),
            "crf23": x264_size(episode, 8.0, 23),
        }
        probe_path.write_text(json.dumps({
            "probe_8s": probe_rows,
            "probe_x264_bytes": competitors,
            "show": {"path": str(episode), **vmeta},
        }, indent=2), encoding="utf-8")
        print("PROBE_DONE", flush=True)
    if not run_full:
        print("stopped before the full episode", flush=True)
        return
    print("full episode", flush=True)
    full_rows = []
    for q in VIDEO_Q:
        print(f"  full Q={q}", flush=True)
        full_rows.append(video_pass(episode, vmeta, q, None, write_view=False))
        print(f"    {full_rows[-1]}", flush=True)
        (WATCH / "full_partial.json").write_text(json.dumps(full_rows, indent=2), encoding="utf-8")
    print("x264 full", flush=True)
    full_competitors = {
        "crf18": x264_size(episode, None, 18),
        "crf23": x264_size(episode, None, 23),
    }
    report = {
        "pin": fspack.PIN,
        "engine_sha256": engine["sha256"],
        "valve": engine["valve"],
        "declared": declared_block(),
        "songs": songs,
        "show": {
            "path": str(episode),
            **vmeta,
            "probe_8s": probe_rows,
            "probe_x264_bytes": competitors,
            "full": full_rows,
            "full_x264_bytes": full_competitors,
        },
    }
    out = paths.save_score("long_report.json", report)
    write_md(report, paths.SCORES / "LONG_REPORT.md")
    print(f"wrote {out}", flush=True)


def fmt_db(value, exact: bool) -> str:
    if exact:
        return "exact"
    if value is None:
        return "—"
    return f"{value:.2f}"


def write_md(report: dict, path: Path) -> None:
    lines = [
        "# Long-form delivery",
        "",
        f"Pin {report['pin']}. Valve {report['valve']:.12f} is recorded and is not a bitrate.",
        "",
        "Predictor for pictures: the previous reconstructed frame. The frame before the first is black.",
        "Sound: reversible mid/side, then a 3-level integer Haar split.",
        "Declared Haar steps, finest detail through the approximation: 4Q, 2Q, 2Q, Q.",
        "Declared chroma step: 2Q when Q > 1.",
        "Residual packer: zlib -9. This is the same packer used as the baseline, so a smaller archive means the residual got smaller.",
        "",
        "## Retrace",
        "",
        "The compressed state is phase, D_eff, gauge index, and residual. The decoder already holds the engine. The rate that can fall is the residual given that library.",
        "",
        "The first bench had six holes.",
        "",
        "1. The avi and the mp3 were scored as container bytes. Those bytes were already compressed, so D_eff landed at 25 and the ratio sat near 1. Packing a finished codec again cannot beat the codec.",
        "2. Each pixel was predicted from the pixel to its left. Frame N is already known to the decoder when frame N+1 is coded. Leaving the previous reconstruction out of the library is why one second of intra residual was about ten times the mpeg4 that produced the frames.",
        "3. The order-0 coder treats residual bytes as independent. Runs and repeated rows need a repetition model. This run uses zlib -9 and names it. It is not a new entropy coder.",
        "4. The listening clip was 5 seconds, 16 kHz, mono. A person cannot judge the song from that.",
        "5. The first-bench SSIM was the mean of non-overlapping 8x8 blocks, not windowed SSIM. PSNR was the ordinary mean-square score and still stands.",
        "6. The valve fraction is stored. It is not a bitrate.",
        "",
        "This run keeps Q fixed. Picture prediction is the previous reconstruction, with no motion search. Chroma uses 2Q when Q > 1. Haar steps are 4Q, 2Q, 2Q, Q from the finest detail to the approximation. Q=1 keeps every residual and has to restore the samples or the pixels exactly.",
        "",
        "Two library terms are still absent, and they were not added after the scores. A moving edge is still a full residual, because there is no displacement gauge. A quiet sample and a masked sample take the same step, because there is no masking model. FLAC is the lossless audio comparison. The mp3 size is the perceptual comparison. x264 crf 18 and crf 23, video only, are the picture comparisons. The episode file includes its own audio, so a video-only archive is not a win over that file unless the report says so. The watch mp4 is a lossless wrap of the restore plus the original audio. Its size is not the archive size.",
        "",
        "## Songs",
        "",
    ]
    for song in report["songs"]:
        lines += [
            f"### {Path(song['path']).name}",
            "",
            f"Source {song['source_bytes']} bytes, {song['seconds']:.2f} s, 44100 Hz stereo.",
            f"Decoded PCM {song['pcm_bytes']} bytes. FLAC of that PCM {song['flac_bytes']} bytes.",
            "",
            "| Q | Archive bytes | Ratio vs mp3 | SNR dB | Max abs | Listen |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in song["ladder"]:
            ratio = row["archive_bytes"] / song["source_bytes"]
            lines.append(
                f"| {row['q']} | {row['archive_bytes']} | {ratio:.3f} | "
                f"{fmt_db(row['snr_db'], row['exact'])} | {row['max_abs']} | `{row['wav']}` |"
            )
        lines.append("")
    show = report["show"]
    lines += [
        "## South Park S01E01",
        "",
        f"`{show['path']}`",
        f"{show['width']}x{show['height']}, {show['codec']}, {show['duration']:.1f} s, file {show['source_bytes']} bytes, {show['source_bit_rate']} bit/s including audio.",
        "",
        "### 8 second probe",
        "",
        "| Q | Frames | Archive bytes | PSNR Y | Max abs Y | Watch |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in show["probe_8s"]:
        lines.append(
            f"| {row['q']} | {row['frames']} | {row['archive_bytes']} | "
            f"{fmt_db(row['psnr_y_db'], row['exact'])} | {row['max_abs_y']} | {row['view'] or '—'} |"
        )
    lines += [
        "",
        f"x264 video-only on the same 8 seconds: crf 18 = {show['probe_x264_bytes']['crf18']} bytes, crf 23 = {show['probe_x264_bytes']['crf23']} bytes.",
        "",
        "### Full episode",
        "",
        "| Q | Frames | Seconds | Archive bytes | PSNR Y | Max abs Y | Watch |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in show["full"]:
        lines.append(
            f"| {row['q']} | {row['frames']} | {row['seconds']:.1f} | {row['archive_bytes']} | "
            f"{fmt_db(row['psnr_y_db'], row['exact'])} | {row['max_abs_y']} | {row['view'] or '—'} |"
        )
    lines += [
        "",
        f"x264 video-only on the full episode: crf 18 = {show['full_x264_bytes']['crf18']} bytes, crf 23 = {show['full_x264_bytes']['crf23']} bytes.",
        "",
        "The watch files use the original episode audio and a lossless x264 wrap (qp 0) of the reconstructed frames, so the picture you see is the FSPACK restore.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
