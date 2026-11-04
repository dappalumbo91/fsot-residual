#!/usr/bin/env python3
"""FSPACK: an FSOT 2.1 packer.

Lossless members round-trip byte for byte. Video and audio quality modes
keep a closed-loop prediction residual and a step Q that is chosen before
the file is seen. The entropy coder is an order-1 rANS. The engine pin is
AEB2AD. Q is a declared distortion step, not a fitted coefficient, and the
POOF/SUCTION fraction is recorded and is not used as a compression ratio.

Commands:
  python fspack.py selftest
  python fspack.py pack INPUT -o OUT.fsp
  python fspack.py unpack OUT.fsp -o RESTORED
  python fspack.py bench
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

ENGINE_PATH = Path(r"C:\Users\damia\Desktop\FSOT-2.1-Lean\vendor\fsot_compute.py")
PIN = "AEB2AD"
PIN_PREFIX = "aeb2adad6e80f487"
MAGIC = b"FSP1"
SCALE_BITS = 12
SCALE = 1 << SCALE_BITS
RANS_L = 1 << 23
# Half a byte. Below this the phase is a bound condensate. At or above it, free flow.
PHASE_SPLIT_BITS = 4.0
VIDEO_Q = (1, 4, 8, 16)
AUDIO_Q = (1, 16, 64, 256)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_engine():
    raw = ENGINE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not digest.startswith(PIN_PREFIX):
        raise SystemExit(f"engine pin mismatch: {digest[:16]} (wanted {PIN_PREFIX})")
    spec = importlib.util.spec_from_file_location("fsot_compute_aeb2ad", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {ENGINE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    poof = float(mod.POOF)
    suction = float(mod.SUCTION)
    valve = poof / (poof + suction)
    return {
        "sha256": digest,
        "K": float(mod.K),
        "phi": float(mod.PHI),
        "poof": poof,
        "suction": suction,
        "valve": valve,
    }


def zoom_from_entropy(entropy_bits: float) -> tuple[int, str, int]:
    """Same rounding as the engine nest: D_eff(g) = round(5 * 5^(g/19))."""
    h = min(8.0, max(0.0, entropy_bits))
    g = int(round(19.0 * h / 8.0))
    g = min(19, max(0, g))
    d_eff = int(round(5.0 * (5.0 ** (g / 19.0))))
    phase = "bound" if h < PHASE_SPLIT_BITS else "free"
    return d_eff, phase, g


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    total = int(counts.sum())
    nz = counts[counts > 0].astype(np.float64)
    p = nz / total
    return float(-(p * np.log2(p)).sum())


def zig(n: int) -> int:
    if n >= 0:
        return n * 2
    return (-n) * 2 - 1


def unzig(z: int) -> int:
    if z & 1:
        return -((z + 1) // 2)
    return z // 2


def write_varints(values: list[int]) -> bytes:
    out = bytearray()
    for n in values:
        z = zig(n)
        while z >= 128:
            out.append((z & 127) | 128)
            z >>= 7
        out.append(z)
    return bytes(out)


def read_varints(blob: bytes, count: int) -> list[int]:
    vals: list[int] = []
    i = 0
    n = len(blob)
    for _ in range(count):
        z = 0
        shift = 0
        while True:
            if i >= n:
                raise ValueError("truncated varint stream")
            b = blob[i]
            i += 1
            z |= (b & 127) << shift
            if b < 128:
                break
            shift += 7
            if shift > 63:
                raise ValueError("varint too long")
        vals.append(unzig(z))
    if i != n:
        raise ValueError(f"varint stream has {n - i} trailing bytes")
    return vals


def quantize(resid: int, q: int) -> int:
    if q == 1:
        return resid
    if resid >= 0:
        return (resid + q // 2) // q
    return -(((-resid) + q // 2) // q)


def delta256(data: bytes) -> bytes:
    if not data:
        return b""
    src = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    prev = np.empty_like(src)
    prev[0] = 0
    prev[1:] = src[:-1]
    return ((src - prev) & 255).astype(np.uint8).tobytes()


def undelta256(data: bytes) -> bytes:
    if not data:
        return b""
    diff = np.frombuffer(data, dtype=np.uint8).astype(np.uint16)
    return np.cumsum(diff, dtype=np.uint16).astype(np.uint8).tobytes()


def normalize(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies over 0..255 summing to SCALE. Seen symbols keep a positive slot."""
    counts = counts.astype(np.int64)
    freq = np.zeros(256, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        base = SCALE // 256
        freq[:] = base
        freq[0] += SCALE - int(freq.sum())
    else:
        raw = counts * SCALE // total
        seen = counts > 0
        raw[seen & (raw == 0)] = 1
        diff = SCALE - int(raw.sum())
        if diff > 0:
            order = np.argsort(-counts, kind="stable")
            k = 0
            while diff > 0:
                raw[int(order[k % 256])] += 1
                diff -= 1
                k += 1
                if k > SCALE * 4:
                    raise RuntimeError("normalize failed to fill")
        elif diff < 0:
            order = np.argsort(-raw, kind="stable")
            k = 0
            while diff < 0:
                idx = int(order[k % 256])
                floor = 1 if counts[idx] > 0 else 0
                if raw[idx] > floor:
                    raw[idx] -= 1
                    diff += 1
                k += 1
                if k > SCALE * 8:
                    raise RuntimeError("normalize failed to trim")
        freq = raw
    if int(freq.sum()) != SCALE:
        raise RuntimeError(f"freq sum {int(freq.sum())}")
    if np.any((counts > 0) & (freq <= 0)):
        raise RuntimeError("seen symbol lost its frequency")
    cum = np.zeros(257, dtype=np.int64)
    np.cumsum(freq, out=cum[1:])
    return freq, cum


def build_order0(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    if data:
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    else:
        counts = np.zeros(256, dtype=np.int64)
    return normalize(counts)


def rans_encode(data: bytes) -> bytes:
    """Static order-0 rANS. The frequency table is 256 uint16 values, 512 bytes."""
    freq, cum = build_order0(data)
    renorm: list[int] = []
    x = RANS_L
    freq_i = freq.tolist()
    cum_i = cum.tolist()
    for sym in reversed(data):
        f = freq_i[sym]
        if f <= 0:
            raise RuntimeError(f"zero frequency for byte {sym}")
        start = cum_i[sym]
        x_max = ((RANS_L >> SCALE_BITS) << 8) * f
        while x >= x_max:
            renorm.append(x & 255)
            x >>= 8
        x = ((x // f) << SCALE_BITS) + (x % f) + start
        if x >= (1 << 32):
            raise RuntimeError("rANS state left 32 bits")
    payload = x.to_bytes(4, "little") + bytes(reversed(renorm))
    table = freq.astype("<u2").tobytes()
    return struct.pack("<I", len(data)) + table + payload


def rans_decode(blob: bytes) -> bytes:
    table_bytes = 256 * 2
    if len(blob) < 4 + table_bytes + 4:
        raise ValueError("rANS blob too short")
    (n,) = struct.unpack_from("<I", blob, 0)
    table_end = 4 + table_bytes
    freq = np.frombuffer(blob[4:table_end], dtype="<u2").astype(np.int32)
    cum = np.zeros(257, dtype=np.int32)
    np.cumsum(freq, out=cum[1:])
    stream = blob[table_end:]
    if len(stream) < 4:
        raise ValueError("rANS payload too short")
    x = int.from_bytes(stream[:4], "little")
    pos = 4
    out = bytearray(n)
    mask = SCALE - 1
    stream_len = len(stream)
    freq_i = freq.tolist()
    cum_i = cum.tolist()
    for i in range(n):
        cf = x & mask
        sym = 0
        # Binary search on the cumulative table. Slot s owns [cum[s], cum[s+1]).
        lo = 0
        hi = 255
        while lo < hi:
            mid = (lo + hi) // 2
            if cum_i[mid + 1] <= cf:
                lo = mid + 1
            else:
                hi = mid
        sym = lo
        f = freq_i[sym]
        start = cum_i[sym]
        x = f * (x >> SCALE_BITS) + cf - start
        while x < RANS_L:
            if pos >= stream_len:
                raise ValueError("rANS stream ended early")
            x = (x << 8) | stream[pos]
            pos += 1
        out[i] = sym
    return bytes(out)


def dpcm_bytes(data: bytes, q: int, sample_bits: int) -> tuple[bytes, bytes]:
    """Closed-loop DPCM. sample_bits 8 or 16. Returns (varint residual, reconstructed bytes)."""
    if sample_bits == 8:
        src = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
        lo, hi = 0, 255
        width = 1
    elif sample_bits == 16:
        if len(data) % 2:
            raise ValueError("pcm byte length is odd")
        src = np.frombuffer(data, dtype="<i2").astype(np.int32)
        lo, hi = -32768, 32767
        width = 2
    else:
        raise ValueError(sample_bits)
    recon = np.empty_like(src)
    quants = np.empty_like(src)
    prev = 0
    for i, sample in enumerate(src.tolist()):
        qq = quantize(int(sample) - prev, q)
        rec = prev + (qq if q == 1 else qq * q)
        if rec < lo:
            rec = lo
        elif rec > hi:
            rec = hi
        recon[i] = rec
        quants[i] = qq
        prev = rec
    residual = write_varints(quants.tolist())
    if sample_bits == 8:
        raw = recon.astype(np.uint8).tobytes()
    else:
        raw = recon.astype("<i2").tobytes()
    if width and len(raw) != len(src) * width:
        raise RuntimeError("reconstructed width drifted")
    return residual, raw


def undpcm_bytes(residual: bytes, count: int, q: int, sample_bits: int) -> bytes:
    quants = read_varints(residual, count)
    if sample_bits == 8:
        lo, hi = 0, 255
        recon = np.empty(count, dtype=np.int32)
    else:
        lo, hi = -32768, 32767
        recon = np.empty(count, dtype=np.int32)
    prev = 0
    for i, qq in enumerate(quants):
        rec = prev + (qq if q == 1 else qq * q)
        if rec < lo:
            rec = lo
        elif rec > hi:
            rec = hi
        recon[i] = rec
        prev = rec
    if sample_bits == 8:
        return recon.astype(np.uint8).tobytes()
    return recon.astype("<i2").tobytes()


def yuv_dpcm(frame: bytes, width: int, height: int, q: int) -> tuple[bytes, bytes]:
    """Left-predictor DPCM on Y, U, and V. Frame is yuv420p."""
    if width % 2 or height % 2:
        raise ValueError("yuv420p needs even width and height")
    y_n = width * height
    c_w, c_h = width // 2, height // 2
    c_n = c_w * c_h
    need = y_n + 2 * c_n
    if len(frame) != need:
        raise ValueError(f"frame is {len(frame)} bytes, expected {need}")
    planes = (
        (frame[:y_n], width, height),
        (frame[y_n:y_n + c_n], c_w, c_h),
        (frame[y_n + c_n:], c_w, c_h),
    )
    chunks: list[bytes] = []
    recon = bytearray()
    for plane, w, h in planes:
        src = np.frombuffer(plane, dtype=np.uint8)
        rec = np.empty(w * h, dtype=np.uint8)
        quants: list[int] = []
        for y in range(h):
            left = 0
            row = y * w
            for x in range(w):
                pix = int(src[row + x])
                qq = quantize(pix - left, q)
                hat = qq if q == 1 else qq * q
                value = left + hat
                if value < 0:
                    value = 0
                elif value > 255:
                    value = 255
                rec[row + x] = value
                left = value
                quants.append(qq)
        chunks.append(write_varints(quants))
        recon.extend(rec.tobytes())
    # Length-prefix each plane so the decoder can split them.
    blob = bytearray()
    for chunk in chunks:
        blob += struct.pack("<I", len(chunk))
        blob += chunk
    return bytes(blob), bytes(recon)


def yuv_undpcm(blob: bytes, width: int, height: int, frames: int, q: int) -> bytes:
    c_w, c_h = width // 2, height // 2
    plane_dims = ((width, height), (c_w, c_h), (c_w, c_h))
    out = bytearray()
    pos = 0
    for _ in range(frames):
        for w, h in plane_dims:
            if pos + 4 > len(blob):
                raise ValueError("truncated plane header")
            (nbytes,) = struct.unpack_from("<I", blob, pos)
            pos += 4
            chunk = blob[pos:pos + nbytes]
            pos += nbytes
            vals = read_varints(chunk, w * h)
            rec = np.empty(w * h, dtype=np.int32)
            k = 0
            for y in range(h):
                left = 0
                row = y * w
                for _x in range(w):
                    qq = vals[k]
                    k += 1
                    hat = qq if q == 1 else qq * q
                    value = left + hat
                    if value < 0:
                        value = 0
                    elif value > 255:
                        value = 255
                    rec[row + _x] = value
                    left = value
            out.extend(rec.astype(np.uint8).tobytes())
    if pos != len(blob):
        raise ValueError("trailing video residual")
    return bytes(out)


def pack_record(payload: bytes, meta: dict) -> bytes:
    meta = dict(meta)
    meta["payload_bytes"] = len(payload)
    header = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(header) > 0xFFFFFFFF:
        raise ValueError("header too large")
    return MAGIC + struct.pack("<I", len(header)) + header + payload


def unpack_record(blob: bytes) -> tuple[dict, bytes]:
    if blob[:4] != MAGIC:
        raise ValueError("not an FSPACK file")
    (hlen,) = struct.unpack_from("<I", blob, 4)
    meta = json.loads(blob[8:8 + hlen].decode("utf-8"))
    payload = blob[8 + hlen:]
    if len(payload) != meta.get("payload_bytes"):
        raise ValueError("payload length does not match the header")
    return meta, payload


def seal(transformed: bytes, meta: dict) -> bytes:
    coded = rans_encode(transformed)
    meta["coded_bytes"] = len(coded)
    meta["transform_bytes"] = len(transformed)
    return pack_record(coded, meta)


def open_sealed(blob: bytes) -> tuple[dict, bytes]:
    meta, payload = unpack_record(blob)
    transformed = rans_decode(payload)
    if len(transformed) != meta["transform_bytes"]:
        raise ValueError("decoded transform length mismatch")
    return meta, transformed


def base_meta(engine: dict, kind: str, source: bytes, transform: str, q: int | None) -> dict:
    entropy = shannon_entropy(source)
    d_eff, phase, generation = zoom_from_entropy(entropy)
    return {
        "format": "FSPACK",
        "version": 1,
        "pin": PIN,
        "engine_sha256": engine["sha256"],
        "K": engine["K"],
        "valve": engine["valve"],
        "kind": kind,
        "phase": phase,
        "d_eff": d_eff,
        "generation": generation,
        "entropy_bits": entropy,
        "q": q,
        "original_bytes": len(source),
        "sha256": hashlib.sha256(source).hexdigest(),
        "transform": transform,
        "entropy_coder": "rans-order0-static",
    }


def pack_bytes(data: bytes, engine: dict) -> bytes:
    meta = base_meta(engine, "bytes", data, "delta256", None)
    return seal(delta256(data), meta)


def unpack_bytes(blob: bytes) -> bytes:
    meta, transformed = open_sealed(blob)
    if meta["kind"] != "bytes":
        raise ValueError(f"archive kind is {meta['kind']}, not bytes")
    data = undelta256(transformed)
    check_sha(meta, data)
    return data


def check_sha(meta: dict, data: bytes) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if digest != meta["sha256"]:
        raise ValueError("sha256 mismatch")
    if len(data) != meta["original_bytes"]:
        raise ValueError("size mismatch")


def pack_directory(root: Path, engine: dict) -> bytes:
    parts = bytearray()
    files = [p for p in root.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    manifest = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise ValueError(f"refusing path {rel}")
        raw = path.read_bytes()
        rel_b = rel.encode("utf-8")
        parts += struct.pack("<H", len(rel_b))
        parts += rel_b
        parts += struct.pack("<Q", len(raw))
        parts += raw
        manifest.append({"path": rel, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    blob = bytes(parts)
    meta = base_meta(engine, "directory", blob, "delta256", None)
    meta["files"] = manifest
    meta["root_name"] = root.name
    return seal(delta256(blob), meta)


def unpack_directory(blob: bytes, dest: Path) -> None:
    meta, transformed = open_sealed(blob)
    if meta["kind"] != "directory":
        raise ValueError(f"archive kind is {meta['kind']}")
    data = undelta256(transformed)
    check_sha(meta, data)
    dest.mkdir(parents=True, exist_ok=True)
    pos = 0
    seen = []
    while pos < len(data):
        if pos + 2 > len(data):
            raise ValueError("truncated directory entry")
        (plen,) = struct.unpack_from("<H", data, pos)
        pos += 2
        rel = data[pos:pos + plen].decode("utf-8")
        pos += plen
        (size,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        raw = data[pos:pos + size]
        pos += size
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"refusing path {rel}")
        target = dest / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        seen.append(rel)
    names = [item["path"] for item in meta["files"]]
    if seen != names:
        raise ValueError("directory manifest does not match the payload")


def pack_yuv(raw: bytes, engine: dict, width: int, height: int, frames: int, q: int) -> tuple[bytes, bytes]:
    pieces = []
    recon = bytearray()
    frame_bytes = width * height * 3 // 2
    if len(raw) != frames * frame_bytes:
        raise ValueError("yuv length does not match width, height, and frames")
    for i in range(frames):
        frame = raw[i * frame_bytes:(i + 1) * frame_bytes]
        blob, back = yuv_dpcm(frame, width, height, q)
        pieces.append(blob)
        recon.extend(back)
    stream = bytearray()
    for blob in pieces:
        stream += struct.pack("<I", len(blob))
        stream += blob
    meta = base_meta(engine, "yuv420p", raw, "left-dpcm-varint", q)
    meta.update({"width": width, "height": height, "frames": frames, "pix_fmt": "yuv420p"})
    return seal(bytes(stream), meta), bytes(recon)


def unpack_yuv(blob: bytes) -> tuple[dict, bytes]:
    meta, transformed = open_sealed(blob)
    if meta["kind"] != "yuv420p":
        raise ValueError(f"archive kind is {meta['kind']}")
    width, height, frames, q = meta["width"], meta["height"], meta["frames"], meta["q"]
    pos = 0
    planes_blob = bytearray()
    for _ in range(frames):
        (nbytes,) = struct.unpack_from("<I", transformed, pos)
        pos += 4
        planes_blob += transformed[pos:pos + nbytes]
        pos += nbytes
    if pos != len(transformed):
        raise ValueError("trailing frame index")
    raw = yuv_undpcm(bytes(planes_blob), width, height, frames, q)
    if q == 1:
        check_sha(meta, raw)
    return meta, raw


def pack_pcm(raw: bytes, engine: dict, rate: int, channels: int, q: int) -> tuple[bytes, bytes]:
    if channels != 1:
        raise ValueError("this build packs mono pcm")
    residual, recon = dpcm_bytes(raw, q, 16)
    meta = base_meta(engine, "pcm_s16le", raw, "dpcm-varint", q)
    meta.update({"sample_rate": rate, "channels": channels, "sample_format": "s16le"})
    return seal(residual, meta), recon


def unpack_pcm(blob: bytes) -> tuple[dict, bytes]:
    meta, residual = open_sealed(blob)
    if meta["kind"] != "pcm_s16le":
        raise ValueError(f"archive kind is {meta['kind']}")
    count = meta["original_bytes"] // 2
    raw = undpcm_bytes(residual, count, meta["q"], 16)
    if meta["q"] == 1:
        check_sha(meta, raw)
    return meta, raw


def ratio(packed: int, original: int) -> float | None:
    if original == 0:
        return None
    return packed / original


def psnr(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    err = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(err * err))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((peak * peak) / mse)


def ssim_8x8(a: np.ndarray, b: np.ndarray, width: int, height: int) -> float:
    """Mean of non-overlapping 8x8 SSIM blocks on one plane."""
    aa = a.reshape(height, width).astype(np.float64)
    bb = b.reshape(height, width).astype(np.float64)
    h = height - (height % 8)
    w = width - (width % 8)
    aa = aa[:h, :w]
    bb = bb[:h, :w]
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    scores = []
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            ba = aa[y:y + 8, x:x + 8]
            bbk = bb[y:y + 8, x:x + 8]
            mu_a = ba.mean()
            mu_b = bbk.mean()
            va = ba.var()
            vb = bbk.var()
            cov = float(((ba - mu_a) * (bbk - mu_b)).mean())
            scores.append(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (va + vb + c2)))
    return float(np.mean(scores)) if scores else 1.0


def y_planes(raw: bytes, width: int, height: int) -> list[np.ndarray]:
    frame = width * height * 3 // 2
    y_n = width * height
    out = []
    for i in range(len(raw) // frame):
        start = i * frame
        out.append(np.frombuffer(raw[start:start + y_n], dtype=np.uint8))
    return out


def quality_pair(original: bytes, restored: bytes, width: int, height: int) -> dict:
    if len(original) != len(restored):
        raise ValueError("quality pair length mismatch")
    frame = width * height * 3 // 2
    y_n = width * height
    c_n = y_n // 4
    psnr_y = []
    psnr_u = []
    psnr_v = []
    ssim = []
    max_abs = 0
    for i in range(len(original) // frame):
        o = original[i * frame:(i + 1) * frame]
        r = restored[i * frame:(i + 1) * frame]
        oy = np.frombuffer(o[:y_n], dtype=np.uint8)
        ry = np.frombuffer(r[:y_n], dtype=np.uint8)
        ou = np.frombuffer(o[y_n:y_n + c_n], dtype=np.uint8)
        ru = np.frombuffer(r[y_n:y_n + c_n], dtype=np.uint8)
        ov = np.frombuffer(o[y_n + c_n:], dtype=np.uint8)
        rv = np.frombuffer(r[y_n + c_n:], dtype=np.uint8)
        psnr_y.append(psnr(oy, ry, 255))
        psnr_u.append(psnr(ou, ru, 255))
        psnr_v.append(psnr(ov, rv, 255))
        ssim.append(ssim_8x8(oy, ry, width, height))
        max_abs = max(max_abs, int(np.max(np.abs(oy.astype(np.int16) - ry.astype(np.int16)))))
    def mean_finite(xs: list[float]) -> float | None:
        finite = [v for v in xs if math.isfinite(v)]
        if len(finite) != len(xs):
            return None
        return float(np.mean(finite))
    return {
        "psnr_y_db": None if any(math.isinf(v) for v in psnr_y) else mean_finite(psnr_y),
        "psnr_u_db": None if any(math.isinf(v) for v in psnr_u) else mean_finite(psnr_u),
        "psnr_v_db": None if any(math.isinf(v) for v in psnr_v) else mean_finite(psnr_v),
        "exact": all(math.isinf(v) for v in psnr_y + psnr_u + psnr_v),
        "ssim_y_8x8": float(np.mean(ssim)),
        "max_abs_y": max_abs,
    }


def snr_pcm(original: bytes, restored: bytes) -> dict:
    o = np.frombuffer(original, dtype="<i2").astype(np.float64)
    r = np.frombuffer(restored, dtype="<i2").astype(np.float64)
    err = o - r
    noise = float(np.mean(err * err))
    signal = float(np.mean(o * o))
    snr = math.inf if noise == 0 else 10.0 * math.log10(signal / noise)
    return {
        "snr_db": None if math.isinf(snr) else snr,
        "exact": bool(noise == 0),
        "max_abs": int(np.max(np.abs(err))) if len(err) else 0,
    }


def run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg failed")


def ffprobe_video(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(proc.stdout)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(num) / float(den),
        "codec": stream["codec_name"],
    }


def selftest() -> None:
    engine = load_engine()
    assert abs(engine["valve"] - 0.5107285887739611) < 1e-12
    d0, phase0, _ = zoom_from_entropy(0.0)
    d8, phase8, _ = zoom_from_entropy(8.0)
    if d0 != 5 or phase0 != "bound":
        raise SystemExit(f"zoom at 0 bits is {d0} {phase0}")
    if d8 != 25 or phase8 != "free":
        raise SystemExit(f"zoom at 8 bits is {d8} {phase8}")
    samples = [
        b"",
        b"\x00",
        b"\xff" * 50,
        bytes(range(256)) * 4,
        np.random.default_rng(1).integers(0, 256, size=4000, dtype=np.uint8).tobytes(),
        b"abracadabra" * 200,
    ]
    for blob in samples:
        back = rans_decode(rans_encode(blob))
        if back != blob:
            raise SystemExit(f"rANS mismatch at length {len(blob)}")
        packed = pack_bytes(blob, engine)
        if unpack_bytes(packed) != blob:
            raise SystemExit("byte pack mismatch")
    # Closed-loop video block, exact at Q=1 and lossy at Q=8.
    w, h = 32, 16
    rng = np.random.default_rng(2)
    y = rng.integers(0, 256, size=w * h, dtype=np.uint8)
    u = rng.integers(0, 256, size=(w // 2) * (h // 2), dtype=np.uint8)
    v = rng.integers(0, 256, size=(w // 2) * (h // 2), dtype=np.uint8)
    frame = y.tobytes() + u.tobytes() + v.tobytes()
    for q in (1, 8):
        blob, recon = yuv_dpcm(frame, w, h, q)
        back = yuv_undpcm(blob, w, h, 1, q)
        if back != recon:
            raise SystemExit(f"video decode mismatch at Q={q}")
        if q == 1 and back != frame:
            raise SystemExit("Q=1 video is not exact")
    pcm = rng.integers(-20000, 20000, size=1000, dtype=np.int16).astype("<i2").tobytes()
    for q in (1, 64):
        residual, recon = dpcm_bytes(pcm, q, 16)
        back = undpcm_bytes(residual, 1000, q, 16)
        if back != recon:
            raise SystemExit(f"pcm decode mismatch at Q={q}")
        if q == 1 and back != pcm:
            raise SystemExit("Q=1 pcm is not exact")
    folder = RESULTS / "_selftest_src"
    if folder.exists():
        for child in folder.rglob("*"):
            if child.is_file():
                child.unlink()
    (folder / "sub").mkdir(parents=True, exist_ok=True)
    (folder / "a.txt").write_text("fsot pack\n", encoding="utf-8")
    (folder / "sub" / "b.bin").write_bytes(bytes(range(100)))
    archive = pack_directory(folder, engine)
    dest = RESULTS / "_selftest_out"
    if dest.exists():
        for child in sorted(dest.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    unpack_directory(archive, dest)
    if (dest / "a.txt").read_text(encoding="utf-8") != "fsot pack\n":
        raise SystemExit("directory text mismatch")
    if (dest / "sub" / "b.bin").read_bytes() != bytes(range(100)):
        raise SystemExit("directory binary mismatch")
    print(f"selftest ok  pin {PIN}  valve {engine['valve']:.12f}  K {engine['K']:.12f}")


def finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def bench() -> None:
    engine = load_engine()
    RESULTS.mkdir(exist_ok=True)
    video_path = Path(r"G:\Cinema and movies and shows\movies\Getaway [2013] HDRip XViD-ETRG\sample.avi")
    audio_path = Path(r"G:\Music discography's\music i made\cries_in_the_beat.mp3")
    text_path = ROOT / "DIRECTION.md"
    if not video_path.is_file():
        raise SystemExit(f"missing video {video_path}")
    if not audio_path.is_file():
        raise SystemExit(f"missing audio {audio_path}")

    report: dict = {
        "pin": PIN,
        "engine_sha256": engine["sha256"],
        "K": engine["K"],
        "poof": engine["poof"],
        "suction": engine["suction"],
        "valve": engine["valve"],
        "phase_split_bits": PHASE_SPLIT_BITS,
        "video_q_ladder": list(VIDEO_Q),
        "audio_q_ladder": list(AUDIO_Q),
        "notes": [
            "Q is the declared residual step. It was fixed before this run.",
            "The valve fraction is recorded from the engine and is not a file's compression ratio.",
            "Byte archives must restore sha256-identical bytes.",
            "Q=1 picture and pcm modes must restore the decoded samples exactly.",
            "An already-compressed mpeg4 or mp3 container is an entropy control.",
        ],
    }

    print("packing text and the compressed containers")
    containers = []
    for label, path in (
        ("direction_md", text_path),
        ("sample_avi_container", video_path),
        ("cries_mp3_container", audio_path),
    ):
        data = path.read_bytes()
        packed = pack_bytes(data, engine)
        restored = unpack_bytes(packed)
        if restored != data:
            raise SystemExit(f"{label} round trip failed")
        zipped = len(zlib.compress(data, 9))
        meta, _payload = unpack_record(packed)
        row = {
            "label": label,
            "path": str(path),
            "bytes": len(data),
            "fspack_bytes": len(packed),
            "zlib9_bytes": zipped,
            "fspack_ratio": ratio(len(packed), len(data)),
            "zlib9_ratio": ratio(zipped, len(data)),
            "phase": meta["phase"],
            "d_eff": meta["d_eff"],
            "entropy_bits": meta["entropy_bits"],
            "sha256_ok": True,
        }
        containers.append(row)
        print(f"  {label}: {len(data)} -> fspack {len(packed)} zlib {zipped} D_eff {meta['d_eff']} {meta['phase']}")
    report["containers"] = containers

    info = ffprobe_video(video_path)
    frames = 24
    print(f"decoding {frames} frames from {video_path.name} ({info['width']}x{info['height']})")
    yuv_path = RESULTS / "source_24f.yuv"
    run_ffmpeg(["-i", str(video_path), "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "yuv420p", str(yuv_path)])
    yuv = yuv_path.read_bytes()
    width, height = info["width"], info["height"]
    video_rows = []
    contact_frames = {}
    for q in VIDEO_Q:
        print(f"  video Q={q}")
        packed, recon = pack_yuv(yuv, engine, width, height, frames, q)
        _meta, decoded = unpack_yuv(packed)
        if decoded != recon:
            raise SystemExit(f"video Q={q} unpack mismatch")
        if q == 1 and decoded != yuv:
            raise SystemExit("video Q=1 is not exact")
        scores = quality_pair(yuv, decoded, width, height)
        out_path = RESULTS / f"getaway_q{q}.fsp"
        out_path.write_bytes(packed)
        row = {
            "q": q,
            "raw_bytes": len(yuv),
            "fspack_bytes": len(packed),
            "ratio_vs_raw": ratio(len(packed), len(yuv)),
            **scores,
            "archive": str(out_path),
        }
        video_rows.append(row)
        contact_frames[q] = decoded[:width * height * 3 // 2]
        exact = "exact" if scores["exact"] else f"PSNR Y {scores['psnr_y_db']:.2f} dB  SSIM {scores['ssim_y_8x8']:.4f}"
        print(f"    {len(yuv)} -> {len(packed)}  {exact}")
    raw_zlib = len(zlib.compress(yuv, 9))
    report["video"] = {
        "path": str(video_path),
        "codec": info["codec"],
        "width": width,
        "height": height,
        "fps": info["fps"],
        "frames": frames,
        "raw_yuv_bytes": len(yuv),
        "zlib9_raw_bytes": raw_zlib,
        "ladder": video_rows,
    }

    # Viewing copies. libx264 qp 0 stores these reconstructed frames without a second lossy step.
    view_dir = RESULTS / "frames"
    view_dir.mkdir(exist_ok=True)
    one = width * height * 3 // 2
    pngs = []
    for label, frame in (("source", yuv[:one]), *[ (f"q{q}", contact_frames[q]) for q in VIDEO_Q ]):
        raw_frame = view_dir / f"{label}.yuv"
        raw_frame.write_bytes(frame)
        png = view_dir / f"{label}.png"
        run_ffmpeg([
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}",
            "-i", str(raw_frame), "-frames:v", "1", str(png),
        ])
        pngs.append(png)
        raw_frame.unlink()
    sheet = RESULTS / "getaway_contact.png"
    run_ffmpeg([
        "-i", str(pngs[0]), "-i", str(pngs[1]), "-i", str(pngs[2]), "-i", str(pngs[3]), "-i", str(pngs[4]),
        "-filter_complex", "[0][1][2][3][4]hstack=inputs=5",
        str(sheet),
    ])
    q8 = contact_frames[8]
    # Repeat the single compared frame is not the clip. Write the full Q=8 reconstruction.
    q8_yuv = RESULTS / "getaway_q8.yuv"
    _packed_unused, q8_full = pack_yuv(yuv, engine, width, height, frames, 8)
    q8_yuv.write_bytes(q8_full)
    del q8
    clip = RESULTS / "getaway_q8.mp4"
    run_ffmpeg([
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}",
        "-r", f"{info['fps']:.8f}", "-i", str(q8_yuv),
        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(clip),
    ])
    q8_yuv.unlink()
    yuv_path.unlink()
    report["video"]["contact_sheet"] = str(sheet)
    report["video"]["q8_view_mp4"] = str(clip)
    report["video"]["contact_order"] = ["source", "q1", "q4", "q8", "q16"]

    print("decoding 5 seconds of mono audio")
    pcm_path = RESULTS / "cries_5s.pcm"
    run_ffmpeg(["-i", str(audio_path), "-t", "5", "-ac", "1", "-ar", "16000", "-f", "s16le", str(pcm_path)])
    pcm = pcm_path.read_bytes()
    audio_rows = []
    for q in AUDIO_Q:
        print(f"  audio Q={q}")
        packed, recon = pack_pcm(pcm, engine, 16000, 1, q)
        _meta, decoded = unpack_pcm(packed)
        if decoded != recon:
            raise SystemExit(f"audio Q={q} unpack mismatch")
        if q == 1 and decoded != pcm:
            raise SystemExit("audio Q=1 is not exact")
        scores = snr_pcm(pcm, decoded)
        out_path = RESULTS / f"cries_q{q}.fsp"
        out_path.write_bytes(packed)
        if q == 1 or q == 64:
            wav = RESULTS / f"cries_q{q}.wav"
            (RESULTS / f"cries_q{q}.pcm").write_bytes(decoded)
            run_ffmpeg(["-f", "s16le", "-ar", "16000", "-ac", "1", "-i", str(RESULTS / f"cries_q{q}.pcm"), str(wav)])
            (RESULTS / f"cries_q{q}.pcm").unlink()
        row = {
            "q": q,
            "raw_bytes": len(pcm),
            "fspack_bytes": len(packed),
            "ratio_vs_raw": ratio(len(packed), len(pcm)),
            **scores,
            "archive": str(out_path),
        }
        audio_rows.append(row)
        exact = "exact" if scores["exact"] else f"SNR {scores['snr_db']:.2f} dB"
        print(f"    {len(pcm)} -> {len(packed)}  {exact}")
    pcm_path.unlink()
    report["audio"] = {
        "path": str(audio_path),
        "decoded": "5 seconds, 16000 Hz, mono, s16le",
        "zlib9_raw_bytes": len(zlib.compress(pcm, 9)),
        "ladder": audio_rows,
        "listen": {
            "q1": str(RESULTS / "cries_q1.wav"),
            "q64": str(RESULTS / "cries_q64.wav"),
        },
    }

    out_json = RESULTS / "report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report_md(report, RESULTS / "REPORT.md")
    print(f"wrote {out_json}")


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def fmt_db(value: float | None, exact: bool) -> str:
    if exact:
        return "exact"
    if value is None:
        return "—"
    return f"{value:.2f}"


def write_report_md(report: dict, path: Path) -> None:
    lines = [
        "# FSPACK bench",
        "",
        f"Engine pin {report['pin']}, sha256 `{report['engine_sha256']}`.",
        f"K = {report['K']:.12f}. POOF/(POOF+SUCTION) = {report['valve']:.12f}.",
        "That valve is the pump fraction from the engine. It is not the compression ratio of these files.",
        "",
        "Q was fixed as 1, 4, 8, 16 for pictures and 1, 16, 64, 256 for audio before the files were scored.",
        "Byte archives were checked by sha256. Q=1 picture and audio restores were checked against the decoded samples.",
        "",
        "## Containers",
        "",
        "These inputs are already encoded, or they are prose. The ratio is packed size divided by original size.",
        "",
        "| File | Bytes | FSPACK | zlib -9 | FSPACK ratio | zlib ratio | D_eff | Phase |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["containers"]:
        lines.append(
            f"| {row['label']} | {row['bytes']} | {row['fspack_bytes']} | {row['zlib9_bytes']} | "
            f"{fmt_ratio(row['fspack_ratio'])} | {fmt_ratio(row['zlib9_ratio'])} | {row['d_eff']} | {row['phase']} |"
        )
    video = report["video"]
    lines += [
        "",
        "## Video",
        "",
        f"Source `{video['path']}`.",
        f"Decoded {video['frames']} frames, {video['width']}x{video['height']}, {video['codec']}, {video['fps']:.5f} fps.",
        f"Raw yuv420p {video['raw_yuv_bytes']} bytes. zlib -9 on that raw stream: {video['zlib9_raw_bytes']} bytes.",
        "PSNR and SSIM compare the FSPACK restore with those decoded frames. The 8x8 SSIM is the mean of non-overlapping blocks on Y.",
        "",
        "| Q | FSPACK bytes | Ratio vs raw | PSNR Y | PSNR U | PSNR V | SSIM Y | Max abs Y |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in video["ladder"]:
        lines.append(
            f"| {row['q']} | {row['fspack_bytes']} | {fmt_ratio(row['ratio_vs_raw'])} | "
            f"{fmt_db(row['psnr_y_db'], row['exact'])} | {fmt_db(row['psnr_u_db'], row['exact'])} | "
            f"{fmt_db(row['psnr_v_db'], row['exact'])} | {row['ssim_y_8x8']:.4f} | {row['max_abs_y']} |"
        )
    lines += [
        "",
        f"Contact sheet, source then Q=1, 4, 8, 16: `{video['contact_sheet']}`.",
        f"Q=8 viewing file, stored with x264 qp 0 so the wrap does not add a second lossy step: `{video['q8_view_mp4']}`.",
        "",
        "## Audio",
        "",
        f"Source `{report['audio']['path']}`.",
        report["audio"]["decoded"] + f". zlib -9 on that pcm: {report['audio']['zlib9_raw_bytes']} bytes.",
        "",
        "| Q | FSPACK bytes | Ratio vs raw | SNR dB | Max abs |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["audio"]["ladder"]:
        lines.append(
            f"| {row['q']} | {row['fspack_bytes']} | {fmt_ratio(row['ratio_vs_raw'])} | "
            f"{fmt_db(row['snr_db'], row['exact'])} | {row['max_abs']} |"
        )
    lines += [
        "",
        f"Exact restore: `{report['audio']['listen']['q1']}`.",
        f"Q=64 restore: `{report['audio']['listen']['q64']}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_pack(args: argparse.Namespace) -> None:
    engine = load_engine()
    src = Path(args.input)
    if src.is_dir():
        blob = pack_directory(src, engine)
    elif src.is_file():
        blob = pack_bytes(src.read_bytes(), engine)
    else:
        raise SystemExit(f"missing {src}")
    dest = Path(args.output)
    dest.write_bytes(blob)
    print(f"wrote {dest} ({len(blob)} bytes)")


def cmd_unpack(args: argparse.Namespace) -> None:
    blob = Path(args.input).read_bytes()
    meta, _payload = unpack_record(blob)
    dest = Path(args.output)
    kind = meta["kind"]
    if kind == "bytes":
        dest.write_bytes(unpack_bytes(blob))
    elif kind == "directory":
        unpack_directory(blob, dest)
    elif kind == "yuv420p":
        _meta, raw = unpack_yuv(blob)
        dest.write_bytes(raw)
    elif kind == "pcm_s16le":
        _meta, raw = unpack_pcm(blob)
        dest.write_bytes(raw)
    else:
        raise SystemExit(f"unknown kind {kind}")
    print(f"unpacked {kind} -> {dest}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="FSPACK, an FSOT 2.1 archive and quality coder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("bench")
    pack = sub.add_parser("pack")
    pack.add_argument("input")
    pack.add_argument("-o", "--output", required=True)
    unpack = sub.add_parser("unpack")
    unpack.add_argument("input")
    unpack.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "bench":
        bench()
    elif args.cmd == "pack":
        cmd_pack(args)
    elif args.cmd == "unpack":
        cmd_unpack(args)


if __name__ == "__main__":
    main(sys.argv[1:])
