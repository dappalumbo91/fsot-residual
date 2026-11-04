#!/usr/bin/env python3
"""Fold 4. Declared before this pass is scored.

The quotients and the fold1 payloads stay. The waste is the alphabet
stored in front of each payload: every symbol as a raw int16. Those
lists are sorted and unique, so the information is the first symbol
and the positive gaps. Each gap is an Elias gamma code. Nothing else
is tried. A gap of 1 is the shortest code. No Rice parameter. No
second width.

Zoom bytes are copied. Buckets are untouched. Bruno is included
because the quotients do not change; the new margin is not a rung.
No coarsen. No wav.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold1
import fspack
import paths
import refine_fullband as band
import refine_gauge as gauge
import refine_zoom as zoom

ROOT = Path(__file__).resolve().parent
CODER = ROOT / "fold_range.exe"
Q = 2048
RATE = 44100
MAGIC = b"FS2F"
RULE = (
    "same quotients and same fold1 payloads; sorted support stored as "
    "zigzag Elias gamma of the first symbol and gamma of each positive gap; "
    "zoom blob copied; Bruno included; new margin is not a rung; no wav"
)
STEMS = (
    "bruno_mars_just_the_way_you_are",
    "cage_the_elephant_aint_no_rest",
    "skrillex_first_of_the_year",
    "vnv_nation_beloved",
)


class BitOut:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.n = 0
        self.bits = 0

    def write(self, bit: int) -> None:
        self.acc = (self.acc << 1) | (bit & 1)
        self.n += 1
        self.bits += 1
        if self.n == 8:
            self.buf.append(self.acc)
            self.acc = 0
            self.n = 0

    def gamma(self, number: int) -> None:
        if number < 1:
            raise SystemExit("gamma code of a non-positive integer")
        length = number.bit_length()
        for _i in range(length - 1):
            self.write(0)
        for shift in range(length - 1, -1, -1):
            self.write((number >> shift) & 1)

    def finish(self) -> bytes:
        if self.n:
            self.buf.append(self.acc << (8 - self.n))
        return bytes(self.buf)


class BitIn:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.i = 0
        self.acc = 0
        self.n = 0

    def read(self) -> int:
        if self.n == 0:
            if self.i >= len(self.data):
                raise SystemExit("gamma code ran off the support")
            self.acc = self.data[self.i]
            self.i += 1
            self.n = 8
        self.n -= 1
        return (self.acc >> self.n) & 1

    def gamma(self) -> int:
        zeros = 0
        while self.read() == 0:
            zeros += 1
            if zeros > 40:
                raise SystemExit("gamma code is wider than an int16 gap")
        number = 1
        for _i in range(zeros):
            number = (number << 1) | self.read()
        return number


def zigzag(value: int) -> int:
    if value >= 0:
        return value << 1
    return (-value << 1) - 1


def unzigzag(value: int) -> int:
    if value & 1:
        return -((value + 1) >> 1)
    return value >> 1


def encode_support(symbols: np.ndarray) -> bytes:
    vals = [int(v) for v in symbols]
    if any(vals[i] <= vals[i - 1] for i in range(1, len(vals))):
        raise SystemExit("support is not strictly increasing")
    out = BitOut()
    out.gamma(zigzag(vals[0]) + 1)
    prev = vals[0]
    for value in vals[1:]:
        out.gamma(value - prev)
        prev = value
    packed = out.finish()
    if len(packed) > 65535:
        raise SystemExit("gamma support does not fit in uint16")
    return struct.pack("<H", len(packed)) + packed


def decode_support(blob: bytes, offset: int, uniq: int) -> tuple[np.ndarray, int]:
    (size,) = struct.unpack_from("<H", blob, offset)
    offset += 2
    packed = blob[offset:offset + size]
    offset += size
    if len(packed) != size:
        raise SystemExit("truncated gamma support")
    reader = BitIn(packed)
    symbols = np.empty(uniq, dtype=np.int32)
    symbols[0] = unzigzag(reader.gamma() - 1)
    for index in range(1, uniq):
        symbols[index] = symbols[index - 1] + reader.gamma()
    return symbols, offset


def repack_plane(blob: bytes, offset: int) -> tuple[bytes, int]:
    n, uniq = struct.unpack_from("<II", blob, offset)
    offset += 8
    symbols = np.frombuffer(blob[offset:offset + uniq * 2], dtype="<i2").astype(np.int32)
    offset += uniq * 2
    (payload_len,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    payload = blob[offset:offset + payload_len]
    offset += payload_len
    if len(payload) != payload_len:
        raise SystemExit("truncated fold1 payload")
    packed = struct.pack("<II", n, uniq)
    packed += encode_support(symbols)
    packed += struct.pack("<I", payload_len)
    packed += payload
    return packed, offset


def unpack_plane(blob: bytes, offset: int) -> tuple[np.ndarray, int]:
    n, uniq = struct.unpack_from("<II", blob, offset)
    offset += 8
    symbols, offset = decode_support(blob, offset, uniq)
    (payload_len,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    payload = blob[offset:offset + payload_len]
    offset += payload_len
    query = struct.pack("<III", n, uniq, payload_len) + payload
    done = subprocess.run([str(CODER), "dec"], input=query, capture_output=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"fold coder dec exited {done.returncode}")
    if len(done.stdout) != n * 2:
        raise SystemExit("decoded fold length does not match")
    inverse = np.frombuffer(done.stdout, dtype="<u2").astype(np.int32)
    return symbols[inverse], offset


def transform(source: bytes) -> bytes:
    if source[:4] != b"FS2A":
        raise SystemExit("fold4 reads a fold1 archive")
    (zoom_len,) = struct.unpack_from("<I", source, 26)
    offset = 30 + zoom_len
    parts = [MAGIC, source[4:offset]]
    for _channel in range(2):
        for _band in range(4):
            plane, offset = repack_plane(source, offset)
            parts.append(plane)
            (n_ctx,) = struct.unpack_from("<H", source, offset)
            offset += 2
            ctx = [struct.pack("<H", n_ctx)]
            for _i in range(n_ctx):
                ctx.append(source[offset:offset + 2])
                offset += 2
                plane, offset = repack_plane(source, offset)
                ctx.append(plane)
            parts.append(b"".join(ctx))
    if offset != len(source):
        raise SystemExit("fold1 archive was not fully consumed")
    return b"".join(parts)


def unpack_song(blob: bytes, rate: int):
    if blob[:4] != MAGIC:
        raise SystemExit("not a fold4 archive")
    q, original, frames = struct.unpack_from("<III", blob, 4)
    offset = 16
    edges = list(struct.unpack_from("<HHHHH", blob, offset))
    offset += 10
    if edges != gauge.band_edges(edges[-1], rate, zoom.HOP):
        raise SystemExit("archive edges are not the octave cuts")
    (zoom_len,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    import zlib
    zoom_raw = zlib.decompress(blob[offset:offset + zoom_len])
    offset += zoom_len
    mid_b = np.frombuffer(zoom_raw[: frames * 4], dtype=np.uint8).astype(np.int16).copy().reshape(frames, 4)
    side_b = np.frombuffer(zoom_raw[frames * 4:], dtype=np.uint8).astype(np.int16).copy().reshape(frames, 4)
    planes = []
    for _channel in range(2):
        cols = []
        for band_i in range(4):
            width = edges[band_i + 1] - edges[band_i]
            first, offset = unpack_plane(blob, offset)
            plane = np.zeros((frames, width), dtype=np.int32)
            plane[0] = first
            (n_ctx,) = struct.unpack_from("<H", blob, offset)
            offset += 2
            streams = {}
            cursors = {}
            for _i in range(n_ctx):
                (bucket_id,) = struct.unpack_from("<h", blob, offset)
                offset += 2
                values, offset = unpack_plane(blob, offset)
                streams[int(bucket_id)] = values
                cursors[int(bucket_id)] = 0
            import refine_context as context
            for t in range(frames - 1):
                prev = context.context_bucket(plane[t])
                for k in range(width):
                    bucket_id = int(prev[k])
                    cursor = cursors[bucket_id]
                    plane[t + 1, k] = streams[bucket_id][cursor]
                    cursors[bucket_id] = cursor + 1
            cols.append(plane)
        planes.append(np.column_stack(cols))
    if offset != len(blob):
        raise SystemExit("trailing fold4 bytes")
    return q, original, planes[0], mid_b, planes[1], side_b


def self_check() -> None:
    if not CODER.is_file():
        raise SystemExit("fold coder is not built")
    done = subprocess.run([str(CODER), "self"], capture_output=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"fold self exited {done.returncode}")
    symbols = np.array([-2, 0, 3, 5], dtype=np.int32)
    packed = encode_support(symbols)
    back, offset = decode_support(packed, 0, 4)
    if offset != len(packed) or not np.array_equal(back, symbols):
        raise SystemExit("gamma support did not round-trip")
    raw = fold1.pack_plane(np.array([0, -2, 3, 3, -2, 0, 5, -2], dtype=np.int32))
    plane, used = repack_plane(raw, 0)
    if used != len(raw):
        raise SystemExit("fold1 plane was not fully read")
    values, end = unpack_plane(plane, 0)
    if end != len(plane) or not np.array_equal(values, np.array([0, -2, 3, 3, -2, 0, 5, -2])):
        raise SystemExit("fold4 plane did not round-trip")
    print("fold4 self-check ok", flush=True)


def main() -> None:
    self_check()
    out = paths.begin_pass("fold4")
    opus = band.opus_table()
    songs = []
    for stem in STEMS:
        print(f"song {stem}", flush=True)
        source = (paths.CURRENT / "fold1" / f"{stem}_q{Q}.fs2a").read_bytes()
        q, original, mid_qq, mid_b, side_qq, side_b = fold1.unpack_song(source, RATE)
        blob = transform(source)
        got_q, got_n, got_mid, got_mb, got_side, got_sb = unpack_song(blob, RATE)
        if got_q != q or got_n != original:
            raise SystemExit(f"{stem} header did not round-trip")
        if not np.array_equal(got_mid, mid_qq) or not np.array_equal(got_side, side_qq):
            raise SystemExit(f"{stem} quotients did not round-trip")
        if not np.array_equal(got_mb, mid_b) or not np.array_equal(got_sb, side_b):
            raise SystemExit(f"{stem} zoom did not round-trip")
        (zoom_len,) = struct.unpack_from("<I", source, 26)
        if blob[30:30 + zoom_len] != source[30:30 + zoom_len]:
            raise SystemExit(f"{stem} zoom bytes were rewritten")
        archive_path = out / f"{stem}_q{Q}.fs2f"
        archive_path.write_bytes(blob)
        seconds = original / RATE
        opus_n = opus[stem]
        row = {
            "q": Q,
            "fold1_bytes": len(source),
            "archive_bytes": len(blob),
            "opus96_bytes": opus_n,
            "ratio_vs_opus96": len(blob) / opus_n,
            "ratio_vs_fold1": len(blob) / len(source),
            "under_opus": len(blob) < opus_n,
            "margin_bytes": opus_n - len(blob),
            "kbps": len(blob) * 8 / seconds / 1000,
            "quotients_match": True,
            "zoom_bytes_match": True,
            "archive": str(archive_path),
        }
        songs.append({"stem": stem, "seconds": seconds, "row": row})
        paths.save_score("fold4_report.json", {
            "pin": fspack.PIN,
            "declared": {
                "rule": RULE,
                "songs": list(STEMS),
                "q": Q,
                "magic": MAGIC.decode("ascii"),
            },
            "songs": songs,
        })
        print(
            f"  archive {len(blob)} fold1 {len(source)} opus {opus_n} "
            f"ratio {row['ratio_vs_opus96']:.6f} margin {row['margin_bytes']} "
            f"{row['kbps']:.1f}kbps",
            flush=True,
        )
    print("fold4 done", flush=True)


if __name__ == "__main__":
    main()
