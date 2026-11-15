#!/usr/bin/env python3
"""生成演示用的基线灰度 TIFF（仅用 Python 标准库，无需 numpy）。

生成未压缩、单页、8 位灰度、单条带的经典 TIFF，可注入平移与整体亮度变化，
也可把 Compression 标签改成非 1 以演示“不支持的影像”分支（像素负载仍原样写出，
服务端在读取像素前即按标签拒绝）。

用法：
  make_sample_tiff.py OUT.tif [--size 96] [--seed 1]
      [--shift DY DX] [--brightness GAIN OFFSET] [--compressed]
"""

from __future__ import annotations

import argparse
import math
import struct
import sys


def render_pixels(size: int, seed: int, shift: tuple[int, int],
                  gain: float, offset: int) -> bytes:
    dy, dx = shift
    out = bytearray(size * size)
    for r in range(size):
        for c in range(size):
            # np.roll 语义：取源坐标（回绕），shift 表示画面内容的移动量
            sr = (r - dy) % size
            sc = (c - dx) % size
            h = ((sr * 73856093) ^ (sc * 19349663) ^ (seed * 83492791)) & 0xFF
            v = 0.55 * h + 60.0 * math.sin(sc / 9.0 + seed) + 40.0 * math.cos(sr / 13.0)
            v = max(0, min(255, int(v * gain + offset)))
            out[r * size + c] = v
    return bytes(out)


def write_tiff(path: str, size: int, pixels: bytes, compressed: bool) -> None:
    assert len(pixels) == size * size
    # 标签按标签号升序；273/279 为单条带，内联值即可（4 字节 LONG）
    ifd_offset = 8
    entries = [
        (256, 4, 1, size),       # ImageWidth
        (257, 4, 1, size),       # ImageLength
        (258, 3, 1, 8),          # BitsPerSample
        (259, 3, 1, 5 if compressed else 1),  # Compression（5=LZW 标记，演示拒绝路径）
        (262, 3, 1, 1),          # PhotometricInterpretation=黑为零
        (273, 4, 1, None),       # StripOffsets（回填）
        (277, 3, 1, 1),          # SamplesPerPixel
        (278, 4, 1, size),       # RowsPerStrip
        (279, 4, 1, size * size),  # StripByteCounts
        (284, 3, 1, 1),          # PlanarConfiguration
    ]
    entries.sort(key=lambda e: e[0])
    data_offset = ifd_offset + 2 + len(entries) * 12 + 4
    blob = bytearray(b"II" + struct.pack("<HI", 42, ifd_offset))
    blob += struct.pack("<H", len(entries))
    for tag, ftype, count, value in entries:
        if tag == 273:
            value = data_offset
        fmt = {1: "B", 3: "H", 4: "I"}[ftype]
        raw = struct.pack("<" + fmt, value)
        blob += struct.pack("<HHI", tag, ftype, count) + raw + b"\x00" * (4 - len(raw))
    blob += struct.pack("<I", 0)  # 单页：下一个 IFD 为 0
    assert len(blob) == data_offset
    blob += pixels
    with open(path, "wb") as f:
        f.write(blob)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shift", type=int, nargs=2, metavar=("DY", "DX"), default=(0, 0))
    ap.add_argument("--brightness", type=float, nargs=2, metavar=("GAIN", "OFFSET"),
                    default=(1.0, 0.0))
    ap.add_argument("--compressed", action="store_true")
    args = ap.parse_args()
    if not (64 <= args.size <= 512):
        print("边长须在 64～512 之间", file=sys.stderr)
        return 2
    pixels = render_pixels(args.size, args.seed, tuple(args.shift),
                           args.brightness[0], int(args.brightness[1]))
    write_tiff(args.out, args.size, pixels, args.compressed)
    print(f"wrote {args.out} ({args.size}x{args.size}, "
          f"{'compressed-tag' if args.compressed else 'baseline'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
