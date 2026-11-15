"""TIFF 解码与同视野比对（全部在内存中完成，绝不落盘）。

支持范围（题目约束）：
- 经典 TIFF（字节序 II/MM、版本号 42），单 IFD（单页）；
- 未压缩（Compression=1）、条带（Strip）组织，8 位、单通道灰度
  （SamplesPerPixel=1、BitsPerSample=8、PhotometricInterpretation=0/1）；
- 宽、高均在 64～512 像素之间，且两件影像尺寸相同才进入比对。

比对方法：
- 归一化互相关（NCC，皮尔逊相关系数）对整体亮度的线性变化不敏感；
- 在水平、垂直各 ±MAX_SHIFT 像素的位移网格上搜索最佳重叠；
- 以百分位拉伸评估有效纹理占比，大片纯色/纹理不足直接判“无法判定”；
- 最佳位移与次优位移得分差过小（多个位移难分）时同样判“无法判定”，
  避免对周期性/弱纹理内容误报同视野。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MIN_SIDE = 64
MAX_SIDE = 512
MAX_SHIFT = 8

# 判定阈值（NCC 取值 [-1, 1]）
NCC_SAME_THRESHOLD = 0.90      # 最佳位移相似度达到该值才可能判同视野
AMBIGUITY_MARGIN = 0.05        # 最佳与次佳位移得分差小于该值视为难分
FLAT_PATCH_STD = 1.0           # 重叠块灰度标准差低于该值视为纯色块（0-255 量程）
TEXTURE_RATIO_MIN = 0.04       # 有效纹理像素占比下限（4%）
TEXTURE_STD_MIN = 0.06         # 拉伸后图像标准差下限
TEXTURE_CONTRAST = 0.12        # 拉伸后偏离均值超过该值计为有效纹理像素

# TIFF 标签号
TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_PHOTOMETRIC = 262
TAG_STRIP_OFFSETS = 273
TAG_SAMPLES_PER_PIXEL = 277
TAG_ROWS_PER_STRIP = 278
TAG_STRIP_BYTE_COUNTS = 279
TAG_PLANAR_CONFIG = 284
TAG_TILE_WIDTH = 322
TAG_TILE_LENGTH = 323
TAG_SAMPLE_FORMAT = 339
TAG_EXTRA_SAMPLES = 338

# TIFF 字段类型 -> 字节长度
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4}
_TYPE_STRUCT = {1: "B", 2: "c", 3: "H", 4: "I"}


class UnsupportedImage(Exception):
    """影像不满足比对前置条件；消息为可直接展示的中文原因。"""


@dataclass(frozen=True)
class TextureInfo:
    ratio: float          # 有效纹理像素占比
    std_norm: float       # 百分位拉伸后的标准差（0-1 量程）


def decode_tiff(data: bytes) -> np.ndarray:
    """严格解码基线灰度 TIFF，返回 dtype=uint8、shape=(高, 宽) 的数组。

    任何不满足支持范围的情况都抛 UnsupportedImage（带中文原因）。
    """
    if not isinstance(data, (bytes, bytearray)):
        raise UnsupportedImage("影像内容不是有效的字节序列")
    if len(data) < 8:
        raise UnsupportedImage("文件过短，不是完整的 TIFF 影像")

    order_magic = bytes(data[0:2])
    if order_magic == b"II":
        endian = "<"
    elif order_magic == b"MM":
        endian = ">"
    else:
        raise UnsupportedImage("缺少 TIFF 字节序标记（II/MM），不是 TIFF 影像")

    magic, ifd_offset = struct.unpack_from(endian + "HI", data, 2)
    if magic == 43:
        raise UnsupportedImage("BigTIFF（版本号 43）不在支持范围，仅支持经典 TIFF")
    if magic != 42:
        raise UnsupportedImage(f"TIFF 版本号 {magic} 不受支持（仅支持版本号 42）")
    if ifd_offset < 8 or ifd_offset + 2 > len(data):
        raise UnsupportedImage("首个 IFD 偏移越界，TIFF 结构已损坏")

    tags = _read_ifd(data, endian, ifd_offset)
    # ---- 组织方式与采样格式的硬性约束 ----
    if TAG_TILE_WIDTH in tags or TAG_TILE_LENGTH in tags:
        raise UnsupportedImage("不支持瓦片（Tile）组织的 TIFF，仅支持条带（Strip）影像")
    if TAG_SAMPLE_FORMAT in tags and _scalar(tags, TAG_SAMPLE_FORMAT, endian) != 1:
        raise UnsupportedImage("仅支持无符号整数采样，不支持浮点/其他样本格式")
    if TAG_EXTRA_SAMPLES in tags:
        raise UnsupportedImage("存在 ExtraSamples 额外通道，仅支持单通道灰度图")

    planar = _scalar(tags, TAG_PLANAR_CONFIG, endian, default=1)
    if planar != 1:
        raise UnsupportedImage(f"平面配置 PlanarConfiguration={planar} 不受支持，仅支持单平面连续存储")

    spp = _scalar(tags, TAG_SAMPLES_PER_PIXEL, endian, default=1)
    if spp != 1:
        raise UnsupportedImage(f"每像素样本数 SamplesPerPixel={spp}，仅支持单通道灰度图（=1）")

    if TAG_BITS_PER_SAMPLE not in tags:
        raise UnsupportedImage("缺少 BitsPerSample 标签，无法确认位深")
    bits = _scalar(tags, TAG_BITS_PER_SAMPLE, endian)
    if bits != 8:
        raise UnsupportedImage(f"位深 BitsPerSample={bits}，仅支持 8 位灰度图")

    compression = _scalar(tags, TAG_COMPRESSION, endian, default=1)
    if compression != 1:
        raise UnsupportedImage(f"压缩方式 Compression={compression}，仅支持未压缩影像（=1）")

    if TAG_PHOTOMETRIC not in tags:
        raise UnsupportedImage("缺少 PhotometricInterpretation 标签，无法确认色彩空间")
    photometric = _scalar(tags, TAG_PHOTOMETRIC, endian)
    if photometric not in (0, 1):
        raise UnsupportedImage(
            f"光度解释 PhotometricInterpretation={photometric}，仅支持灰度（0=白为零/1=黑为零）"
        )

    # ---- 尺寸 ----
    if TAG_IMAGE_WIDTH not in tags or TAG_IMAGE_LENGTH not in tags:
        raise UnsupportedImage("缺少 ImageWidth/ImageLength 标签，无法确定影像尺寸")
    width = _scalar(tags, TAG_IMAGE_WIDTH, endian)
    height = _scalar(tags, TAG_IMAGE_LENGTH, endian)
    if width <= 0 or height <= 0:
        raise UnsupportedImage(f"影像尺寸非法：{width}×{height}")
    if not (MIN_SIDE <= width <= MAX_SIDE) or not (MIN_SIDE <= height <= MAX_SIDE):
        raise UnsupportedImage(
            f"边长 {width}×{height} 超出支持范围：宽、高均须在 {MIN_SIDE}～{MAX_SIDE} 像素之间"
        )

    # ---- 条带装配 ----
    if TAG_STRIP_OFFSETS not in tags or TAG_STRIP_BYTE_COUNTS not in tags:
        raise UnsupportedImage("缺少 StripOffsets/StripByteCounts 标签，无法定位像素数据")
    offsets = _as_array(tags, TAG_STRIP_OFFSETS, endian)
    byte_counts = _as_array(tags, TAG_STRIP_BYTE_COUNTS, endian)
    if len(offsets) != len(byte_counts) or not offsets:
        raise UnsupportedImage("StripOffsets 与 StripByteCounts 数量不一致或为空")

    rows_per_strip = _scalar(tags, TAG_ROWS_PER_STRIP, endian, default=height)
    if rows_per_strip <= 0:
        raise UnsupportedImage(f"RowsPerStrip={rows_per_strip} 非法")
    expected_strips = (height + rows_per_strip - 1) // rows_per_strip
    if len(offsets) != expected_strips:
        raise UnsupportedImage(
            f"条带数 {len(offsets)} 与 RowsPerStrip 推算的 {expected_strips} 不一致"
        )

    pixels = np.empty(width * height, dtype=np.uint8)
    for idx, (offset, count) in enumerate(zip(offsets, byte_counts)):
        row0 = idx * rows_per_strip
        rows = min(rows_per_strip, height - row0)
        expected = rows * width
        if count != expected:
            raise UnsupportedImage(
                f"第 {idx} 条带字节数 {count} 与期望 {expected}（{rows} 行 × {width} 列）不一致"
            )
        if offset < 0 or offset + count > len(data):
            raise UnsupportedImage(f"第 {idx} 条带数据偏移越界，TIFF 结构已损坏")
        pixels[row0 * width:(row0 + rows) * width] = np.frombuffer(
            data[offset:offset + count], dtype=np.uint8, count=count
        )

    return pixels.reshape((height, width))


def _read_ifd(data: bytes, endian: str, offset: int) -> dict[int, tuple[int, int, bytes]]:
    """读取单个 IFD，返回 {tag: (field_type, count, raw_value_bytes)}。

    值不超过 4 字节时内联在条目内，否则条目给出指向值数据的偏移；
    全部做越界校验，并拒绝多页 IFD 与重复标签。
    """
    if offset + 2 > len(data):
        raise UnsupportedImage("IFD 条目计数字段越界，TIFF 结构已损坏")
    (entry_count,) = struct.unpack_from(endian + "H", data, offset)
    if entry_count == 0:
        raise UnsupportedImage("IFD 不包含任何标签，不是有效的影像")
    entries_end = offset + 2 + entry_count * 12 + 4
    if entries_end > len(data):
        raise UnsupportedImage("IFD 条目区域越界，TIFF 结构已损坏")

    tags: dict[int, tuple[int, int, bytes]] = {}
    cursor = offset + 2
    for _ in range(entry_count):
        tag, field_type, count = struct.unpack_from(endian + "HHI", data, cursor)
        value_field = data[cursor + 8:cursor + 12]
        cursor += 12
        if tag in tags:
            raise UnsupportedImage(f"IFD 中标签 {tag} 重复出现，TIFF 结构不合法")
        if field_type not in _TYPE_SIZE:
            # 未知类型（LONG8/IFD8 等）说明不是基线 TIFF，直接拒绝。
            raise UnsupportedImage(f"标签 {tag} 使用了不受支持的字段类型 {field_type}")
        unit = _TYPE_SIZE[field_type]
        total = unit * count
        if total == 0:
            raise UnsupportedImage(f"标签 {tag} 的计数为 0，TIFF 结构不合法")
        if total <= 4:
            raw = value_field[:total]
        else:
            (value_offset,) = struct.unpack_from(endian + "I", value_field, 0)
            if value_offset + total > len(data):
                raise UnsupportedImage(f"标签 {tag} 的值数据偏移越界，TIFF 结构已损坏")
            raw = data[value_offset:value_offset + total]
        tags[tag] = (field_type, count, raw)

    (next_ifd,) = struct.unpack_from(endian + "I", data, cursor)
    if next_ifd != 0:
        raise UnsupportedImage("存在多个 IFD（多页 TIFF），仅支持单页影像")
    return tags


def _scalar(tags: dict[int, tuple[int, int, bytes]], tag: int, endian: str,
            default: int | None = None) -> int:
    if tag not in tags:
        if default is not None:
            return default
        raise UnsupportedImage(f"缺少必需标签 {tag}")
    field_type, count, raw = tags[tag]
    if count != 1 or field_type not in (1, 3, 4):
        raise UnsupportedImage(f"标签 {tag} 应为单个 SHORT/LONG 值")
    if field_type == 1:
        return raw[0]
    return struct.unpack(endian + _TYPE_STRUCT[field_type], raw)[0]


def _as_array(tags: dict[int, tuple[int, int, bytes]], tag: int, endian: str) -> list[int]:
    if tag not in tags:
        raise UnsupportedImage(f"缺少必需标签 {tag}")
    field_type, count, raw = tags[tag]
    if field_type not in (1, 3, 4):
        raise UnsupportedImage(f"标签 {tag} 的字段类型 {field_type} 不受支持")
    fmt = _TYPE_STRUCT[field_type]
    unit = _TYPE_SIZE[field_type]
    return [struct.unpack_from(endian + fmt, raw, i * unit)[0] for i in range(count)]


# ---------------------------------------------------------------------- #
# 同视野比对
# ---------------------------------------------------------------------- #
def texture_info(frame: np.ndarray) -> TextureInfo:
    """以 2%/98% 百分位拉伸后统计有效纹理占比，对整体亮度变化不敏感。"""
    x = frame.astype(np.float64)
    lo, hi = np.percentile(x, (2.0, 98.0))
    if hi - lo < 1e-9:
        return TextureInfo(ratio=0.0, std_norm=0.0)
    stretched = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    ratio = float(np.mean(np.abs(stretched - stretched.mean()) > TEXTURE_CONTRAST))
    return TextureInfo(ratio=ratio, std_norm=float(stretched.std()))


def _ncc(va: np.ndarray, vb: np.ndarray) -> float | None:
    """两个等长向量的归一化互相关；任一接近常量则返回 None。"""
    sa = va.std()
    sb = vb.std()
    if sa < FLAT_PATCH_STD or sb < FLAT_PATCH_STD:
        return None
    return float(np.mean((va - va.mean()) * (vb - vb.mean())) / (sa * sb))


def _shift_search(a: np.ndarray, b: np.ndarray) -> list[dict]:
    """在 ±MAX_SHIFT 的位移网格上计算 NCC，按得分降序返回候选位移。

    位移 (dy, dx) 表示影像 B 相对影像 A 的平移：A[r, c] 与 B[r-dy, c-dx] 对齐。
    """
    height, width = a.shape
    candidates: list[dict] = []
    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
        r0_a, r1_a = max(0, dy), min(height, height + dy)
        r0_b, r1_b = max(0, -dy), min(height, height - dy)
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            c0_a, c1_a = max(0, dx), min(width, width + dx)
            c0_b, c1_b = max(0, -dx), min(width, width - dx)
            va = a[r0_a:r1_a, c0_a:c1_a].astype(np.float64).ravel()
            vb = b[r0_b:r1_b, c0_b:c1_b].astype(np.float64).ravel()
            ncc = _ncc(va, vb)
            if ncc is None:
                continue
            candidates.append({
                "dy": dy, "dx": dx,
                "ncc": round(ncc, 6),
                "overlap_height": int(r1_a - r0_a),
                "overlap_width": int(c1_a - c0_a),
                "a_rows": [r0_a, r1_a], "a_cols": [c0_a, c1_a],
                "b_rows": [r0_b, r1_b], "b_cols": [c0_b, c1_b],
            })
    candidates.sort(key=lambda item: item["ncc"], reverse=True)
    return candidates


def compare_tiff(data_a: bytes, data_b: bytes) -> dict:
    """解码并比对两件 TIFF 原件（明文字节，仅存内存），返回结构化结论。"""
    unsupported: dict[str, list[str]] = {"a": [], "b": []}
    frames: dict[str, np.ndarray] = {}
    for key, data in (("a", data_a), ("b", data_b)):
        try:
            frames[key] = decode_tiff(data)
        except UnsupportedImage as exc:
            unsupported[key].append(str(exc))

    if unsupported["a"] or unsupported["b"]:
        return {
            "state": "undetermined",
            "unsupported": unsupported,
            "reasons": _unsupported_reasons(unsupported),
            "dimensions": None,
            "best_shift": None,
            "overlap": None,
            "similarity": None,
            "ambiguity_margin": None,
            "second_best_shift": None,
            "texture": None,
            "candidates": [],
        }

    a, b = frames["a"], frames["b"]
    height, width = a.shape
    reasons: list[str] = []
    if b.shape != a.shape:
        reasons.append(
            f"两件影像尺寸不同（A={width}×{height}，B={b.shape[1]}×{b.shape[0]}），"
            "同尺寸前提下才可比对"
        )
        return _undetermined(reasons, dimensions=(width, height),
                             dimensions_b=(int(b.shape[1]), int(b.shape[0])))

    tex_a = texture_info(a)
    tex_b = texture_info(b)
    weak: list[str] = []
    if min(tex_a.ratio, tex_b.ratio) < TEXTURE_RATIO_MIN:
        weak.append(
            f"有效纹理像素占比不足（A={tex_a.ratio:.1%}，B={tex_b.ratio:.1%}，"
            f"下限 {TEXTURE_RATIO_MIN:.0%}），疑似大片纯色画面"
        )
    if min(tex_a.std_norm, tex_b.std_norm) < TEXTURE_STD_MIN:
        weak.append(
            f"拉伸后灰度标准差过低（A={tex_a.std_norm:.3f}，B={tex_b.std_norm:.3f}，"
            f"下限 {TEXTURE_STD_MIN}），有效纹理不足"
        )

    candidates = _shift_search(a, b)
    texture_payload = {
        "a_ratio": round(tex_a.ratio, 6),
        "b_ratio": round(tex_b.ratio, 6),
        "min_ratio": round(min(tex_a.ratio, tex_b.ratio), 6),
        "a_std_norm": round(tex_a.std_norm, 6),
        "b_std_norm": round(tex_b.std_norm, 6),
    }

    if not candidates:
        weak.append("所有位移的重叠区域均接近纯色，无法计算相似度")
        return _undetermined(
            weak, dimensions=(width, height), texture=texture_payload,
        )

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    margin = round(best["ncc"] - second["ncc"], 6) if second else 1.0
    overlap = {
        "a": {"rows": best["a_rows"], "cols": best["a_cols"]},
        "b": {"rows": best["b_rows"], "cols": best["b_cols"]},
        "width": best["overlap_width"],
        "height": best["overlap_height"],
    }

    base = {
        "unsupported": {"a": [], "b": []},
        "dimensions": [width, height],
        "best_shift": {"dy": best["dy"], "dx": best["dx"]},
        "overlap": overlap,
        "similarity": best["ncc"],
        "ambiguity_margin": margin,
        "second_best_shift": (
            {"dy": second["dy"], "dx": second["dx"], "ncc": second["ncc"]}
            if second else None
        ),
        "texture": texture_payload,
        "candidates": [
            {"dy": c["dy"], "dx": c["dx"], "ncc": c["ncc"]} for c in candidates[:5]
        ],
    }

    if weak:
        return {**base, "state": "undetermined", "reasons": weak}

    if best["ncc"] < NCC_SAME_THRESHOLD:
        return {
            **base, "state": "different_fov",
            "reasons": [
                f"最佳位移（dy={best['dy']}, dx={best['dx']}）相似度 "
                f"{best['ncc']:.4f} 低于同视野阈值 {NCC_SAME_THRESHOLD:.2f}"
            ],
        }

    if margin < AMBIGUITY_MARGIN:
        return {
            **base, "state": "undetermined",
            "reasons": [
                f"最佳位移与次优位移得分差仅 {margin:.4f}（下限 {AMBIGUITY_MARGIN}），"
                "多个位移难分，为避免误报不判同视野"
            ],
        }

    return {
        **base, "state": "same_fov",
        "reasons": [
            f"最佳位移（dy={best['dy']}, dx={best['dx']}）相似度 {best['ncc']:.4f} "
            f"≥ {NCC_SAME_THRESHOLD:.2f}，且与次优位移差距 {margin:.4f} "
            f"≥ {AMBIGUITY_MARGIN}，峰值唯一"
        ],
    }


def _unsupported_reasons(unsupported: dict[str, list[str]]) -> list[str]:
    reasons: list[str] = []
    for key, label in (("a", "影像 A"), ("b", "影像 B")):
        for reason in unsupported[key]:
            reasons.append(f"{label}不受支持：{reason}")
    return reasons


def _undetermined(
    reasons: list[str],
    dimensions: tuple[int, int] | None = None,
    dimensions_b: tuple[int, int] | None = None,
    texture: dict | None = None,
) -> dict:
    return {
        "state": "undetermined",
        "unsupported": {"a": [], "b": []},
        "reasons": reasons,
        "dimensions": list(dimensions) if dimensions else None,
        "dimensions_b": list(dimensions_b) if dimensions_b else None,
        "best_shift": None,
        "overlap": None,
        "similarity": None,
        "ambiguity_margin": None,
        "second_best_shift": None,
        "texture": texture,
        "candidates": [],
    }
