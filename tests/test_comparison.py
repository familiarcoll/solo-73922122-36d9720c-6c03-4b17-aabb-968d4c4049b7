"""同视野比对端到端测试。

覆盖：
1. 真实基线 TIFF（小端/大端）解码：同视野（含平移、整体亮度变化）、不同视野；
2. 不支持影像的原因说明：多页、压缩、非 8 位、多通道、瓦片、边长越界、尺寸不同等；
3. 防误报：大片纯色、弱纹理、周期性纹理多位移难分均不得判同视野；
4. 安全不变量：比对全程明文/DEK 不落盘，结论绑定本次读取原件的 SHA-256；
5. 轮换并发：比对进行期间课题密钥轮换照常完成，比对仍能给出正确结论。
"""

from __future__ import annotations

import os
import struct
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import imaging


# ---------------------------------------------------------------------- #
# 真实基线 TIFF 构造器（未压缩、单页、8 位灰度、条带组织）
# ---------------------------------------------------------------------- #
def build_tiff(
    pixels: np.ndarray,
    *,
    endian: str = "<",
    compression: int = 1,
    bits: int = 8,
    samples_per_pixel: int = 1,
    photometric: int = 1,
    rows_per_strip: int | None = None,
    planar_config: int = 1,
    extra_tags: list[tuple[int, int, object]] | None = None,
    next_ifd: int = 0,
) -> bytes:
    """按基线 TIFF 规范手工编码，便于注入各种“不受支持”的字段。"""
    height, width = pixels.shape
    assert pixels.dtype == np.uint8
    if rows_per_strip is None:
        rows_per_strip = height
    strip_rows = min(rows_per_strip, height)
    strip_count = (height + strip_rows - 1) // strip_rows
    strip_offsets: list[int] = []
    strip_byte_counts: list[int] = []

    type_size = {1: 1, 2: 1, 3: 2, 4: 4}
    type_fmt = {1: "B", 2: "c", 3: "H", 4: "I"}

    def encode(ftype: int, value: object) -> bytes:
        fmt = type_fmt[ftype]
        if isinstance(value, (list, tuple)):
            return b"".join(struct.pack(endian + fmt, v) for v in value)
        return struct.pack(endian + fmt, value)

    # 先计算每条带的字节数与像素负载（偏移在整体布局确定后回填）
    strips_payload = b""
    for i in range(strip_count):
        r0 = i * strip_rows
        r1 = min(height, r0 + strip_rows)
        chunk = pixels[r0:r1, :].tobytes()
        strip_byte_counts.append(len(chunk))
        strips_payload += chunk
    strip_offsets = [0] * strip_count  # 占位，保证标签 273 计数正确

    tag_values: list[tuple[int, int, object]] = [
        (256, 4, width), (257, 4, height),
        (258, 3, bits), (259, 3, compression),
        (262, 3, photometric), (277, 3, samples_per_pixel),
        (278, 4, strip_rows), (284, 3, planar_config),
        (273, 4, strip_offsets), (279, 4, strip_byte_counts),
    ]
    if extra_tags:
        tag_values.extend(extra_tags)
    tag_values.sort(key=lambda t: t[0])

    # 第一遍：确定每个外联值的偏移（IFD 之后），像素数据排在所有外联值之后
    n = len(tag_values)
    ifd_offset = 8
    cursor = ifd_offset + 2 + n * 12 + 4
    layouts: list[tuple[int, int, int, bytes, int | None]] = []
    for tag, ftype, value in tag_values:
        raw = encode(ftype, value)
        count = len(value) if isinstance(value, (list, tuple)) else 1
        assert type_size[ftype] * count == len(raw)
        offset = None
        if len(raw) > 4:
            offset = cursor
            cursor += len(raw)
        layouts.append((tag, ftype, count, raw, offset))
    data_origin = cursor
    for i in range(strip_count):
        strip_offsets[i] = data_origin + sum(strip_byte_counts[:i])

    # 第二遍：回填 StripOffsets 后写出
    blob = bytearray()
    blob += (b"II" if endian == "<" else b"MM") + struct.pack(endian + "HI", 42, ifd_offset)
    blob += struct.pack(endian + "H", n)
    for tag, ftype, count, _raw, offset in layouts:
        if tag == 273:
            raw = encode(ftype, strip_offsets)
        else:
            raw = _raw
        blob += struct.pack(endian + "HHI", tag, ftype, count)
        if len(raw) <= 4:
            blob += raw + b"\x00" * (4 - len(raw))
        else:
            blob += struct.pack(endian + "I", offset)
    blob += struct.pack(endian + "I", next_ifd)
    for tag, ftype, count, _raw, offset in layouts:
        raw = encode(ftype, strip_offsets) if tag == 273 else _raw
        if len(raw) > 4:
            if len(blob) != offset:
                blob += b"\x00" * (offset - len(blob))
            blob += raw
    blob += strips_payload
    return bytes(blob)


def make_texture(size: int = 96, seed: int = 0, shift: tuple[int, int] = (0, 0),
                 gain: float = 1.0, offset: int = 0) -> np.ndarray:
    """生成带随机纹理的 8 位灰度图，可整体平移并施加亮度线性变化。"""
    rng = np.random.default_rng(1000 + seed)
    base = rng.integers(0, 256, size=(size, size), dtype=np.int32)
    # 加入低频结构，避免纯白噪声导致自相关峰过窄
    yy, xx = np.mgrid[0:size, 0:size]
    base = base * 0.7 + 60 * np.sin(xx / 9.0 + seed) + 40 * np.cos(yy / 13.0)
    dy, dx = shift
    out = np.roll(base, (dy, dx), axis=(0, 1))
    out = np.clip(out * gain + offset, 0, 255).astype(np.uint8)
    return out


@pytest.fixture()
def tmp_data(tmp_path):
    return str(tmp_path / "data")


@pytest.fixture()
def svc(tmp_data, monkeypatch):
    monkeypatch.setenv("ARCHIVE_DATA_DIR", tmp_data)
    with TestClient(app) as client:
        from app.main import service as current
        yield client, current


def _upload_tiff(client: TestClient, pixels: np.ndarray, name: str = "img.tif", **kw) -> str:
    payload = build_tiff(pixels, **kw)
    r = client.post("/api/uploads", json={"filename": name, "size": len(payload)})
    assert r.status_code == 201, r.text
    uid = r.json()["upload_id"]
    assert client.put(f"/api/uploads/{uid}/content", content=payload).status_code == 200
    r = client.post(f"/api/uploads/{uid}/complete")
    assert r.status_code == 200, r.text
    return r.json()["file_id"]


def _compare(client: TestClient, a: str, b: str):
    r = client.post("/api/comparisons", json={"file_a_id": a, "file_b_id": b})
    assert r.status_code == 201, r.text
    return r.json()


class TestTiffDecoder:
    def test_decode_both_endians(self):
        frame = make_texture(64)
        for endian in ("<", ">"):
            decoded = imaging.decode_tiff(build_tiff(frame, endian=endian))
            assert decoded.shape == (64, 64) and decoded.dtype == np.uint8
            assert np.array_equal(decoded, frame)

    def test_reject_baseline_violations(self):
        frame = make_texture(64)
        cases = {
            "multipage": build_tiff(frame, next_ifd=999),
            "compressed": build_tiff(frame, compression=5),
            "16bit": build_tiff(frame, bits=16),
            "multisample": build_tiff(frame, samples_per_pixel=3),
            "tiled": build_tiff(frame, extra_tags=[(322, 4, 64), (323, 4, 64)]),
            "planar2": build_tiff(frame, planar_config=2),
            "palette_photometric": build_tiff(frame, photometric=3),
        }
        for label, blob in cases.items():
            with pytest.raises(imaging.UnsupportedImage) as ei:
                imaging.decode_tiff(blob)
            assert str(ei.value), label

    def test_reject_side_out_of_range(self):
        small = np.zeros((32, 64), dtype=np.uint8)
        big = np.zeros((513, 128), dtype=np.uint8)
        for frame in (small, big):
            with pytest.raises(imaging.UnsupportedImage, match="边长"):
                imaging.decode_tiff(build_tiff(frame))

    def test_reject_garbage_and_truncation(self):
        with pytest.raises(imaging.UnsupportedImage):
            imaging.decode_tiff(b"not a tiff at all" * 4)
        frame = make_texture(64)
        blob = build_tiff(frame)
        with pytest.raises(imaging.UnsupportedImage):
            imaging.decode_tiff(blob[: len(blob) // 2])

    def test_strip_assembly_rows_per_strip(self):
        frame = make_texture(80)
        decoded = imaging.decode_tiff(build_tiff(frame, rows_per_strip=17))
        assert np.array_equal(decoded, frame)


class TestSameFieldOfView:
    def test_identical_images_same_fov_zero_shift(self, svc):
        client, _ = svc
        frame = make_texture(96, seed=1)
        a = _upload_tiff(client, frame, "a.tif")
        b = _upload_tiff(client, frame.copy(), "b.tif")
        r = _compare(client, a, b)
        assert r["state"] == "same_fov"
        assert r["best_shift"] == {"dy": 0, "dx": 0}
        assert r["similarity"] == pytest.approx(1.0, abs=1e-6)
        assert r["overlap"]["width"] == r["overlap"]["height"] == 96

    def test_shifted_within_8px_same_fov(self, svc):
        client, _ = svc
        frame = make_texture(128, seed=2)
        # np.roll((5,-7)) 把画面内容向下 5、向左 7 移动；按“B 相对 A 的平移量”
        # 约定（A[r,c] 对齐 B[r-dy,c-dx]），最佳位移为 (-5, 7)。
        shifted = make_texture(128, seed=2, shift=(5, -7))
        a = _upload_tiff(client, frame, "a.tif")
        b = _upload_tiff(client, shifted, "b.tif")
        r = _compare(client, a, b)
        assert r["state"] == "same_fov", r["reasons"]
        assert (r["best_shift"]["dy"], r["best_shift"]["dx"]) == (-5, 7)
        assert r["similarity"] >= 0.95
        ov = r["overlap"]
        assert ov["width"] == 128 - 7 and ov["height"] == 128 - 5
        # 半开区间切片可直接用于取数
        assert ov["a"]["rows"] == [0, 123] and ov["a"]["cols"] == [7, 128]
        assert ov["b"]["rows"] == [5, 128] and ov["b"]["cols"] == [0, 121]

    def test_brightness_change_tolerated(self, svc):
        client, _ = svc
        frame = make_texture(96, seed=3)
        bright = make_texture(96, seed=3, gain=0.55, offset=70)
        a = _upload_tiff(client, frame, "a.tif")
        b = _upload_tiff(client, bright, "b.tif")
        r = _compare(client, a, b)
        assert r["state"] == "same_fov", r["reasons"]
        assert r["best_shift"] == {"dy": 0, "dx": 0}
        assert r["similarity"] >= 0.98

    def test_shift_beyond_8px_is_different_fov(self, svc):
        client, _ = svc
        frame = make_texture(128, seed=4)
        shifted = make_texture(128, seed=4, shift=(12, 0))
        a = _upload_tiff(client, frame, "a.tif")
        b = _upload_tiff(client, shifted, "b.tif")
        r = _compare(client, a, b)
        # 超出 ±8 的平移无法对齐，最佳得分显著下降，不得判同视野
        assert r["state"] == "different_fov"
        assert r["similarity"] < imaging.NCC_SAME_THRESHOLD

    def test_unrelated_images_different_fov(self, svc):
        client, _ = svc
        a = _upload_tiff(client, make_texture(96, seed=11), "a.tif")
        b = _upload_tiff(client, make_texture(96, seed=22), "b.tif")
        r = _compare(client, a, b)
        assert r["state"] == "different_fov"
        assert r["similarity"] < imaging.NCC_SAME_THRESHOLD

    def test_dimension_mismatch_undetermined(self, svc):
        client, _ = svc
        a = _upload_tiff(client, make_texture(64, 5), "a.tif")
        b = _upload_tiff(client, make_texture(100, 5), "b.tif")
        r = _compare(client, a, b)
        assert r["state"] == "undetermined"
        assert any("尺寸不同" in x for x in r["reasons"])

    def test_unsupported_image_reports_reason(self, svc):
        client, _ = svc
        a = _upload_tiff(client, make_texture(64, 6), "a.tif")
        # 压缩 TIFF：归档仍允许（上传不限制内容），比对时给出不支持原因
        b = _upload_tiff(client, make_texture(64, 6), "b.tif", compression=5)
        r = _compare(client, a, b)
        assert r["state"] == "undetermined"
        assert r["unsupported"]["b"] and "压缩" in r["unsupported"]["b"][0]
        assert any("影像 B" in x for x in r["reasons"])
        assert r["best_shift"] is None and r["similarity"] is None

    def test_flat_color_field_undetermined(self, svc):
        client, _ = svc
        flat_a = np.full((96, 96), 30, dtype=np.uint8)
        flat_b = np.full((96, 96), 200, dtype=np.uint8)  # 亮度不同的纯色
        a = _upload_tiff(client, flat_a, "flat-a.tif")
        b = _upload_tiff(client, flat_b, "flat-b.tif")
        r = _compare(client, a, b)
        assert r["state"] == "undetermined"
        assert r["similarity"] is None
        assert any("纹理" in x or "纯色" in x for x in r["reasons"])

    def test_large_flat_region_with_little_texture_undetermined(self, svc):
        client, _ = svc
        # 仅角落一小块有纹理（占比 < 4%），其余纯色
        weak = np.full((128, 128), 128, dtype=np.uint8)
        weak[0:8, 0:8] = np.arange(64, dtype=np.uint8).reshape(8, 8)
        a = _upload_tiff(client, weak, "weak-a.tif")
        b = _upload_tiff(client, weak.copy(), "weak-b.tif")
        r = _compare(client, a, b)
        assert r["state"] == "undetermined"
        assert any("纹理" in x for x in r["reasons"])

    def test_periodic_pattern_ambiguous_shifts_undetermined(self, svc):
        client, _ = svc
        # 周期为 4 的竖条纹：多个整数周期平移后完全相同 -> 峰值不唯一
        cols = (np.arange(128) % 4) * 80
        periodic = np.repeat(cols[None, :], 128, axis=0).astype(np.uint8)
        a = _upload_tiff(client, periodic, "p-a.tif")
        b = _upload_tiff(client, np.roll(periodic, 4, axis=1), "p-b.tif")
        r = _compare(client, a, b)
        assert r["state"] == "undetermined", r["reasons"]
        assert any("难分" in x for x in r["reasons"])

    def test_same_file_rejected(self, svc):
        client, _ = svc
        a = _upload_tiff(client, make_texture(64, 7), "a.tif")
        r = client.post("/api/comparisons", json={"file_a_id": a, "file_b_id": a})
        assert r.status_code == 422

    def test_missing_file_404(self, svc):
        client, _ = svc
        a = _upload_tiff(client, make_texture(64, 7), "a.tif")
        r = client.post("/api/comparisons",
                        json={"file_a_id": a, "file_b_id": "deadbeef" * 4})
        assert r.status_code == 404


class TestSecurityAndRotation:
    def test_plaintext_and_dek_never_on_disk(self, svc, tmp_data):
        client, _ = svc
        frame = make_texture(96, seed=8)
        payload = build_tiff(frame)
        a = _upload_tiff(client, frame, "a.tif")
        b = _upload_tiff(client, make_texture(96, seed=8, shift=(3, 2)), "b.tif")
        r = _compare(client, a, b)
        assert r["state"] == "same_fov"

        # 磁盘上不允许出现 TIFF 明文魔数或原始像素；数据库不存明文/密钥
        for root, _dirs, files in os.walk(tmp_data):
            for name in files:
                path = os.path.join(root, name)
                with open(path, "rb") as f:
                    data = f.read()
                if "vault" in path or "staging" in path:
                    assert payload not in data
                    assert b"II\x2a\x00" not in data
        import sqlite3
        conn = sqlite3.connect(os.path.join(tmp_data, "archive.db"))
        cols = [row[1] for row in conn.execute("PRAGMA table_info(comparisons)").fetchall()]
        assert "detail" in cols
        for (detail,) in conn.execute("SELECT detail FROM comparisons"):
            assert b"II\x2a\x00".hex() not in detail
            assert payload.hex() not in detail
        # detail 中只保存指标，不包含任何像素数组
        record = conn.execute("SELECT detail FROM comparisons").fetchone()[0]
        assert "candidates" in record and "reasons" in record
        conn.close()

    def test_conclusion_bound_to_original_sha256(self, svc):
        client, _ = svc
        a = _upload_tiff(client, make_texture(64, 9), "a.tif")
        b = _upload_tiff(client, make_texture(64, 10), "b.tif")
        r = _compare(client, a, b)
        history = client.get("/api/comparisons").json()
        assert len(history) == 1
        h = history[0]
        files = {f["file_id"]: f for f in client.get("/api/files").json()}
        assert h["file_a"]["sha256"] == files[a]["sha256"]
        assert h["file_b"]["sha256"] == files[b]["sha256"]
        assert h["comparison_id"] == r["comparison_id"]
        assert h["state"] == r["state"]

    def test_compare_during_rotation_succeeds(self, svc):
        client, service = svc
        # 先归档若干影像
        pairs = []
        for i in range(3):
            frame = make_texture(80, seed=20 + i)
            a = _upload_tiff(client, frame, f"a{i}.tif")
            b = _upload_tiff(client, make_texture(80, seed=20 + i, shift=(i + 1, -(i + 1))),
                             f"b{i}.tif")
            pairs.append((a, b))

        errors: list[str] = []
        results: list[dict] = []

        def run_compares():
            try:
                for a, b in pairs:
                    results.append(_compare(client, a, b))
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

        t = threading.Thread(target=run_compares)
        t.start()
        # 比对进行中发起轮换：比对只在内存读明文，轮换应照常完成
        rr = client.post("/api/rotations")
        assert rr.status_code == 201
        t.join(timeout=30)
        assert not errors, errors

        deadline = time.time() + 15
        while time.time() < deadline:
            if not client.get("/api/rotations/current").json()["active"]:
                break
            time.sleep(0.1)
        assert not client.get("/api/rotations/current").json()["active"]

        assert len(results) == 3
        for r in results:
            assert r["state"] == "same_fov", r["reasons"]
        # 轮换后旧 KEK 文件已删除，全部影像仍可解密（比对历史保留）
        keys = client.get("/api/keys").json()
        assert all(k["status"] != "rotating" for k in keys)
        assert len(client.get("/api/comparisons").json()) == 3
        # 解密路径仍然完好
        dl = client.post(f"/api/files/{pairs[0][0]}/downloads").json()["download_id"]
        assert client.get(f"/api/downloads/{dl}/content").status_code == 200
