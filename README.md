# 科研影像加密归档（单课题 · 单进程）

FastAPI + SQLite + `cryptography` + `numpy` 实现的单课题 TIFF 加密归档网页：每件影像以独立
随机数据密钥（DEK）做 AES-256-GCM 加密，DEK 再由课题密钥（KEK，AES-256-GCM）封装后入库；
密文与课题密钥**分目录持久保存**，原件明文与数据密钥**绝不落盘**。支持上传、下载、
课题密钥轮换及进度查看，轮换期间上传下载照常，重启自动续办。另提供**同视野比对**：
在网页选择两件已归档影像，内存解密后比较是否拍到同一视野，给出最佳位移、重叠范围、
相似度与“同视野 / 不同视野 / 无法判定”结论。

## 一、快速开始

### 方式 A：Docker Compose（推荐）

```bash
docker compose up --build
# 浏览器打开
open http://localhost:8000/        # macOS
# 或直接访问 http://localhost:8000/
```

数据全部持久化在命名卷 `archive-data` 内（容器内 `/data`）：

```
/data
├── keys/      # 课题密钥文件 *.kek（权限 600），与密文分目录
├── vault/     # 已归档密文 *.blob（nonce || GCM 密文+tag）
├── staging/   # 上传暂存密文（登记成功后原子移入 vault/，重启清理孤儿）
└── archive.db # SQLite（WAL）：文件/上传/下载/轮换元数据与“封装后的 DEK”
```

停止与清理：

```bash
docker compose down            # 停止，保留数据卷
docker compose down -v         # 同时删除数据卷（清空全部归档）
```

### 方式 B：本地 Python 运行（无需 Docker）

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
ARCHIVE_DATA_DIR=./data uvicorn app.main:app --host 127.0.0.1 --port 8000
# 浏览器入口：http://127.0.0.1:8000/
```

## 二、浏览器入口能做什么

打开 `http://localhost:8000/`（单页应用，原生 JS，无构建步骤）：

1. **课题密钥与轮换**：查看当前 KEK、`keys/` 目录状态，一键发起轮换，实时进度
   （重封装件数、阻塞中的上传/下载数、旧钥文件是否已删除）；
2. **演示控制（暂停点）**：可分别暂停/恢复“上传待登记”和“下载待解封”两个闸门；
3. **上传**：选择 `.tif/.tiff`（每件 ≤ 8 MiB），页面展示三阶段状态机；
4. **同视野比对**：从下拉框选择两件已归档影像，一键比对（详见下节）；
5. **已归档影像与下载**：查看每件影像当前封装在哪个 KEK 下，发起两阶段下载；
6. **操作日志**：所有关键动作带时间戳输出。

### 同视野比对怎么看结论

仅接受**尺寸相同、边长 64～512 像素的单页、未压缩、8 位灰度 TIFF**；不满足条件的影像
不会被强行比较，而是逐件给出中文原因（如多页、压缩、非 8 位、多通道、瓦片、边长越界）。

- **方法**：内存解密后用归一化互相关（NCC，皮尔逊相关系数）在水平、垂直各 ±8 px 的
  位移网格（17×17=289 个候选）上搜索，NCC 对整体亮度的线性变化（增益/偏置）天然不敏感；
- **输出**：最佳位移 `(dy, dx)`（约定 B 相对 A 的平移，`A[r,c]` 对齐 `B[r-dy,c-dx]`）、
  最佳位移下两件影像各自的重叠行列切片（半开区间）与重叠尺寸、NCC 相似度、与次优位移的
  得分差、纹理充分性指标、位移得分前 5 名；
- **三态结论**：
  - `同视野 same_fov`：最佳 NCC ≥ 0.90 **且** 与次优位移差距 ≥ 0.05（峰值唯一）；
  - `不同视野 different_fov`：影像可比较，但最佳 NCC 低于阈值；
  - `无法判定 undetermined`：影像不受支持、尺寸不同、大片纯色/有效纹理不足
    （百分位拉伸后纹理像素占比 < 4%）、或多个位移得分难分——这些情形**一律不得误报同视野**。
- **比对期间可照常轮换课题密钥**：解密只在服务方法的锁内短暂完成，NCC 搜索在锁外执行，
  不阻塞轮换线程；比对记录固化两件原件的 SHA-256，结论可追溯到“本次读取的两件原件”。

## 三、本地演示脚本

脚本自动启动服务并完整演示题目要求的三类场景：

```bash
pip install -r requirements.txt
PORT=8090 ./scripts/demo.sh /tmp/demo
```

演示内容：

| 场景 | 演示断言 |
| --- | --- |
| 正常上传/下载 | 逐字节一致；vault/staging 无明文 TIFF 魔数；密钥与密文分目录 |
| **上传待登记暂停** | 闸门关闭后上传停在 `pending_register`（DEK 仅在内存）；轮换停在收尾且 `blocking_uploads=1`；恢复后该上传由**新 KEK** 封装，不留下旧钥引用 |
| **下载待解封暂停** | 已受理下载停在 `accepted`；轮换 `blocking_downloads=1` 且旧 KEK 文件保留；恢复后在途下载读完原件，轮换才完成 |
| **轮换中重启** | 轮换未完成时退出；重启后自动续办至完成、未完成上传按规则失败并清理暂存、全部已归档文件仍可解密且与原件一致 |
| **同视野比对** | ±8px 内平移与整体亮度变化判同视野并给出位移/重叠/相似度；无关纹理判不同视野；压缩影像判无法判定并说明原因；比对进行中轮换照常完成且结论正确 |

受限沙箱（禁止 `kill` 系统调用）中用“同一数据目录 + 新端口新进程”模拟重启：

```bash
PORT=8092 PORT2=8093 RESTART=report ./scripts/demo.sh /tmp/demo
```

恢复逻辑与进程重启完全相同（都是重新执行 `Service._recover()`），仅进程替换方式不同。

## 四、测试

```bash
pip install -r requirements-dev.txt
pytest -q
```

29 个端到端测试（FastAPI TestClient + 真实 SQLite/文件系统），覆盖：明文不落盘不变量、
8 MiB 上限、轮换只重封装 DEK 且密文逐字节不变、旧 KEK 文件删除后才报完成、两个暂停点、
在途下载读完原件、同期只允许一次轮换、轮换中重启续办、重启清理暂存；以及同视野比对的
TIFF 严格解码（II/MM 双字节序、条带装配）、平移/亮度/无关/超界位移、纯色与弱纹理防误报、
周期纹理多位移难分防误报、不支持影像原因说明、比对历史与原件 SHA-256 绑定、
比对与轮换并发、比对全程明文/DEK 不落盘。

## 五、HTTP API 摘要

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/uploads` | 受理上传（filename、size，>8 MiB 返回 413） |
| PUT | `/api/uploads/{id}/content` | 上传密文前置：服务端用内存随机 DEK 加密并暂存 |
| POST | `/api/uploads/{id}/complete` | **待登记暂停点** → 用当前 KEK 封装 DEK 并登记归档 |
| POST | `/api/uploads/{id}/abort` | 中止上传并清理暂存 |
| POST | `/api/files/{file_id}/downloads` | 受理下载，快照 KEK 标识与封装 DEK |
| GET | `/api/downloads/{id}/content` | **待解封暂停点** → 解封并读完原件，返回明文 |
| POST | `/api/downloads/{id}/abort` | 中止下载 |
| POST | `/api/rotations` | 发起轮换（已有轮换时 409） |
| GET | `/api/rotations/current` | 轮换状态与进度（页面 1.5s 轮询） |
| POST | `/api/comparisons` | 同视野比对：内存解密两件影像后返回结论/位移/重叠/相似度 |
| GET | `/api/comparisons` | 比对历史（含两件原件 SHA-256 与完整指标 JSON） |
| GET | `/api/files` `/api/keys` `/api/uploads` `/api/downloads` | 列表查询 |
| POST | `/api/demo/gates/{upload_register,download_unwrap}` | 演示暂停门 `{"paused": true/false}` |

## 六、设计要点与安全不变量

1. **两级密钥**：每件影像独立随机 DEK（32 B，仅存在于进程内存）；KEK（32 B）保存在
   `keys/*.kek`（带魔数头、权限 600、原子写 + fsync）。封装与原件加密使用不同的 GCM
   nonce 与 AAD，防止消息类型混用。
2. **明文不落盘**：上传内容读取后立即在内存加密，临时文件与最终 blob 均为
   `nonce || 密文||tag`；下载明文只随 HTTP 响应返回，不写磁盘。
3. **轮换只动封装**：`files.blob_path/nonce`（原件密文）永不修改，仅
   `unwrap(旧KEK) → wrap(新KEK)` 更新 `wrapped_dek/dek_nonce/current_kek_id`，每件独立
   事务提交，天然可中断、可续办。
4. **同期只轮换一次**：存在 `running/completing` 轮换时再次发起直接 409；旧 KEK 发起
   瞬间置 `rotating`，新上传恒封装到新 active KEK，**收尾后不存在旧钥引用**。
5. **完成条件（全部满足才报完成）**：无文件引用旧 KEK、无待登记/在途上传、无在途下载、
   数据库无残留的旧 KEK 封装材料，最后**实际删除旧 KEK 文件**才写 `completed`。
6. **在途下载可读**：受理下载即快照 KEK 标识与封装 DEK，与 `files` 行随后的重封装解耦；
   下载完成后立即清空快照封装材料。
7. **暂停门与死锁规避**：闸门关闭时会话在检查点等待，等待期间释放全局锁并在唤醒后复查
   状态，避免与闸门控制请求死锁；服务关闭可在 1 个轮询周期内打断等待。
8. **崩溃恢复（`_recover`）**：未完成上传（内存 DEK 已随进程消失）标记 `failed` 并清理
   暂存；未完成下载标记 `interrupted`；清理孤儿暂存；未完成轮换重置为 `running` 并由
   后台线程续办；已归档文件的 DEK 始终能被某个未删除的 KEK 解封，因此**始终可解密**。
9. **并发模型**：单进程多线程，所有 SQLite 访问与共享状态在一把可重入锁内串行化，
   SQLite 另开 WAL + `busy_timeout` 兜底；密钥文件、blob 均为原子替换写入。
10. **同视野比对的内存边界与防误报**：比对在锁内仅完成“解封 DEK → 内存解密 → SHA-256
    复核”，随后释放锁，TIFF 解码与 289 个位移的 NCC 搜索全部在锁外、进程内存中进行，
    既不阻塞轮换收尾（旧 KEK 可按既有规则删除），也不产生任何临时文件；`comparisons`
    表只存元数据、指标 JSON 与两件原件的 SHA-256，不含像素或密钥材料。纯色/弱纹理先于
    相关性判定拦截，峰值唯一性（最佳与次优差距）再拦一道，周期性内容不会误报同视野。

## 七、目录结构

```
app/
├── crypto.py          # AES-256-GCM 原语：DEK/KEK 生成、封装/解封、密钥文件原子读写
├── db.py              # SQLite schema（project_keys/files/uploads/downloads/rotations/comparisons）
├── imaging.py         # 基线 TIFF 严格解码 + NCC 位移搜索、纹理充分性与三态判定
├── service.py         # 状态机、轮换工作线程、暂停门、崩溃恢复、同视野比对（核心）
├── main.py            # FastAPI 路由与 lifespan
└── static/index.html  # 浏览器单页应用
scripts/demo.sh                 # 四类场景的本地演示脚本
scripts/make_sample_tiff.py     # 纯标准库的演示 TIFF 生成器（平移/亮度/压缩标签注入）
tests/test_archive.py           # 9 个归档/轮换端到端测试
tests/test_comparison.py        # 20 个同视野比对端到端测试（含真实 TIFF 构造器）
Dockerfile, docker-compose.yml
```
