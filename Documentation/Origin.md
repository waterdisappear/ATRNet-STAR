# ATRNet-STAR Subset – Sandstone (30° Grazing Angle)

This subset contains 12 Ku-band fully polarimetric SAR acquisitions over a sandstone scene (43 vehicles) with a **30° grazing angle** and azimuth angles **0°–330°** in 30° steps. Each strip includes annotations, auxiliary parameters, POS trajectory, raw echoes, and focused products.

---

## Data Overview

| Item | Value |
|------|-------|
| Scene | Sandstone / 沙地背景 |
| Grazing angle | 30° |
| Azimuth | 0°, 30°, …, 330° (12 strips, STR1–STR12) |
| Band | Ku (~14.6 GHz) |
| Polarizations | HH, HV, VH, VV |
| Azimuth start pulse | **1024** (`azbias1024` in filenames) |

### Strip ↔ folder mapping

| Strip | Folder | POS file |
|-------|--------|----------|
| STR1 | `30deg_0azi_ID1` | `Pos/ID1-11/export_Mission 1.out` |
| STR2 | `30deg_60azi_ID2` | ID1-11 |
| STR3 | `30deg_120azi_ID3` | ID1-11 |
| STR4 | `30deg_180azi_ID4` | ID1-11 |
| STR5 | `30deg_240azi_ID5` | ID1-11 |
| STR6 | `30deg_300azi_ID6` | ID1-11 |
| STR7 | `30deg_30azi_ID7` | ID1-11 |
| STR8 | `30deg_90azi_ID8` | ID1-11 |
| STR9 | `30deg_150azi_ID9` | ID1-11 |
| STR10 | `30deg_210azi_ID10` | ID1-11 |
| STR11 | `30deg_270azi_ID11` | ID1-11 |
| STR12 | `30deg_330azi_ID12` | `Pos/ID12/export_Mission 1.out` |

---

## Folder Structure

```
Raw_data/
├── Readme.md
├── Scripts/            # MATLAB scripts (ParaRead, POSRead, RawDataRead)
└── Subset_Sandstone/
    ├── Annotation/     # target bounding-box labels (.xml)
    ├── Assistfile/     # AUX, acquisition XML, mission AssistData
    ├── Pos/            # Applanix POS (.out)
    ├── Rawfile/        # raw echo (.dat), one file per strip
    └── Result/         # focused DOM/IMG/SLC products (.tif/.slc)
```

---

## Assistfile/ layout

Each azimuth subfolder contains **per-polarization AUX files** and **one acquisition XML**:

| Pattern | Description | Copied from (original data) |
|---------|-------------|----------------------------|
| `AUX_KuSAR_*_STR*_azbias1024.dat` | Binary radar/geometry/trajectory → `ParaRead.m` | Original `result/IDn/` |
| `AUX_KuSAR_*_azbias1024.log` | Text summary | Original `result/IDn/` |
| `RAW_XSAR_*_STR*.xml` | Acquisition metadata (MiniSAR XML) | Original `result/IDn/` |

**Mission-level merged auxiliary** at `Assistfile/` root:

| File in subset | Original source | Status |
|----------------|-----------------|--------|
| `AssistData_ID1-11.dat` | `AssistFile/AssistData.dat` | **Copied** (~865 MB) |
| `AssistData_ID12.dat` | `AssistFile-ID12/AssistData.dat` | **Copied** (~839 MB) |

> **Note:** The same per-strip `AUX_*.dat/.log` files are also copied into `Result/` (duplicate of `Assistfile/`, matching the original `result/` layout).
>
> **Verified layout:** each `Assistfile/` azimuth folder has 9 files (4× `.dat`, 4× `.log`, 1× `.xml`); root has two `AssistData_*.dat` files.

Polarization codes: `H1H1`=HH, `H1V1`=HV, `V1H1`=VH, `V1V1`=VV.

---

## Other folders

### Annotation/
Target labels (`*.xml`) for DOM and IMG products (bounding boxes, categories).

### Rawfile/
One `RAW_XSAR_*_STR*.dat` per strip. Binary raw echoes before focusing.

### Result/
| Type | Example | Description |
|------|---------|-------------|
| `DOM_*.tif` | `DOM_KuSAR_H1H1_STR1_azbias1024.tif` | Ground-range detected image |
| `DOM_*_float.tif` | same base name + `_float` | Float-amplitude ground-range image |
| `IMG_*.tif` | `IMG_KuSAR_H1H1_STR1_azbias1024.tif` | Slant-range detected image |
| `SLC_*.slc` | `SLC_KuSAR_H1H1_STR1_azbias1024.slc` | Single-look complex data |
| `AUX_KuSAR_*.dat` / `*.log` | same as Assistfile | Duplicate copy from original `result/IDn/` (17 files per strip with imaging products) |

### Pos/
Applanix POS export: `export_Mission 1.out` (+ optional `.log`). STR1–STR11 share one file; STR12 uses a separate acquisition day file.

---

## MATLAB scripts

All scripts are in **`Scripts/`**. In MATLAB, `cd` to that folder, then run:

```matlab
ParaRead      % edit 'file' path to your AUX .dat
POSRead       % set stripId = 1..12
RawDataRead   % set stripId and pulseIndex (default 1024 = azbias1024)
```

Comments are **English + Chinese** (UTF-8). If MATLAB shows garbled Chinese, set Preferences > MATLAB > General > UTF-8.

---

## FAQ

### Q1: Where is the POS `.out` file?
Under `Subset_Sandstone/Pos/`. Use `POSRead.m`. High-rate INS/GPS is also summarized in AUX trajectory fields (`lat_ref`, `lng_ref`, `alt_ref`).

### Q2: What coordinate system are x, y, z in ParaRead?
**Local earth-fixed frame** (North-East-Down style platform frame: X = nose, Y = wing, Z = belly). GPS latitude/longitude/altitude are in `lat_ref`, `lng_ref`, `alt_ref` (degrees / degrees / metres).

### Q3: Pulse width Tr and chirp rate Kr?
Read from AUX: field `Tr` (s), `Br` (Hz), `Kr_sign`. Compute:

`Kr = (Br / Tr) * Kr_sign`

`ParaRead.m` prints these automatically.

### Q4: How do AUX parameters align with raw echoes?
- `pulse_num` in AUX = azimuth lines in the **focused image**.
- Filename tag `azbias1024` → **image row 1 ↔ raw echo pulse 1024**.
- General rule: **image row i ↔ raw pulse (1024 + i − 1)** for i = 1 … `pulse_num`.
- Each raw pulse: **128-byte header** (platform pose) + `pulse_len` × (I1, Q1, I2, Q2) int16 samples.

### Q5: Full-pol raw data layout?
- `op_mode = 1` (full polarimetric), `pp_mode = 1` (ping-pong transmit).
- Dual RF channels, sample order: **I1, Q1, I2, Q2** (int16).
- Four polarizations reconstructed from alternating H/V transmit pulses (two pulses per pol set).

---

## AUX `.dat` header (512 bytes, summary)

| # | Field | Type | Unit / note |
|---|-------|------|-------------|
| 1 | op_mode | int64 | 0 single / 1 full-pol / 2 interferometric |
| 2 | pp_mode | int64 | 0 standard / 1 ping-pong |
| 3 | Kr_sign | int64 | 0 negative / 1 positive chirp |
| 4 | fc | double | Hz |
| 5 | fd | double | Hz (IF) |
| 6 | Br | double | Hz |
| 7 | Fsr | double | Hz |
| 8 | Tr | double | s (pulse width) |
| 9 | theta_bw | double | rad |
| 10 | Ba | double | Hz |
| 11 | PRF | double | Hz |
| 12 | pulse_num | int64 | azimuth lines |
| 13 | pulse_len | int64 | range bins |
| … | (geometry, Doppler, APC, scene corners) | … | see ParaRead.m |
| — | + trajectory arrays | double[] | 7×pulse_num ref + 10×pulse_num_orig high-rate |

---

## Raw echo packet format

```
bytes per pulse = 128 + pulse_len × 8
                = header + pulse_len × 4 int16 values (CH1, CH2, CH1, CH2... alternating)
```

samples per channel = pulse_len × 2

---

## Important notes

- **License:** CC BY-NC 4.0 (same as ATRNet-STAR). No redistribution without author consent.
- **AUX duplication:** Per-strip `AUX_*.dat/.log` exist in both `Assistfile/` and `Result/` (same as original data layout).
- **AssistData_*.dat:** Large mission-level files; use per-pol `AUX_KuSAR_*.dat` for strip-specific processing.

## Contact

Weijie Li — Postdoctoral Researcher, NUDT — lwj2150508321@sina.com

Last updated: 2026-07-05

---

# ATRNet-STAR 子集 – 沙地场景（30° 擦地角）

本子集包含沙地背景（43 辆车）Ku 波段 **全极化** SAR 数据，**擦地角 30°**，方位角 **0°–330°**（步进 30°，共 12 条带 STR1–STR12）。每条带提供标注、辅助参数、POS 轨迹、原始回波及聚焦产品。

---

## 数据概览

| 项目 | 说明 |
|------|------|
| 场景 | 沙地背景 |
| 擦地角 | 30° |
| 方位角 | 0°, 30°, …, 330°（12 条带） |
| 波段 | Ku（约 14.6 GHz） |
| 极化 | HH, HV, VH, VV |
| 方位起始脉冲 | **1024**（文件名 `azbias1024`） |

### 条带与文件夹对应

见上表（STR1→`30deg_0azi_ID1`，…，STR12→`30deg_330azi_ID12`；STR1–11 共用 `Pos/ID1-11/`，STR12 用 `Pos/ID12/`）。

---

## 目录结构

```
Raw_data/
├── Readme.md
├── Scripts/            # MATLAB 脚本（ParaRead、POSRead、RawDataRead）
└── Subset_Sandstone/
    ├── Annotation/     # 目标标注 (.xml)
    ├── Assistfile/     # AUX、采集 XML、任务级 AssistData
    ├── Pos/            # Applanix POS (.out)
    ├── Rawfile/        # 原始回波 (.dat)，每条带 1 个
    └── Result/         # 聚焦产品 (.tif / .slc)
```

---

## Assistfile/ 组织方式

每个方位角子文件夹含 **4 极化 AUX** + **1 个采集 XML**：

| 文件模式 | 说明 | 复制来源（原始数据） |
|---------|------|---------------------|
| `AUX_KuSAR_*_STR*_azbias1024.dat` | 二进制雷达/几何/轨迹 → `ParaRead.m` | 原始 `result/IDn/` |
| `AUX_KuSAR_*_azbias1024.log` | 文本摘要 | 原始 `result/IDn/` |
| `RAW_XSAR_*_STR*.xml` | 采集元数据（MiniSAR XML） | 原始 `result/IDn/` |

**任务级合并辅助文件**（`Assistfile/` 根目录）：

| 子集文件 | 原始来源 | 状态 |
|---------|---------|------|
| `AssistData_ID1-11.dat` | `AssistFile/AssistData.dat` | **已复制**（约 865 MB） |
| `AssistData_ID12.dat` | `AssistFile-ID12/AssistData.dat` | **已复制**（约 839 MB） |

> **说明：** 分条带 `AUX_*.dat/.log` 在 `Assistfile/` 与 `Result/` 中各有一份（与原始 `result/` 目录结构一致，允许重复）。
>
> **已核对：** 每个 `Assistfile/` 方位角子文件夹 9 个文件；根目录 2 个 `AssistData_*.dat`。

极化代码：`H1H1`=HH，`H1V1`=HV，`V1H1`=VH，`V1V1`=VV。

---

## 其他目录

### Annotation/
DOM / IMG 对应的目标标注 XML（边界框、类别等）。

### Rawfile/
每条带 1 个 `RAW_XSAR_*_STR*.dat` 原始回波文件。

### Result/
| 类型 | 说明 |
|------|------|
| `DOM_*.tif` | 地距检测图像 |
| `DOM_*_float.tif` | 地距浮点幅度图 |
| `IMG_*.tif` | 斜距检测图像 |
| `SLC_*.slc` | 单视复数据 |
| `AUX_KuSAR_*.dat` / `*.log` | 与 Assistfile 相同，自原始 `result/IDn/` 复制（每条带与成像产品并存） |

### Pos/
Applanix POS 导出文件 `export_Mission 1.out`（及可选 `.log`）。

---

## MATLAB 脚本

脚本均在 **`Scripts/`** 目录。在 MATLAB 中 `cd` 到该目录后运行：

```matlab
ParaRead      % 修改 file 指向 AUX .dat
POSRead       % 设置 stripId = 1..12
RawDataRead   % 设置 stripId 与 pulseIndex（默认 1024）
```

注释为 **中英双语**（UTF-8）。若 MATLAB 中文乱码，请在 预设 > MATLAB > 常规 中启用 UTF-8。

---

## 常见问题（FAQ）

### Q1：POS 的 `.out` 在哪里？
在 `Subset_Sandstone/Pos/`，用 `POSRead.m` 读取。AUX 轨迹段亦含 GPS（`lat_ref/lng_ref/alt_ref`）。

### Q2：ParaRead 中 x,y,z 是什么坐标系？
**大地平面三维坐标系**（X 机头、Y 机翼、Z 机腹）。GPS 经纬高见 `lat_ref`、`lng_ref`、`alt_ref`（度/度/米）。

### Q3：脉冲时宽 Tr 与调频率 Kr？
AUX 中读取 `Tr`、`Br`、`Kr_sign`，计算：**Kr = (Br / Tr) × Kr_sign**。`ParaRead.m` 会自动打印。

### Q4：AUX 与回波脉冲如何对应？
- AUX 中 `pulse_num` = 聚焦图像方位向行数。
- `azbias1024` 表示图像第 1 行对应原始回波第 **1024** 个脉冲。
- 一般：**图像第 i 行 ↔ 回波第 (1024 + i − 1) 个脉冲**。
- 每脉冲：**128 字节包头** + `pulse_len` × (I1,Q1,I2,Q2) int16。

### Q5：全极化回波格式？
- `op_mode=1` 全极化，`pp_mode=1` 乒乓发射。
- 双通道存储顺序：**I1, Q1, I2, Q2**（int16）。
- 四极化由 H/V 交替发射的两脉冲组合得到。

---

## AUX `.dat` 头文件（512 字节，摘要）

| 序号 | 字段 | 类型 | 说明 |
|------|------|------|------|
| 1 | op_mode | int64 | 0 单通道 / 1 全极化 / 2 干涉 |
| 2 | pp_mode | int64 | 0 标准 / 1 乒乓 |
| 3 | Kr_sign | int64 | 0 负调频 / 1 正调频 |
| 4–8 | fc, fd, Br, Fsr, Tr | double | 载频、中频、带宽、采样率、脉宽 |
| 12–13 | pulse_num, pulse_len | int64 | 方位行数、距离点数 |
| … | 几何、多普勒、APC、四角经纬度 | … | 见 ParaRead.m |
| — | 轨迹数组 | double[] | 7×pulse_num 参考 + 10×pulse_num_orig 高采样 |

---

## 原始回波包格式

```
每脉冲字节数 = 128 + pulse_len × 8
             = 包头 + pulse_len × 4 个 int16（按 CH1, CH2, CH1, CH2... 交替排列）
```

每通道采样点数 = pulse_len × 2

---

## 重要说明

- **许可：** CC BY-NC 4.0，未经作者同意不得向第三方转发。
- **AUX 重复存放：** 分条带 `AUX_*.dat/.log` 在 `Assistfile/` 与 `Result/` 中均有（与原始数据一致）。
- **AssistData_*.dat：** 体积较大的任务级文件；条带处理优先用分极化 `AUX_KuSAR_*.dat`。

## 联系方式

李玮杰 — 国防科技大学 — lwj2150508321@sina.com

最后更新：2026-07-05
