<p align="right">
  <b>沙地</b> | <a href="./Origin_City_zh.md">城区</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="./Origin.md">English</a> | <b>中文</b>
</p>

<div align="center">

# ATRNet-STAR 子集 – 沙地场景（30° 擦地角）

<p align="center">
  <a href="https://github.com/waterdisappear/ATRNet-STAR"><img src="https://img.shields.io/badge/Dataset-ATRNet--STAR-blue"></a>
  <a href="https://huggingface.co/datasets/waterdisappear/ATRNet-STAR"><img src="https://img.shields.io/badge/Raw%20data-Hugging%20Face-yellow"></a>
  <a href="https://arxiv.org/abs/2501.13354"><img src="https://img.shields.io/badge/Paper-arXiv-red"></a>
</p>

<p align="center">
  <a href="#数据概览">📖 概览</a> |
  <a href="#目录结构">📁 目录</a> |
  <a href="#Assistfile-组织方式">🧭 Assistfile</a> |
  <a href="#常见问题FAQ">❓ FAQ</a> |
  <a href="#联系方式">📜 联系</a>
</p>

</div>

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
├── Origin.md
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
             = 包头 + pulse_len × (I1,Q1,I2,Q2，各 int16)
```

STR1 示例：`pulse_len=3280` → 每脉冲 **26 368** 字节。

---

## 重要说明

- **许可：** CC BY-NC 4.0，未经作者同意不得向第三方转发。
- **AUX 重复存放：** 分条带 `AUX_*.dat/.log` 在 `Assistfile/` 与 `Result/` 中均有（与原始数据一致）。
- **AssistData_*.dat：** 体积较大的任务级文件；条带处理优先用分极化 `AUX_KuSAR_*.dat`。

## 联系方式

李卫杰 — 国防科技大学 — lwj2150508321@sina.com

最后更新：2026-07-05
