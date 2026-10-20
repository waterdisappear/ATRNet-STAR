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

本子集包含沙地背景 Ku 波段 **全极化** SAR 数据，**擦地角 30°**，方位角 **0°–330°**（步进 30°，共 12 条带 STR1–STR12）。每条带提供标注、辅助参数、POS 轨迹、原始回波及聚焦产品。

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

| 文件模式 | 说明 |
|---------|------|
| `AUX_KuSAR_*_STR*_azbias1024.dat` | 二进制雷达/几何/轨迹 → `ParaRead.m` |
| `AUX_KuSAR_*_azbias1024.log` | 文本摘要 |
| `RAW_XSAR_*_STR*.xml` | 采集 / 成像任务元数据（MiniSAR XML） |

**任务级合并辅助文件**（`Assistfile/` 根目录）：

| 文件 | 说明 |
|------|------|
| `AssistData_ID1-11.dat` | STR1–STR11 合并辅助 |
| `AssistData_ID12.dat` | STR12 合并辅助 |

> **说明：** 分条带 `AUX_*.dat/.log` 在 `Assistfile/` 与 `Result/`（成像产品旁）中各有一份。

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
| `AUX_KuSAR_*.dat` / `*.log` | 与 Assistfile 相同，亦存放于成像产品旁 |

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
在 `Subset_Sandstone/Pos/`，用 `POSRead.m` 读取。AUX 轨迹段亦含 GPS（`lat_ref` / `lng_ref` / `alt_ref`）。

### Q2：ParaRead 中 x,y,z 是什么坐标系？
**大地平面三维坐标系**（X 机头、Y 机翼、Z 机腹）。GPS 经纬高见 `lat_ref`、`lng_ref`、`alt_ref`（度/度/米）。

### Q3：脉冲时宽 Tr 与调频率 Kr？
由 AUX 读取 `Tr`（s）、`Br`（Hz）、`Kr_sign`：

`Kr = (Br / Tr) × Kr_sign`

`Kr_sign = 1` 为正调频，`0` 为负调频。`ParaRead.m` 会打印上述量。

### Q4：XML / AUX 中的计数与 RAW、聚焦图像如何对应？
下列字段**含义不同**，不宜混用：

| 来源 | 字段 | 含义（沙地 STR1 示例） |
|------|------|------------------------|
| 成像 XML（`RAW_*.xml`） | `Fsr` | 原始采样率（**100 MHz**） |
| 成像 XML | `N_RgStart`、`N_RgLen` | 聚焦所用**距离窗**（如 3736、3280），不是改写原始采样钟 |
| 成像 XML | `pulse_num` | 本成像任务使用的方位脉冲数（如 45000） |
| 成像 XML | `az_offset` / 文件名 `azbias####` | 聚焦产品起始脉冲号（此处为 **1024**） |
| 分极化 AUX（`AUX_*_H1H1_*.dat`） | `pulse_num`、`pulse_len` | 聚焦**图像**方位行数 × 距离点数（如 5626 × 3280） |
| 分极化 AUX | `Fsr`、`PRF` | 随该聚焦产品给出的数值（STR1 HH：`Fsr ≈ 10.656 MHz`，`PRF ≈ 125 Hz`） |

`azbias1024`：聚焦图第 `i` 行 ↔ 成像任务脉冲号 `(1024 + i − 1)`（`i = 1 …` AUX 的 `pulse_num`）。

分极化 AUX 描述的是聚焦并分离后的 HH/HV/VH/VV **产品**；其中的 `op_mode` / `pp_mode` 随产品文件给出，不表示混合 RAW 流的工作模式重述。

### Q5：全极化 RAW 格式？
- 采集为全极化、**乒乓发射**、**双接收通道**。
- 四极化由 H/V 交替发射的脉冲对形成；已发布的 `*_H1H1_*` 等为分离后的产品。
- 磁盘 RAW 帧格式见下一节。逐脉冲 128 字节包头的字节级字段表不在本公开发布范围内。

### Q6：XML 中 `flag_MoCo` / `flag_PGA` / `detdelay` 是什么？
- `flag_MoCo` / `flag_PGA`（城区风格 XML 中为 `IfExcuteMoCo` / `IfExcutePGA`）是成像任务的**配置标志**。
- ATRNet-STAR **发布**的是 `Result/` 中常规 `SLC_*` / `IMG_*` / `DOM_*`（文件名不含 “with PGA” 标记）。文件名明确标注 PGA 的产品为 PGA 结果，不在本发布包内。
- 沙地 MiniSAR XML 中的 **`detdelay = 18`** 表示 GPS–UTC 闰秒（与城区 XML 的 `GPSLeap` 同义），不是距离向延时采样数。

### Q7：不同产品 / 视角之间的相位？
发布 SLC 保留该产品成像链路的聚焦复相位。本发布未另附全局相位标准化层，亦未提供跨条带绝对相位基准。多视角切片元数据使用共享的类型名 / `target_id` 标签；相位相干性在单幅聚焦图像内定义。

---

## AUX `.dat` 头文件（512 字节，摘要）

| 序号 | 字段 | 类型 | 说明 |
|------|------|------|------|
| 1 | op_mode | int64 | 0 单通道 / 1 全极化 / 2 干涉 |
| 2 | pp_mode | int64 | 0 标准 / 1 乒乓 |
| 3 | Kr_sign | int64 | 0 负调频 / 1 正调频 |
| 4–8 | fc, fd, Br, Fsr, Tr | double | 载频、中频、带宽、采样率、脉宽 |
| 9–11 | theta_bw, Ba, PRF | double | 方位波束宽、多普勒带宽、PRF |
| 12–13 | pulse_num, pulse_len | int64 | 聚焦图像方位行数、距离点数 |
| 14 | amp_bit | int64 | 幅度图像位数 |
| 15–16 | Xbin, Rbin | double | 方位/距离像素间距（m） |
| … | 几何、多普勒、APC、四角经纬度 | … | 见 `ParaRead.m` |
| — | 轨迹数组 | double[] | 7×`pulse_num` 参考 + 10×`pulse_num_orig` 高采样 |

AUX 中 `Rmin` 为**脉压后**第一个距离门对应的斜距。

---

## 原始回波包格式（沙地 Ku RAW `.dat`）

磁盘记录的**固定帧长**由成像 XML 中的采样率与脉宽决定（`Fsr = 100e6`，`Tr = 5e-4`）：

```
每脉冲距离采样数 = Fsr × Tr = 50000
每脉冲字节数     = 128 + 50000 × 4 = 200128
                 = [128 字节包头] + 双通道 int16 载荷
```

- **端序：** 小端（`flag_endian = 0`）。
- 跳过 128 字节包头后，载荷为双接收通道交错的 `int16`（见 `RawDataRead.m`）。
- **`N_RgStart` / `N_RgLen`** 是在上述 50000 点内选取的聚焦距离窗；聚焦 AUX 的 `pulse_len` 等于 `N_RgLen`（STR1：3280）。该窗长**不是**磁盘 RAW 帧长。
- `RawDataRead.m` 用于读取单帧 RAW 作回波检查，不包含完整聚焦流程。

---

## 重要说明

- **许可：** CC BY-NC 4.0，未经作者同意不得向第三方转发。
- **AUX 重复存放：** 分条带 `AUX_*.dat/.log` 在 `Assistfile/` 与 `Result/` 中均有。
- **AssistData_*.dat：** 体积较大的任务级文件；条带处理使用分极化 `AUX_KuSAR_*.dat`。

## 联系方式

李玮杰 — 国防科技大学 — lwj2150508321@sina.com

最后更新：2026-10-20
