<p align="right">
  <a href="./Origin_zh.md">沙地</a> | <b>城区</b>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="./Origin_City.md">English</a> | <b>中文</b>
</p>

<div align="center">

# ATRNet-STAR 子集 – 城市场景（30° 擦地角，HH）

<p align="center">
  <a href="https://github.com/waterdisappear/ATRNet-STAR"><img src="https://img.shields.io/badge/Dataset-ATRNet--STAR-blue"></a>
  <a href="https://huggingface.co/datasets/waterdisappear/ATRNet-STAR"><img src="https://img.shields.io/badge/Raw%20data-Hugging%20Face-yellow"></a>
  <a href="https://arxiv.org/abs/2501.13354"><img src="https://img.shields.io/badge/Paper-arXiv-red"></a>
</p>

<p align="center">
  <a href="#数据概览">📖 概览</a> |
  <a href="#目录结构">📁 目录</a> |
  <a href="#Assistfile-组织方式">🧭 Assistfile</a> |
  <a href="#条带与文件夹对应">🗺 映射</a> |
  <a href="#联系方式">📜 联系</a>
</p>

</div>

本子集包含城区背景 **X 波段 HH 极化** SAR 数据，**擦地角 30°**，方位角 **0°–355°**（步进 **5°**，共 **72** 条带）。提供辅助参数与聚焦产品（**IMG** + **SLC**）。**不含**目标标注与原始回波。

> 与 [沙地子集](./Origin_zh.md) 对比：城区为 **X 波段**、方位步进 **5°**（**72** 视角）、仅 **HH**；沙地为 **Ku 波段**、步进 **30°**（**12** 视角）、**全极化**，并含 Annotation / Raw / Pos。

---

## 数据概览

| 项目 | 说明 |
|------|------|
| 场景 | 城区背景 |
| 擦地角 | 30° |
| 方位角 | 0°, 5°, …, 355°（72 条带） |
| 波段 | X（约 9.6 GHz） |
| 极化 | 仅 HH（`H1H1`） |
| 产品 | `IMG_*.tif`、`SLC_*.slc`、Assistfile |
| 不含 | Annotation、Rawfile、Pos、HV/VH/VV、DOM / PGA / Pauli |

### 条带与文件夹对应

完整表见 [`Subset_City/manifest.csv`](./Subset_City/manifest.csv)。

| 方位角 (°) | 文件夹 | ID | STR |
|----------:|--------|---:|----:|
| 0 | `30deg_0azi_ID1` | 1 | 1 |
| 5 | `30deg_5azi_ID7` | 7 | 7 |
| 10 | `30deg_10azi_ID13` | 13 | 13 |
| 15 | `30deg_15azi_ID19` | 19 | 19 |
| 20 | `30deg_20azi_ID1` | 1 | 1 |
| 25 | `30deg_25azi_ID7` | 7 | 7 |
| 30 | `30deg_30azi_ID13` | 13 | 13 |
| 35 | `30deg_35azi_ID19` | 19 | 19 |
| 40 | `30deg_40azi_ID1` | 1 | 1 |
| 45 | `30deg_45azi_ID7` | 7 | 7 |
| 50 | `30deg_50azi_ID1` | 1 | 1 |
| 55 | `30deg_55azi_ID7` | 7 | 7 |
| 60 | `30deg_60azi_ID2` | 2 | 2 |
| 65 | `30deg_65azi_ID8` | 8 | 8 |
| 70 | `30deg_70azi_ID14` | 14 | 14 |
| 75 | `30deg_75azi_ID20` | 20 | 20 |
| 80 | `30deg_80azi_ID2` | 2 | 2 |
| 85 | `30deg_85azi_ID8` | 8 | 8 |
| 90 | `30deg_90azi_ID14` | 14 | 14 |
| 95 | `30deg_95azi_ID20` | 20 | 20 |
| 100 | `30deg_100azi_ID2` | 2 | 2 |
| 105 | `30deg_105azi_ID8` | 8 | 8 |
| 110 | `30deg_110azi_ID2` | 2 | 2 |
| 115 | `30deg_115azi_ID8` | 8 | 8 |
| 120 | `30deg_120azi_ID3` | 3 | 3 |
| 125 | `30deg_125azi_ID9` | 9 | 9 |
| 130 | `30deg_130azi_ID15` | 15 | 15 |
| 135 | `30deg_135azi_ID21` | 21 | 21 |
| 140 | `30deg_140azi_ID3` | 3 | 3 |
| 145 | `30deg_145azi_ID9` | 9 | 9 |
| 150 | `30deg_150azi_ID15` | 15 | 15 |
| 155 | `30deg_155azi_ID21` | 21 | 21 |
| 160 | `30deg_160azi_ID3` | 3 | 3 |
| 165 | `30deg_165azi_ID9` | 9 | 9 |
| 170 | `30deg_170azi_ID3` | 3 | 3 |
| 175 | `30deg_175azi_ID9` | 9 | 9 |
| 180 | `30deg_180azi_ID4` | 4 | 4 |
| 185 | `30deg_185azi_ID10` | 10 | 10 |
| 190 | `30deg_190azi_ID16` | 16 | 16 |
| 195 | `30deg_195azi_ID22` | 22 | 22 |
| 200 | `30deg_200azi_ID4` | 4 | 4 |
| 205 | `30deg_205azi_ID10` | 10 | 10 |
| 210 | `30deg_210azi_ID16` | 16 | 16 |
| 215 | `30deg_215azi_ID22` | 22 | 22 |
| 220 | `30deg_220azi_ID4` | 4 | 4 |
| 225 | `30deg_225azi_ID10` | 10 | 10 |
| 230 | `30deg_230azi_ID4` | 4 | 4 |
| 235 | `30deg_235azi_ID10` | 10 | 10 |
| 240 | `30deg_240azi_ID5` | 5 | 5 |
| 245 | `30deg_245azi_ID11` | 11 | 11 |
| 250 | `30deg_250azi_ID17` | 17 | 17 |
| 255 | `30deg_255azi_ID23` | 23 | 23 |
| 260 | `30deg_260azi_ID5` | 5 | 5 |
| 265 | `30deg_265azi_ID11` | 11 | 11 |
| 270 | `30deg_270azi_ID17` | 17 | 17 |
| 275 | `30deg_275azi_ID23` | 23 | 23 |
| 280 | `30deg_280azi_ID5` | 5 | 5 |
| 285 | `30deg_285azi_ID11` | 11 | 11 |
| 290 | `30deg_290azi_ID5` | 5 | 5 |
| 295 | `30deg_295azi_ID11` | 11 | 11 |
| 300 | `30deg_300azi_ID6` | 6 | 6 |
| 305 | `30deg_305azi_ID12` | 12 | 12 |
| 310 | `30deg_310azi_ID18` | 18 | 18 |
| 315 | `30deg_315azi_ID24` | 24 | 24 |
| 320 | `30deg_320azi_ID6` | 6 | 6 |
| 325 | `30deg_325azi_ID12` | 12 | 12 |
| 330 | `30deg_330azi_ID18` | 18 | 18 |
| 335 | `30deg_335azi_ID24` | 24 | 24 |
| 340 | `30deg_340azi_ID6` | 6 | 6 |
| 345 | `30deg_345azi_ID12` | 12 | 12 |
| 350 | `30deg_350azi_ID6` | 6 | 6 |
| 355 | `30deg_355azi_ID12` | 12 | 12 |

---

## 目录结构

```
Raw_data/
├── Origin.md / Origin_City.md / Origin_zh.md / Origin_City_zh.md
└── Subset_City/
    ├── Assistfile/     # AUX、采集 XML、任务级 AssistData
    ├── Result/         # 聚焦 IMG / SLC（+ AUX 副本）
    └── manifest.csv    # 方位角 ↔ 文件夹 / 文件名
```

---

## Assistfile/ 组织方式

每个方位角子文件夹含 **HH AUX** 与 **1 个采集 XML**：

| 文件模式 | 说明 |
|---------|------|
| `AUX_XSAR_H1H1_*_azbias*.dat` | 二进制雷达/几何/轨迹 → `ParaRead.m` |
| `AUX_XSAR_H1H1_*_azbias*.log` | 文本摘要 |
| `RAW_XSAR_*_STR*.xml` | 采集元数据 |

**任务级合并辅助文件**（`Assistfile/` 根目录）：

| 子集文件 |
|---------|
| `AssistData_Y20240513-CQ-X-30(0-5-10-15方位角).dat` |
| `AssistData_Y20240513-CQ-X-30(20-25-30-35方位角).dat` |
| `AssistData_Y20240514-CQ-X-30(40-45方位角).dat` |
| `AssistData_Y20240514-CQ-X-30(50-55方位角).dat` |

> **说明：** 分条带 `AUX_*.dat/.log` 在 `Assistfile/` 与 `Result/` 中各有一份（与沙地子集约定一致）。

极化代码：`H1H1` = HH。

---

## Result/ 内容

| 类型 | 说明 |
|------|------|
| `IMG_*.tif` | 斜距检测图像 |
| `SLC_*.slc` | 单视复数据 |
| `AUX_XSAR_*.dat` / `*.log` | 与 Assistfile 相同，副本 |

---

## 重要说明

- **许可：** CC BY-NC 4.0，未经作者同意不得向第三方转发。
- 部分条带 `azbias` 并非 1024（如 `azbias561`），以文件名 / AUX 为准。
- 条带处理优先用分极化 `AUX_XSAR_H1H1_*.dat`；`AssistData_*.dat` 为体积较大的任务级文件。
- 成像 XML 中的 `RangeStartPoint` / `RangePoints` / `AzimuthOffset` / `AzimuthPoints` 描述聚焦任务窗口；分极化 AUX 的 `pulse_num` 与距离向尺寸为聚焦后 HH 图像大小。`GPSLeap` 为 GPS–UTC 闰秒。`IfExcuteMoCo` / `IfExcutePGA` 为任务配置标志；本城区包发布的是常规 `SLC_*` / `IMG_*`（文件名不含 PGA 标记）。详见 [Raw_data.md](./Raw_data.md)。

## 联系方式

李玮杰 — 国防科技大学 — lwj2150508321@sina.com

最后更新：2026-10-15

