<p align="right">
  <a href="./Origin.md">Sandstone</a> | <b>City</b>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <b>English</b> | <a href="./Origin_City_zh.md">中文</a>
</p>

<div align="center">

# ATRNet-STAR Subset – City (30° Grazing Angle, HH)

<p align="center">
  <a href="https://github.com/waterdisappear/ATRNet-STAR"><img src="https://img.shields.io/badge/Dataset-ATRNet--STAR-blue"></a>
  <a href="https://huggingface.co/datasets/waterdisappear/ATRNet-STAR"><img src="https://img.shields.io/badge/Raw%20data-Hugging%20Face-yellow"></a>
  <a href="https://arxiv.org/abs/2501.13354"><img src="https://img.shields.io/badge/Paper-arXiv-red"></a>
</p>

<p align="center">
  <a href="#Data-Overview">📖 Overview</a> |
  <a href="#Folder-Structure">📁 Structure</a> |
  <a href="#Assistfile-layout">🧭 Assistfile</a> |
  <a href="#Strip--folder-mapping">🗺 Mapping</a> |
  <a href="#Contact">📜 Contact</a>
</p>

</div>

This subset contains **72** X-band **HH-polarization** SAR acquisitions over an urban / city scene (43 vehicles) with a **30° grazing angle** and azimuth angles **0°–355°** in **5°** steps. Each strip includes auxiliary parameters and focused products (**IMG** + **SLC**). Annotation and raw echo data are **not** included.

> Compared with [Subset_Sandstone](./Origin.md): City uses **X-band**, **5°** azimuth sampling (**72** views), and **HH only**; Sandstone uses **Ku-band**, **30°** sampling (**12** views), and **quad-pol** with Annotation / Raw / Pos.

---

## Data Overview

| Item | Value |
|------|-------|
| Scene | City / 城区背景 |
| Grazing / depression angle | 30° |
| Azimuth | 0°, 5°, …, 355° (72 strips) |
| Band | X (~9.6 GHz) |
| Polarization | HH only (`H1H1`) |
| Products | `IMG_*.tif`, `SLC_*.slc`, Assistfile |
| Not included | Annotation, Rawfile, Pos, HV/VH/VV, DOM / PGA / Pauli |

### Strip ↔ folder mapping

Full table also in [`Subset_City/manifest.csv`](./Subset_City/manifest.csv).

| Azimuth (°) | Folder | ID | STR |
|------------:|--------|---:|----:|
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

## Folder Structure

```
Raw_data/
├── Origin.md / Origin_City.md / Origin_zh.md / Origin_City_zh.md
└── Subset_City/
    ├── Assistfile/     # AUX, acquisition XML, mission AssistData
    ├── Result/         # focused IMG / SLC (+ AUX copy)
    └── manifest.csv    # azimuth ↔ folder / file names
```

---

## Assistfile/ layout

Each azimuth subfolder contains **HH AUX** files and **one acquisition XML**:

| Pattern | Description |
|---------|-------------|
| `AUX_XSAR_H1H1_*_azbias*.dat` | Binary radar/geometry/trajectory → `ParaRead.m` |
| `AUX_XSAR_H1H1_*_azbias*.log` | Text summary |
| `RAW_XSAR_*_STR*.xml` | Acquisition metadata |

**Mission-level merged auxiliary** at `Assistfile/` root:

| File in subset |
|----------------|
| `AssistData_Y20240513-CQ-X-30(0-5-10-15方位角).dat` |
| `AssistData_Y20240513-CQ-X-30(20-25-30-35方位角).dat` |
| `AssistData_Y20240514-CQ-X-30(40-45方位角).dat` |
| `AssistData_Y20240514-CQ-X-30(50-55方位角).dat` |

> **Note:** Per-strip `AUX_*.dat/.log` are also copied into `Result/` (same convention as the sandstone subset).

Polarization code: `H1H1` = HH.

---

## Result/ layout

| Type | Example | Description |
|------|---------|-------------|
| `IMG_*.tif` | `IMG_XSAR_H1H1_STR1_azbias1024.tif` | Slant-range detected image |
| `SLC_*.slc` | `SLC_XSAR_H1H1_STR1_azbias1024.slc` | Single-look complex data |
| `AUX_XSAR_*.dat` / `*.log` | same as Assistfile | Duplicate copy |

---

## Important notes

- **License:** CC BY-NC 4.0 (same as ATRNet-STAR). No redistribution without author consent.
- Some strips use non-1024 `azbias` tags (e.g. `azbias561`); the starting pulse is given by the filename / AUX.
- Prefer per-strip `AUX_XSAR_H1H1_*.dat` for geometry; mission `AssistData_*.dat` files are large merged auxiliaries.
- Imaging XML fields such as `RangeStartPoint` / `RangePoints` / `AzimuthOffset` / `AzimuthPoints` describe the focusing job window; per-pol AUX `pulse_num` × range size is the focused HH image. `GPSLeap` is the GPS–UTC leap-second offset. `IfExcuteMoCo` / `IfExcutePGA` are job flags; this City package releases ordinary `SLC_*` / `IMG_*` (no PGA filename tag). See also [Raw_data.md](./Raw_data.md).

## Contact

Weijie Li — Postdoctoral Researcher, NUDT — lwj2150508321@sina.com

Last updated: 2026-10-15

