<p align="right">
  <b>Sandstone</b> | <a href="./Origin_City.md">City</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <b>English</b> | <a href="./Origin_zh.md">中文</a>
</p>

<div align="center">

# ATRNet-STAR Subset – Sandstone (30° Grazing Angle)

<p align="center">
  <a href="https://github.com/waterdisappear/ATRNet-STAR"><img src="https://img.shields.io/badge/Dataset-ATRNet--STAR-blue"></a>
  <a href="https://huggingface.co/datasets/waterdisappear/ATRNet-STAR"><img src="https://img.shields.io/badge/Raw%20data-Hugging%20Face-yellow"></a>
  <a href="https://arxiv.org/abs/2501.13354"><img src="https://img.shields.io/badge/Paper-arXiv-red"></a>
</p>


<p align="center">
  <a href="#Data-Overview">📖 Overview</a> |
  <a href="#Folder-Structure">📁 Structure</a> |
  <a href="#Assistfile-layout">🧭 Assistfile</a> |
  <a href="#FAQ">❓ FAQ</a> |
  <a href="#Contact">📜 Contact</a>
</p>


</div>

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
├── Origin.md
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
                = header + pulse_len × (I1,Q1,I2,Q2 as int16)
```

Example (STR1): `pulse_len = 3280` → **26 368 bytes/pulse**.

---

## Important notes

- **License:** CC BY-NC 4.0 (same as ATRNet-STAR). No redistribution without author consent.
- **AUX duplication:** Per-strip `AUX_*.dat/.log` exist in both `Assistfile/` and `Result/` (same as original data layout).
- **AssistData_*.dat:** Large mission-level files; use per-pol `AUX_KuSAR_*.dat` for strip-specific processing.

## Contact

Weijie Li — Postdoctoral Researcher, NUDT — lwj2150508321@sina.com

Last updated: 2026-07-05
