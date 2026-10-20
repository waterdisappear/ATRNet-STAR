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

This subset contains 12 Ku-band fully polarimetric SAR acquisitions over a sandstone scene with a **30° grazing angle** and azimuth angles **0°–330°** in 30° steps. Each strip includes annotations, auxiliary parameters, POS trajectory, raw echoes, and focused products.

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

| Pattern | Description |
|---------|-------------|
| `AUX_KuSAR_*_STR*_azbias1024.dat` | Binary radar/geometry/trajectory → `ParaRead.m` |
| `AUX_KuSAR_*_azbias1024.log` | Text summary |
| `RAW_XSAR_*_STR*.xml` | Acquisition / imaging-job metadata (MiniSAR XML) |

**Mission-level merged auxiliary** at `Assistfile/` root:

| File | Description |
|------|-------------|
| `AssistData_ID1-11.dat` | Merged auxiliary for STR1–STR11 |
| `AssistData_ID12.dat` | Merged auxiliary for STR12 |

> **Note:** The same per-strip `AUX_*.dat/.log` files also appear under `Result/` (alongside the imaging products).

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
| `AUX_KuSAR_*.dat` / `*.log` | same as Assistfile | Also present next to imaging products |

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
**Local earth-fixed frame** (platform frame: X = nose, Y = wing, Z = belly). GPS latitude/longitude/altitude are in `lat_ref`, `lng_ref`, `alt_ref` (degrees / degrees / metres).

### Q3: Pulse width Tr and chirp rate Kr?
From AUX: `Tr` (s), `Br` (Hz), `Kr_sign`. Chirp rate:

`Kr = (Br / Tr) * Kr_sign`

`Kr_sign = 1` is positive chirp; `0` is negative. `ParaRead.m` prints these automatically.

### Q4: How do XML / AUX counts relate to RAW and the focused image?
These quantities are **not** interchangeable:

| Source | Field | Meaning (Sandstone STR1 example) |
|--------|-------|-----------------------------------|
| Imaging XML (`RAW_*.xml`) | `Fsr` | Native raw sample rate (**100 MHz**) |
| Imaging XML | `N_RgStart`, `N_RgLen` | Range **window used for focusing** (e.g. 3736, 3280)—not a redefinition of the raw clock |
| Imaging XML | `pulse_num` | Azimuth pulses used by this imaging job (e.g. 45000) |
| Imaging XML | `az_offset` / filename `azbias####` | Start pulse index of the focused product (here **1024**) |
| Per-pol AUX (`AUX_*_H1H1_*.dat`) | `pulse_num`, `pulse_len` | Focused **image** size: azimuth lines × range bins (e.g. 5626 × 3280) |
| Per-pol AUX | `Fsr`, `PRF` | Values shipped with that focused product (STR1 HH: `Fsr ≈ 10.656 MHz`, `PRF ≈ 125 Hz`) |

`azbias1024`: focused image row `i` ↔ imaging-job pulse index `(1024 + i − 1)` for `i = 1 … pulse_num` (AUX).

Per-pol AUX files describe the **separated HH/HV/VH/VV products** after focusing. Their `op_mode` / `pp_mode` follow the product file and are not a re-statement of the mixed RAW stream.

### Q5: Full-pol RAW layout?
- Acquisition is full polarimetric with **ping-pong transmit** and **two receive channels**.
- Four polarizations are formed from alternating H/V transmit pulses (two pulses per pol set). Released `*_H1H1_*` / `*_H1V1_*` / `*_V1H1_*` / `*_V1V1_*` products are already separated.
- On-disk RAW frame format is given in the next section. A byte-level ICD for the 128-byte per-pulse header is not included in this release.

### Q6: What do `flag_MoCo` / `flag_PGA` / `detdelay` in the XML mean?
- `flag_MoCo` / `flag_PGA` (and City-style `IfExcuteMoCo` / `IfExcutePGA`) are **imaging-job configuration flags** in `RAW_*.xml`.
- ATRNet-STAR **releases** the ordinary `SLC_*` / `IMG_*` / `DOM_*` products in `Result/` (no “with PGA” filename tag). Files whose names explicitly mark PGA are PGA products; those are outside this release package.
- In the Sandstone MiniSAR XML, **`detdelay = 18`** is the GPS–UTC leap-second offset (same role as `GPSLeap` in the City-style XML), not a range delay in samples.

### Q7: Phase across products / views?
Released SLCs keep the focused complex phase of the imaging chain for that product. This release does not add a separate global phase-standardization layer or a cross-strip absolute phase reference. Multi-view chip sets (`Slant_Range`) share physical vehicle identities via type names / `target_id`; phase coherence is defined within each focused image.

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
| 7 | Fsr | double | Hz (see product AUX / XML notes above) |
| 8 | Tr | double | s (pulse width) |
| 9 | theta_bw | double | rad |
| 10 | Ba | double | Hz |
| 11 | PRF | double | Hz |
| 12 | pulse_num | int64 | focused-image azimuth lines |
| 13 | pulse_len | int64 | focused-image range bins |
| 14 | amp_bit | int64 | amplitude image bit depth |
| 15–16 | Xbin, Rbin | double | m (azimuth / range pixel spacing) |
| … | geometry, Doppler, APC, scene corners | … | see `ParaRead.m` |
| — | trajectory arrays | double[] | 7×`pulse_num` reference + 10×`pulse_num_orig` high-rate |

`Rmin` in AUX is the slant range of the first range gate **after** pulse compression.

---

## Raw echo packet format (Sandstone Ku RAW `.dat`)

On-disk recording uses a **fixed frame length** set by the acquisition sample rate and pulse width in the imaging XML (`Fsr = 100e6`, `Tr = 5e-4`):

```
range samples per pulse  = Fsr × Tr = 50000
bytes per pulse          = 128 + 50000 × 4 = 200128
                         = [128-byte header] + dual-channel int16 payload
```

- **Endianness:** little-endian (`flag_endian = 0`).
- After the 128-byte header, the payload is two receive channels as interleaved `int16` (see `RawDataRead.m`).
- **`N_RgStart` / `N_RgLen`** select the range window for focusing inside those 50000 samples; focused AUX `pulse_len` equals `N_RgLen` (STR1: 3280). That window length is **not** the on-disk RAW frame length.
- `RawDataRead.m` reads one RAW frame for echo inspection; it does not implement the full focus chain.

---

## Important notes

- **License:** CC BY-NC 4.0 (same as ATRNet-STAR). No redistribution without author consent.
- **AUX duplication:** Per-strip `AUX_*.dat/.log` exist in both `Assistfile/` and `Result/`.
- **AssistData_*.dat:** Large mission-level files; strip-level work uses per-pol `AUX_KuSAR_*.dat`.

## Contact

Weijie Li — Postdoctoral Researcher, NUDT — lwj2150508321@sina.com

Last updated: 2026-10-20
