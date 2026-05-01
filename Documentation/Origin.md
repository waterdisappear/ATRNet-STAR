# ATRNet-STAR Subset – Sandstone (30° Grazing Angle, 0°–330° Azimuth, 30° Step)

This subset contains 12 Ku-band fully polarimetric SAR acquisitions over a sandstone scene with a grazing angle of 30° and azimuth angles spanning 0°–330° in 30° increments (12 azimuth settings). It is provided as a sample for researchers who need large image, raw echoes, and auxiliary files across multiple viewing directions.

## Data Overview
- Target scene: Sandstone (terrain type)
- Grazing angle: 30°
- Azimuth angles: 0°, 30°, 60°, 90°, 120°, 150°, 180°, 210°, 240°, 270°, 300°, 330° (30° step; 12 files/sets)
- Band: Ku
- Polarizations: HH, HV, VH, VV
- Strip IDs: STR1–STR12 (see per-azimuth subfolders)
- Azimuth starting pulse: 1024 (as indicated by azbias1024 in filenames)

## Folder Structure

Origin/
├── Subset_Sandstone/
│   ├── Annotation/   # Annotation files (per azimuth folder, target bounding boxes, categories, etc.)
│   ├── Assistfile/   # Auxiliary data (per azimuth folder; radar parameters, trajectory, attitude)
│   ├── Rawfile/      # Raw echo data (per azimuth folder; binary, complex)
│   └── Result/       # Focused products (per azimuth folder; GeoTIFF/SLC)
├── Scripts/          # MATLAB scripts to read and visualise the data
└── Readme.md         # This file

Each data folder under `Subset_Sandstone/` is organized by azimuth, e.g. `30deg_0azi_ID1`, `30deg_30azi_ID7`, …, `30deg_330azi_ID12`.

## Detailed File Descriptions

### 1. Annotation/
Contains annotation files that provide ground‑truth labels for the targets in the scene.

| File type | Description                                                  |
| --------- | ------------------------------------------------------------ |
| `*.xml`   | Target‑level annotations (e.g., bounding box coordinates, category labels). The exact format follows the ATRNet‑STAR dataset specification. For details, refer to the official dataset documentation or contact the authors. |

### 2. Assistfile/
Contains auxiliary files that describe radar system parameters, platform trajectory, and attitude.

| File pattern                       | Description                                                  |
| ---------------------------------- | ------------------------------------------------------------ |
| `AUX_KuSAR_*_STR*_azbias1024.dat`   | Binary parameter file (radar settings, geometry, Doppler, etc.). Read with `ParaRead.m`. |
| `AUX_KuSAR_*_STR*_azbias1024.log`   | Text log file (may contain additional metadata).             |
| `RAW_XSAR_*_STR*_azbias1024*.xml`   | XML metadata describing the raw data acquisition (pattern may vary by azimuth folder). |

**Polarization variants** (replace `*` with `H1H1`, `H1V1`, `V1H1`, `V1V1`):
- `H1H1` → HH
- `H1V1` → HV
- `V1H1` → VH
- `V1V1` → VV

### 3. Rawfile/
Contains the raw radar echo data (before range compression or focusing).

| File pattern              | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `RAW_XSAR_*_STR*.dat`     | Raw echo data in binary format (complex, interleaved I/Q). Use `RawDataRead.m` to inspect. |

### 4. Result/
Contains focused SAR products: GeoTIFF images (amplitude) and Single Look Complex (SLC) files.

| File type   | Example                               | Description                                  |
| ----------- | ------------------------------------- | -------------------------------------------- |
| `DOM_*.tif` | `DOM_KuSAR_H1H1_STR*_azbias1024.tif`  | Ground‑range detected image (GeoTIFF).       |
| `IMG_*.tif` | `IMG_KuSAR_H1H1_STR*_azbias1024.tif`  | Slant‑range detected image (GeoTIFF).        |
| `*.slc`     | `SLC_KuSAR_H1H1_STR*_azbias1024.slc`  | Single Look Complex (phase‑preserving) data. |

### 5. Scripts/
MATLAB helper scripts to read and visualise the data. All scripts are provided with English comments.

| Script          | Purpose                                                      |
| --------------- | ------------------------------------------------------------ |
| `ParaRead.m`    | Read an AUX parameter file (`.dat`). Displays radar parameters, trajectory, and attitude. Generates plots. |
| `POSRead.m`     | Read a POS binary file (`.out`) to extract altitude, yaw, pitch, roll, latitude, longitude. |
| `RawDataRead.m` | Read the raw echo file (`.dat`) and plot time‑domain / frequency‑domain signals. |

**Basic usage example** (inside MATLAB):
```matlab
% 1. Read AUX file
ParaRead;   % modify the 'file' variable inside the script to point to your AUX file

% 2. Read raw echo
RawDataRead; % change the fid path to your raw data file

% 3. Read POS data
POSRead;     % change the fid path to your POS file
```

## Important Notes
- Data usage: This subset is released under the same terms as the full ATRNet-STAR dataset (CC BY-NC 4.0). Redistribution to third parties is prohibited without written consent from the authors.
- File naming: The pattern azbias1024 indicates that the data start at pulse index 1024 in the azimuth direction. This can be used to align chips with the full scene if needed.

## Contact
For questions or to request additional data (e.g., more strips, or raw data subsets), please contact:

Weijie Li
Postdoctoral Researcher
National University of Defense Technology (NUDT)
Email: lwj2150508321@sina.com

Last updated: 2026-04-30
