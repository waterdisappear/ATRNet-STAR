### **Radar Data Product Folder Structure**

The radar output is organized into the following main directories:

*   **Assistfile:** Contains auxiliary data, including POS data for positioning and attitude determination.
*   **Rawfile:** Stores the raw radar echo data.
*   **Result:** Holds the final imaging results.
*   **RTI:** Contains real-time imaging results generated onboard the aircraft.

Subset-level notes (Sandstone / City packaging, scripts, strip tables):

*   [Sandstone – English](./Origin.md) · [中文](./Origin_zh.md)
*   [City – English](./Origin_City.md) · [中文](./Origin_City_zh.md)

### **Provided MATLAB Utility Scripts**

Several `.m` files are provided for data analysis and processing:
*   **Data Reading:** `ParaRead.m` (reads imaging parameters/trajectory), `POSRead.m` (reads POS attitude results), `RawDataRead.m` (reads one RAW frame for echo inspection).
*   **Target Analysis:** `read_data_hxy.m`, `point_analysis.m` (analyze point target sidelobe ratios).
*   **Image Calibration & QC:** `RgCal_PhaErrEst.m` (extracts range compensation function), `RgCal.m` (applies range compensation), `RgCal_Test.m` (views compensation results), `SNR.m` (calculates signal-to-noise ratio).

### **Contents of the `Result` Folder**

This folder contains the output from the SAR image formation process. Key files include:

*   **SLC Files (`*.slc`):** Single Look Complex images (slant-range).
    *   **Naming:** `SLC_XSAR_[Pol]_*.slc` / `SLC_KuSAR_[Pol]_*.slc` (e.g. `*_H1H1_*` for HH).
    *   **Data Type:** 32-bit floating point complex (`complex64` / interleaved float32 real–imag).
    *   **Storage:** Row-major: (1,1), (1,2)…(1,N), then (2,1)… through (M,N). Dimensions from the matching `AUX_*.log` / `AUX_*.dat`.
*   **GeoTIFF Images (`*.tif`):** Quantified images.
    *   **Slant-Range Image (`IMG_*.tif`):** Formed image in slant-range geometry.
    *   **Ground-Range Image (`DOM_*.tif`):** After projection/geocoding to ground-range geometry.
*   **PGA-Processed Files:** Only `slc` / `tif` whose **filenames** mark PGA have undergone Phase Gradient Autofocus. The ATRNet-STAR release uses ordinary `SLC_*` / `IMG_*` / `DOM_*` products without that filename tag.
*   **Pauli Decomposition Files:** SLC set (`BLU_*.slc`, `GRE_*.slc`, `RED_*.slc`) and combined pseudo-color `RGB_*.slc` (when present in the original delivery).
*   **Parameter & Log Files:**
    *   `AUX_*.dat`: Parameter file (`ParaRead.m`).
    *   `AUX_*.log`: Image dimension summary (`RangePoints`, `AzimuthPoints`, intervals).
    *   `RAW_*.xml`: Imaging-job configuration (radar parameters, range/azimuth windows, MoCo/PGA flags, etc.).

### **Key Notes on File Naming Conventions**

*   **Polarization:** `H1H1` / `H1V1` / `V1H1` / `V1V1` → HH / HV / VH / VV. `FULL` indicates a polarimetrically synthesized product.
*   **Strip Identifier:** `STRx` identifies the acquisition strip.
*   **Azimuth Bias:** `azbias####` is the starting pulse index of the focused product (e.g. 1024). Focused row `i` corresponds to imaging-job pulse `(azbias + i − 1)`.

### **RAW echoes, imaging XML, and focused products**

Brief distinctions used throughout this release:

*   **RAW `.dat`:** Pre-focus echo recording. For Sandstone Ku strips in this package, each pulse is a fixed **200 128-byte** frame (`128` byte header + `50000 × 4` byte dual-channel `int16` payload at `Fsr = 100 MHz`, `Tr = 500 μs`). Details: [Origin.md](./Origin.md).
*   **`RAW_*.xml`:** Job file for the ground imager. `N_RgStart` / `N_RgLen` (or City-style `RangeStartPoint` / `RangePoints`) are the **range window for focusing**; XML `pulse_num` / `AzimuthPoints` is the azimuth extent of that job—not the focused-image `pulse_num` in per-pol AUX.
*   **Per-pol AUX / SLC / IMG:** Outputs after focusing and polarization separation. AUX `pulse_num` × `pulse_len` is the focused image size; AUX `Fsr` / `PRF` are the values shipped with that product.
*   **`detdelay` / `GPSLeap`:** GPS–UTC leap seconds in the XML (value 18 in the released Sandstone/City job files), not a range-sample delay.
*   **MoCo / PGA flags in XML:** Configuration of the imaging job. Released ATRNet-STAR `Result` products are the ordinary focused exports described above.
*   **Phase:** Each released SLC carries the focused complex phase of its imaging chain. This package does not define an additional cross-strip absolute phase reference.
*   **Full focus processor:** Not open-sourced. Public scripts support parameter/POS/RAW-frame reading and related checks.

### **Summary of SAR Image Data Types**

The core SAR image data from the processing chain consists of:
1.  **Complex Images (SLC):** Primary full-resolution complex data.
2.  **Quantified Images (TIF):** Visualizable intensity images (slant-range or ground-range).
3.  **Auxiliary Files (AUX):** Supporting parameter and metadata files.

<figure>
<div align="center">
<img src=../Figure/framework.png width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">Radar Data Processing Workflow. </div>
</div>
</figure>

This flowchart outlines the end-to-end path from GPS/IMU and radar echoes through motion-compensated image formation to SLC, geometric products, polarimetric products, optional autofocus, and quantization. Auxiliary parameter files (AUX/XML) document the chain. ATRNet-STAR distributes the focused products and metadata described in the subset Origin pages; PGA-tagged filenames, when present in an original delivery, are separate from the ordinary `SLC_*` files in this release.
