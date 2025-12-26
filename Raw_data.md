### **Radar Data Product Folder Structure**

The radar output is organized into the following main directories:

<figure>
<div align="center">
<img src=Figure/framework.png width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">Radar Data Processing Workflow. This flowchart illustrates the standardized end-to-end workflow for synthetic aperture radar (SAR) data processing, from raw measurements to final analysis-ready products. The process begins with the ingestion and fusion of raw GPS/IMU and radar echo data. Core image formation, aided by motion-compensated trajectory data, produces the foundational Single Look Complex (SLC) image in slant-range geometry. This SLC serves as the central input for several independent, parallel processing modules: the Geometric Module performs terrain correction, resampling, mosaicking, and geocoding to produce georeferenced ground-range maps; the Polarimetry Module generates Pauli-decomposed false-color SLCs for scattering analysis; and the Autofocus (PGA) Module enhances image focus. A Quantization Module can be applied to various image types (SLCs, ground-range images) to generate 8-bit or 16-bit TIFF visualizations. Auxiliary parameter files (AUX/XML) are generated throughout to document the processing chain. </div>
</div>
</figure>

*   **Assistfile:** Contains auxiliary data, including POS data for positioning and attitude determination.
*   **Rawfile:** Stores the raw radar echo data.
*   **Result:** Holds the final imaging results.
*   **RTI:** Contains real-time imaging results generated onboard the aircraft.

### **Provided MATLAB Utility Scripts**

Several `.m` files are provided for data analysis and processing:
*   **Data Reading:** `ParaRead.m` (reads imaging parameters/trajectory), `PosRead.m` (reads POS attitude results), `RawDataRead.m` (checks raw echo intensity).
*   **Target Analysis:** `read_data_hxy.m`, `point_analysis.m` (analyze point target sidelobe ratios).
*   **Image Calibration & QC:** `RgCal_PhaErrEst.m` (extracts range compensation function), `RgCal.m` (applies range compensation), `RgCal_Test.m` (views compensation results), `SNR.m` (calculates signal-to-noise ratio).

### **Contents of the `Result` Folder**

This folder contains the output from the SAR image formation process. Key files include:

*   **SLC Files (`*.slc`):** Single Look Complex images (slant-range).
    *   **Naming:** `SLC_XSAR_[Pol]_*.slc` (e.g., `SLC_XSAR_H1H1_*.slc` for HH polarization).
    *   **Data Type:** 32-bit floating point complex (`complex32`).
    *   **Storage:** Data is stored row-by-row: from (1,1), (1,2)...(1,N) to (M,1), (M,2)...(M,N).
    *   **Reading:** Use the provided `Copy_of_readslc.m` script.
*   **GeoTIFF Images (`*.tif`):** Quantified images.
    *   **Slant-Range Image (`IMG_*.tif`):** The formed image in slant-range geometry. The long side is the range direction, the short side is the azimuth direction.
    *   **Ground-Range Image (`DOM_*.tif`):** The slant-range image after projection/geocoding to ground-range geometry.
*   **PGA-Processed Files:** `slc` and `tif` files marked "with PGA" have undergone Phase Gradient Autofocus (a self-focusing algorithm).
*   **Paul Decomposition Files:** A set of SLC files (`BLU_*.slc`, `GRE_*.slc`, `RED_*.slc`) and a combined pseudo-color `RGB_*.slc` file.
*   **Parameter & Log Files:**
    *   `AUX_*.dat`: Parameter file.
    *   `AUX_*.log`: Image dimension parameters.
    *   `RAW_*.xml`: Configuration file containing the processing parameters.

### **Key Notes on File Naming Conventions**

*   **Polarization:** Tags like `H1H1`, `HV`, `VV` in filenames indicate the polarization channel (HH, cross-pol, VV). `FULL` indicates a polarimetrically synthesized image.
*   **Strip Identifier:** `STRx` (e.g., `STR5`) in a filename identifies the specific data strip or track.
*   **Azimuth Bias:** `azbias####` indicates the starting pulse number (e.g., 1024).

### **Summary of SAR Image Data Types**

The core SAR image data from the processing chain consists of:
1.  **Complex Images (SLC):** Primary full-resolution complex data.
2.  **Quantified Images (TIF):** Visualizable intensity images (slant-range or ground-range).
3.  **Auxiliary Files (AUX):** Supporting parameter and metadata files.
