<h1 align="center"> 🚀 ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild </h1> 

<h5 align="center"><em> Yongxiang Liu, Weijie Li, Li Liu, Jie Zhou, Bowen Peng, Yafei Song, Xuying Xiong, Wei Yang, Tianpeng Liu, Zhen Liu, Xiang Li </em></h5>

<p align="center">
    <a href="#Introduction">📖 Introduction</a> |
    <a href="#User Manual">📘 User Manual</a> |
    <a href="#Motivation">💡 Motivation</a> |
    <a href="#Data Acquisition">🛰️ Data Acquisition</a> |
    <a href="#Statistical Analysis">📊 Statistical Analysis</a> |
    <a href="#Dataset Value">🏆 Dataset Value</a> |
    <a href="#ATRBench">⚙️ ATRBench</a> |
    <a href="#Statement">📜 Statement</a>
</p>

<p align="center">
	<a href="https://ieeexplore.ieee.org/document/11367309/authors#authors"><img src="https://img.shields.io/badge/Paper-TPAMI-blue"></a>
    <a href="https://arxiv.org/abs/2501.13354"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>
    <a href="ATRNet_STAR_中文.pdf"><img src="https://img.shields.io/badge/论文-中文-800080"></a>
    <a href="https://huggingface.co/datasets/waterdisappear/ATRNet-STAR"><img src="https://img.shields.io/badge/Download-Hugging Face-F5C935"></a>
    <a href="https://www.wjx.top/vm/YOHgMtK.aspx"><img src="https://img.shields.io/badge/下载-百度云-blue"></a>
	<a href="https://radars.ac.cn/web/data/getData?dataType=GDHuiYan-ATRNet"><img src="https://img.shields.io/badge/下载-雷达学报-red"></a>
    <a href="https://www.scidb.cn/detail?dataSetId=d9ea44937cb94fba9befe9cdb15ffeed"><img src="https://img.shields.io/badge/下载-科学数据银行-purple"></a>	
</p>

This is the official repository for the dataset **“ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild”**. If you find our work useful, please give us a star ⭐ on GitHub and cite our paper (BibTeX format at the end).

<figure>
<div align="center">
<img src=Figure/FigureBeforeAbstract_00.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
Our ATRNet-STAR dataset contains 40 distinct target types, collected to replace the outdated (though widely used) MSTAR dataset and to significantly advance SAR ATR research.
</div>
</div>
</figure>

---

## 📢 Updates (2026-04-05)

### ✅ Released Data

- **Aligned quad‑polarization data (complex float)**  
  Location: `.\Slant_Range\complex_float_quad`  
  *[Demo script](Figure/fig_quad.png) for aligned quad‑polarization complex data is provided.*

- **RGB auxiliary data**  
  Location: `.\Auxiliary_data\RGB.7z`  
  *[Demo script](Figure/fig_RGB.png) for RGB data is provided.*

- **Raw data (original full‑scene SAR images)**  
  Location: `.\Origin`  
  *Includes [un‑chipped SAR scenes](Documentation/Raw_data.md) with corresponding metadata (auxiliary files, raw echoes, and GeoTIFF images).*

### 🚧 In Progress

- **Rotated bounding box detection data** – 35% completed  
- **Collecting more target samples from various classes** – 10% completed

### 📌 Notes for Researchers

To facilitate further communication, researchers are advised to refer to the relevant documentation and discuss specific data needs with us. We will evaluate whether the requested data can be provided without violating confidentiality agreements.  
For information regarding the release of raw data, please refer to the [Raw Data Documentation](Documentation/Raw_data.md) to see what is available.

---

## 📖 Introduction

As the first step in establishing a large‑scale SAR target database, the **ATRNet-STAR** dataset achieves significant breakthroughs compared to the previous vehicle target benchmark **MSTAR**. Our team spent nearly two years completing scheme design, data acquisition/processing, and benchmarks. ATRNet-STAR comprises nearly **200,000** target images with comprehensive annotations. The dataset features:

- **4 classes, 21 subclasses, and 40 target types** of civilian vehicles (e.g., cars, SUVs, pickups, trucks, buses, tankers)
- **Diverse scenarios** covering urban, industrial, woodlands, bare soil, and sandstone
- **Multiple imaging conditions** with varying angles, bands, and polarization modes
- **Dual data formats**, including floating‑point complex raw data and processed 8‑bit amplitude data

It represents the **largest publicly available SAR vehicle recognition dataset**, 10 times larger than its predecessor, the famous MSTAR. The substantial sample size supports comprehensive research in generation, detection, and classification tasks.

To facilitate methodological innovation and comparative studies, our team developed **ATRBench** – a rigorously designed benchmark containing:

- **7 experimental settings** for robust recognition, few‑shot learning, and transfer learning
- **15 representative methods** covering state‑of‑the‑art approaches

Experimental results demonstrate that SAR Automatic Target Recognition (ATR) under complex conditions remains highly challenging. Notably, large‑scale pretrained models exhibit relatively superior performance, suggesting that large pretraining could enhance cross‑target recognition capabilities. This comprehensive dataset and experimental benchmark establish a new research platform for SAR ATR.

---

## 📘 User Manual

The specific acquisition conditions of the ATRNet-STAR dataset are detailed in the table below. Additional data and model weights will be progressively released based on community needs.

<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>Table 1:</strong> Acquisition parameters of open‑source ATRNet-STAR.
We segment the targets with fixed‑size slices and random offsets to investigate their characteristics and aim for robust recognition under different collection conditions. We varied the target location in urban and factory scenes and collected data at the factory several times. In addition, some targets cannot be labeled due to heavy occlusion.<br>
The carrier frequency of the X‑band is 9.6 GHz, and that of the Ku‑band is 14.6 GHz, both with a bandwidth of 1200 MHz. The platform flight speed is 10 m/s, and the pulse repetition frequency is 124.998 Hz. The flight altitude corresponds to different depression angles.<br>
<em># Types:</em> number of object types. <em>Res.:</em> resolution. <em>Pol.:</em> polarization. <em>Dep.:</em> depression angle. <em>Azi.:</em> target azimuth angle interval. <em>Img. Size:</em> image size. <em># Img.:</em> number of images in ground and slant range coordinate systems.
</div>

| Scene     | *# Types* | Platform | Mode    | ***Band*** | ***Res. (m)*** | ***Pol.*** | Dep. (°)       | Altitude (m)       | Azi. (°) | *Img. Size* | *# Img.* |
| --------- | --------- | -------- | ------- | ---------- | -------------- | ---------- | -------------- | ------------------ | -------- | ----------- | --------- |
| City      | 40        | airborne | strimap | X          | 0.12~0.15      | quad       | 15, 30, 45, 60 | 150, 300, 300, 400 | 5        | 128         | 83,465    |
| Factory   | 40        | airborne | strimap | X/Ku       | 0.12~0.15      | quad       | 15, 30, 45, 60 | 120, 300, 320, 400 | 30       | 128         | 63,597    |
| Sandstone | 40        | airborne | strimap | X/Ku       | 0.12~0.15      | quad       | 15, 30, 45, 60 | 120, 300, 300, 300 | 30       | 128         | 30,720    |
| Woodland  | 11        | airborne | strimap | X/Ku       | 0.12~0.15      | quad       | 15, 30, 45, 60 | 120, 300, 300, 300 | 30       | 128         | 8,094     |
| Bare soil | 11        | airborne | strimap | X/Ku       | 0.12~0.15      | quad       | 15, 30, 45, 60 | 120, 300, 300, 300 | 30       | 128         | 8,448     |

The current open‑source version comprises two data formats:

- **Ground‑range 8‑bit amplitude data** (processed for direct analysis)
- **Complex slant‑range float data** (raw phase‑preserving SAR signals)

Each compressed package is organized according to the 7 experimental configurations in robust recognition of ATRBench, with pre‑split training and test sets. The hierarchical structure follows:  

- Experimental configuration (e.g., SOC‑40) → Dataset split (train/test) → Target class (e.g., Buick_Excelle_GT) → Image files (naming convention: *band_polarization_depression angle_azimuth angle_ID*)

The combined SOC‑40 training and test sets constitute the full open‑source data in this release. We provide **pre‑converted COCO annotations** and **COCO‑format conversion codes** to support object detection research.

<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>Table 2:</strong> XML annotations.
</div>

| Annotation             | Value/Description                          | Example                       |
| ---------------------- | ------------------------------------------ | ----------------------------- |
| filename               | filename of data                           | KU_HH_15_0_253993.tif         |
| height                 | image height                               | 128                           |
| width                  | image width                                | 128                           |
| depth                  | image depth                                | 1                             |
| format                 | image format                               | ground_range_uint8            |
| range_dimension        | range dimension of SAR image               | 1_height                      |
| cross_range_dimension  | cross range dimension of SAR image         | 2_width                       |
| target_id              | id of target in 40 objects                 | 13                            |
| class                  | target class                               | Car                           |
| subclass               | target subclass                            | Medium_SUV                    |
| type                   | target type                                | Changfeng_Cheetah_CFA6473C    |
| length                 | target length                              | 4.8m                          |
| width                  | target width                               | 1.79m                         |
| height                 | target height                              | 1.88m                         |
| xmin                   | bounding box left                          | 25                            |
| xmax                   | bounding box right                         | 65                            |
| ymin                   | bounding box top                           | 26                            |
| ymax                   | bounding box bottom                        | 69                            |
| scene_name             | scene name                                 | sandstone                     |
| platform               | platform                                   | airborne                      |
| imaging_mode           | imaging mode                               | strimap                       |
| band                   | band                                       | KU                            |
| polarization           | polarization                               | HH                            |
| range_resolution       | range resolution                           | 0.15m                         |
| cross_range_resolution | cross‑range resolution                     | 0.15m                         |
| depression_angle       | depression angle                           | 15°                           |
| target_azimuth_angle   | azimuth angle between target and sensor    | 0°                            |

---

## 💡 Motivation

Synthetic Aperture Radar (SAR) imaging can generate high‑resolution imagery regardless of lighting conditions and weather, making it an indispensable tool for Earth observation. SAR remote sensing data enables the analysis and recognition of objects and scenes, becoming a valuable complement to other remote sensing imagery. Consequently, as a fundamental and challenging field in remote sensing image analysis, **SAR Automatic Target Recognition (ATR)** – which autonomously detects and classifies objects of interest (e.g., vehicles, ships, aircraft, buildings) – has been an active research area for decades. SAR ATR has a wide range of civilian and military applications, including global surveillance, military reconnaissance, urban management, disaster assessment, and emergency rescue. Despite remarkable achievements over the past decades, the accurate, robust, and efficient recognition of arbitrary targets in an open world remains an unresolved challenge.

<figure>
<div align="left">
<img src=Figure/fig1_motivation.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>Motivation of our ATRNet-STAR.</strong> Subfigure (a) depicts the most frequent keywords in 21,780 journal papers published in remote sensing (TGRS, JSTARS, GRSL, ISPRS Journal, and JAG) from 2020 to 2024. The size of each word is proportional to its frequency, highlighting that concepts such as “Synthetic Aperture Radar (SAR)”, “image classification”, and “object detection” have garnered substantial attention. Subfigure (b) focuses on the number of publications related to SAR ATR over the past five years – a cross‑area of the concepts highlighted in (a). As the pioneering dataset for SAR target classification, MSTAR has long served as the predominant benchmark due to its unique data diversity and accumulated benchmarks. However, the lack of large‑scale datasets has significantly limited the growth of this research field in recent years.
</div>
</div>
</figure>

**Need for ATRNet** – The advent of big data has propelled the evolution of remote sensing pre‑training foundation models, where large‑scale pre‑training enables efficient cross‑task adaptation with minimal fine‑tuning. However, the scarcity of large‑scale standardized datasets restricts the development of generalizable data‑driven SAR ATR methods compared to the success of foundation models in other remote sensing sensors. Data sensitivity, acquisition costs, annotation difficulty, and the complexity of SAR imaging hinder the establishment of open large‑scale data ecosystems: more than 50% of SAR ATR studies still rely on the 1990s‑era **Moving and Stationary Target Acquisition and Recognition (MSTAR)** dataset. In addition, non‑standardized evaluation protocols across existing benchmarks impede objective algorithm comparison. Therefore, building a large SAR ATR dataset and benchmark is necessary to unlock new model capabilities in this field. We aim for **ATRNet** – a massive, diverse, and standard SAR target dataset benchmark for modeling and recognizing target characteristics.

<figure>
<div align="left">
<img src=Figure/fig3_Timeline.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>Timeline of SAR classification datasets.</strong> Compared to other datasets focusing on target slices in simple scenes, we provide diverse target samples from different scenes to study more difficult issues. Our dataset is a much larger SAR ATR dataset.
</div>
</div>
</figure>

**Need for ATRNet-STAR** – Researchers have struggled to construct many SAR target datasets for ATR tasks. Many SAR classification datasets have significantly improved data diversity. In particular, new SAR detection datasets (SARDet‑100K, FAIR‑CSAR) have emerged with 100,000 images. However, our previous research on SAR foundation models revealed that collecting public datasets yields fewer than 200,000 available target samples due to severe sample imbalance, mainly in ship detection datasets. As the inaugural phase of ATRNet, we focus on **vehicle targets** for three reasons:

1. Recent datasets predominantly consist of spaceborne SAR targets with constrained imaging geometries, whereas airborne platforms offer higher resolution (0.1‑0.3 m) and flexible imaging conditions in complex scenes.
2. Vehicle recognition has driven SAR ATR research for three decades since MSTAR in the 1990s, establishing mature research systems.
3. The 1990s‑era MSTAR dataset, despite its seminal role, suffers from idealized acquisition conditions and saturated performance (near 99% accuracy), failing to reflect real‑world complexities and support innovation in the 2020s.

---

## 🛰️ Data Acquisition

Here, we present our data acquisition pipeline in terms of imaging, annotation, and product.

<figure>
<div align="left">
<img src=Figure/fig_acquisition_00.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>Illustrations of data acquisition.</strong> We annotate and cut to build target slices with corresponding metadata. The range dimension of slant‑range complex data is in the line‑of‑sight direction, which results in the deformation of the target shape in this dimension. Therefore, we also provide ground‑range images after ground projection based on slant‑range amplitude data.
</div>
</div>
</figure>

**Imaging** – We use an Unmanned Aerial Vehicle (UAV) platform to carry sensor equipment and collect data. Two antennas acquire quad‑polarization radar echoes in the X and Ku bands. In addition, a Position and Orientation System (POS) device provides Global Positioning System (GPS) and Inertial Measurement Unit (IMU) information for motion compensation. After imaging processing, we obtain slant‑range complex images. Ground‑range data is obtained by ground projection based on slant‑range amplitude images and POS information.

**Annotation** – Target classes and coordinates are annotated using rectangular box labels based on optical reference images and deployment records. Because we focus on individual target signatures, all objects maintain specified separation distances during placement. Besides our vehicles, other vehicles in the scene are labeled as “other”. After labeling, we acquired the target slices and added a random offset.

**Product** – We offer data products in two coordinate systems. The range dimension of the slant‑range data is the line‑of‑sight direction, and we provide original complex data. The ground‑range images are projected to the true ground distance and processed with nonlinear quantization. The corresponding annotation files include basic image information and target, scene, and sensor parameters.

---

## 📊 Statistical Analysis

We present key characteristics and comparative statistics of our dataset relative to others.

<figure>
<div align="left">
<img src=Figure/fig_target_category.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>Taxonomic systems.</strong> For civilian vehicles, the taxonomy is based on Chinese and European vehicle classification standards, according to the vehicle's purpose, structure, size, and mass. For military vehicles, we followed the MSTAR taxonomic system.<br>
(a) Taxonomic systems of ATRNet-STAR. Our dataset is a comprehensive civilian vehicle map covering 4 classes, 21 subclasses, and 40 types. We provide a detailed illustration of the histogram distribution for these 40 vehicle types, demonstrating the breadth and depth of our dataset. Its sufficient and balanced samples make it well‑equipped to meet various experimental settings and studies.<br>
(b) Target classes of SAR vehicle datasets. We statistically analyze the number of civilian and military vehicle classes and types in SAR vehicle datasets. Our dataset greatly enhances the civilian vehicle richness for SAR ATR, compared to the measured dataset Gotcha and the simulated datasets CVDomes and SARSim.<br>
(c) List of vehicle abbreviations.
</div>
</div>
</figure>

**Class distribution** – ATRNet-STAR includes 4 classes, 21 subclasses, and 40 types of vehicles with balanced samples. Relative to other SAR target datasets, our dataset's enhanced diversity introduces novel formidable challenges to fine‑grained SAR target recognition. Moreover, adequate samples guarantee the availability of a substantial array of target classes across experimental settings, facilitating more diverse investigations.

<figure>
<div align="left">
<img src=Figure/fig7_target_info_00.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>Demo images, subclasses, types, and sizes of 40 vehicles in ATRNet-STAR.</strong> Here are the SAR and corresponding objects. As we can see, the obvious difference between vehicles in the SAR images is the scattering characteristic variation due to size and structure. Therefore, we classify various vehicle types based on their size and structure. Their size (length × width × height) measurements are listed in meters (m). It is worth noting that our SAR images are not in the ideal situation where the target is located right in the center of the demo image (as in MSTAR), but have a random offset similar to the QinetiQ dataset.
</div>
</div>
</figure>

**Object size** – The length, width, and height of our 40 objects are listed in the figure, and we can see that they have various structures, different sizes, and similar size ratios. Compared to another civilian vehicle dataset (Gotcha), our dataset has a wider range of target classes and size distribution.

**Reference target** – Besides deploying corner reflectors for resolution measurements, we produced and placed a reference target of multiple geometries (similar to the “SLICY” target in MSTAR) for scattering research.

**Non‑centered location** – Most SAR target datasets place the target in the image center or contain only the target region. Since remote sensing differs from the human eye view that customarily centers the object of interest, the overhead view requires searching for targets with more interference. We randomly added position offsets to increase recognition difficulty by detecting non‑centered target locations.

<figure>
<div align="center">
<img src=Figure/fig_scene_00.jpg width="55%">
<div align="left">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>Influence of different scenes on target characteristics in SAR images.</strong> Most current SAR target classification datasets are collected in simple scenes. (a) MSTAR is collected in a flat grass scene with a clean background and pseudo‑correlation between target and background. (b) Our ATRNet-STAR samples are collected at different positions in various scenes. In (b), it is clear that the same target under the same imaging angle has obvious signature changes due to different scenes. For example, other objects higher up in front of and behind the target can create occlusions and layovers, reducing reflected energy and increasing non‑target scattering. Additionally, the target shadow in difficult scenes is not as obvious as in MSTAR data. (c) Occlusion and layover. Compared to the bare soil scene without interfering objects, the factory and woodland scenarios exhibit occlusion and layover. Shadows from interfering objects in the former may obscure targets, while trees in the latter are likely to increase non‑target scattering. We illustrate this problem with a single target demo at the same angle across scenes, but occlusion and layover result from a combination of target, interference, and imaging geometry. These statistics are from an SUV (Chang'an CSCS75 Plus, Vehicle 12) at different image angles in three scenes' ground‑range images.
</div>
</div>
</figure>

**Occlusion in scenes** – Whereas previous vehicle datasets collect target samples in ideal environments, we systematically acquire samples across diverse scenes. The figure shows their different influences, with occlusions and layovers in different scene regions. The shadow of buildings and roadside trees in the factory may obscure targets and reduce target scattering; nearby trees in woodland cause more obvious occlusion and increase non‑target scattering.

<figure>
<div align="center">
<img src=Figure/target_azimuth_00.jpg width="55%">
<div align="left">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>Target azimuth angle distribution of (a) MSTAR and (b) ATRNet-STAR.</strong> Compared to the MSTAR dataset, which has depression angles mainly publicized at 17° and 15° with incomplete angles for most target classes, our ATRNet-STAR dataset provides balanced and comprehensive angles for all targets. However, the sampling interval of target angles is sparser due to strip imaging.
</div>
</div>
</figure>

**Imaging angle** – Different imaging angles affect target scattering characteristics, and the imaging geometry relationship can change interference in the scene. Our dataset has various imaging angles with balanced distribution across different depression angles and scenes. However, azimuth angle sampling density exhibits scene dependence due to stripmap mode constraints and operational cost limitations.

<figure>
<div align="center">
<img src=Figure/fig_band_00.jpg width="55%">
<div align="left">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>Ratio of strong and weak points in different bands and polarizations.</strong> It can be noticed that the band and polarization can have a statistically significant effect on target signatures. These statistical data are from the target region inside the vehicle's rectangular box in sandstone scenes. Points with pixel values ≥ 128 are treated as strong scattering points for 8‑bit ground‑range images.
</div>
</div>
</figure>

**Band and polarization** – Most vehicle datasets follow MSTAR's X‑band and HH polarization settings. However, the figure demonstrates that this setting is not the only one to consider, as band and polarization have a noticeable effect on target scattering. We provide two bands and quad‑polarization to support discussion of these sensor parameters.

**Complex data** – Feature extraction based on complex data is a hot research topic, so we provide complex data in slant‑range coordinates stored in `*.mat` format.

**Number of image bits** – RGB images are stored in 8‑bit format, whereas SAR images have a larger range of digital data values. Therefore, we provide different bit formats to investigate quantization for enhancing weaker target scattering points.

**Cross‑platform problem** – We collect a small amount of satellite data synchronously under the same scenes and targets to investigate the difficulty across platforms for future versions.

**Quality control** – To ensure high‑quality data, we considered challenges such as target and site hire, airspace applications, weather conditions, target placement, and data checking. Eight people completed the data acquisition over six months. In addition, 14 labelers and two inspectors performed data annotation for about four months. The entire project took nearly two years.

---

## 🏆 Dataset Value

Based on the diversity of target classes, scenes, imaging conditions, and data formats with detailed annotations, we recommend that this dataset be used for SAR studies in the following areas.

- **Robustness recognition** – Our dataset encompasses diverse acquisition scenarios, with each image accompanied by exhaustive acquisition metadata. Therefore, we can discuss in detail the robustness of SAR recognition algorithms when acquisition conditions vary between training and test sets. If we consider data from different acquisition conditions as domains, the training set can include multiple domains to learn domain‑invariant features – domain generalization.

- **Few‑shot learning** – Besides acquisition condition variation, the inherent challenge of limited training samples due to high SAR collection costs also presents critical research opportunities. The few‑shot setting can be combined with robust recognition, and our dataset's large number of target types can enrich the task construction of meta‑learning settings in existing few‑shot learning.

- **Transfer learning** – Our dataset contains a large diversity of samples that can be used to study the transfer problem of pre‑trained models across different SAR target datasets. In addition, the large number of samples can further increase the volume of SAR target samples to advance the study of self‑supervised learning and foundation models.

- **Incremental learning** – The diversity of our dataset supports investigations into domain incremental learning and class incremental learning. Domain incremental learning can improve algorithm robustness in a dynamic process, and class incremental learning can progressively increase the ability to recognize or reject new classes. These incremental capacities are essential for SAR ATR in open environments.

- **Physical deep learning** – SAR images have unique properties, such as complex phase and polarization. Our dataset provides multi‑format data to facilitate recognition studies leveraging these intrinsic attributes instead of only relying on quantized SAR magnitude images. In addition, detailed metadata can provide more gains for SAR ATR.

- **Generative models** – Beyond recognition tasks, our dataset enables controllable generation of target samples under diverse imaging conditions, as well as estimation of target parameters across varying acquisition scenarios.

We encourage researchers to propose new experimental settings and research issues based on this dataset. Please do not hesitate to contact us if you have new ideas.

---

## ⚙️ ATRBench

We consider **7 experimental settings** with **2 data formats** as classification and detection benchmarks from this dataset, named **ATRBench**. These experimental settings include 2 SOC settings sampled from similar distributions and 5 EOC settings with obvious distribution shifts. The data formats consist of magnitude images in the ground‑range coordinate system and complex images in the slant‑range coordinate system.

Please refer to the <a href="ATRBench/README.md">ATRBench README</a> for details.

---

## 📜 Statement

- If you have any questions, please contact us at: **lwj2150508321@sina.com**
- If you find our work useful, please give us a star ⭐ on GitHub and cite our paper using the following BibTeX entries:

```bibtex
@ARTICLE{liu2026atrnet,
  author={Liu, Yongxiang and Li, Weijie and Liu, Li and Zhou, Jie and Peng, Bowen and Song, Yafei and Xiong, Xuying and Yang, Wei and Liu, Tianpeng and Liu, Zhen and Li, Xiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={{ATRNet-STAR}: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild}, 
  year={2026},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2026.3658649}
}

@ARTICLE{li2025saratr,
  author={Li, Weijie and Yang, Wei and Hou, Yuenan and Liu, Li and Liu, Yongxiang and Li, Xiang},
  journal={IEEE Transactions on Image Processing}, 
  title={SARATR-X: Toward Building a Foundation Model for SAR Target Recognition}, 
  year={2025},
  volume={34},
  number={},
  pages={869-884},
  doi={10.1109/TIP.2025.3531988}
}

@ARTICLE{li2024predicting,
  title = {Predicting gradient is better: Exploring self-supervised learning for SAR ATR with a joint-embedding predictive architecture},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {218},
  pages = {326-338},
  year = {2024},
  issn = {0924-2716},
  doi = {https://doi.org/10.1016/j.isprsjprs.2024.09.013},
  author = {Li, Weijie and Yang, Wei and Liu, Tianpeng and Hou, Yuenan and Li, Yuxuan and Liu, Zhen and Liu, Yongxiang and Liu, Li}
}
