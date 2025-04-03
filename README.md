<h1 align="center"> ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild </h1> 

<h5 align="center"><em> Yongxiang Liu, Weijie Li, Li Liu, Jie Zhou, Bowen Peng, Yafei Song, Xuying Xiong, Wei Yang, Tianpeng Liu, Zhen Liu, Xiang Li </em></h5>

<p align="center">
    <a href="#Introduction">Introduction</a> |
    <a href="#User Manual">User Manual</a> |
    <a href="#Motivation">Motivation</a> |
    <a href="#Data Acquisition">Data Acquisition</a> |
    <a href="#Statistical Analysis">Statistical Analysis</a> |
    <a href="#Dataset Value">Dataset Value</a> 
</p >
<p align="center">
    <a href="#ATRBench">ATRBench</a> |
    <a href="#Updates">Updates</a> |
    <a href="#Statement">Statement</a>
</p >
<p align="center">
    <a href="https://arxiv.org/abs/2501.13354"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>
    <a href="ATRNet_STAR_中文.pdf"><img src="https://img.shields.io/badge/论文-中文-800080"></a>
    <a href="https://huggingface.co/datasets/waterdisappear/ATRNet-STAR"><img src="https://img.shields.io/badge/Download-huggface-F5C935"></a>
    <a href="https://www.wjx.top/vm/YOHgMtK.aspx"><img src="https://img.shields.io/badge/下载-百度云-blue"></a>
</p>



This is the official repository for the dataset “ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild”. 
<figure>
<div align="center">
<img src=Figure/FigureBeforeAbstract_00.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">Our ATRNet-STAR dataset contains 40 distinct target types, collected with the aim of replacing the outdated though widely used MSTAR dataset and making a significant contribution to the advancement of SAR ATR research.</div>
</div>
</figure>

## Introduction

As the first step in establishing a large-scale SAR target database, the ATRNet-STAR dataset achieves significant breakthroughs compared to the previous vehicle target benchmark MSTAR. Our team spent nearly two years completing scheme design, data acquisition/processing, and benchmarks. ATRNet-STAR comprises nearly 200,000 target images with comprehensive annotations. The dataset features:

- 4 classes, 21 subclasses, and 40 target types of civilian vehicles (e.g., cars, SUVs, pickup, trucks, buses, tankers)
- Diverse scenarios covering urban, industry, woodlands, bare soil, and sandstone
- Multiple imaging conditions with varying angles, bands, and polarization modes
- Dual data formats, including floating-point complex raw data and processed 8-bit amplitude data

It represents the largest publicly available SAR vehicle recognition dataset, 10 times larger than its predecessor, the famous MSTAR. The substantial sample size supports comprehensive research in generation, detection, and classification tasks.

To facilitate methodological innovation and comparative studies, our team developed ATRBench - a rigorously designed benchmark containing:

- 7 experimental settings for robust recognition, few-shot learning, and transfer learning
- 15 representative methods covering state-of-the-art approaches

Experimental results demonstrate that SAR Automatic Target Recognition (ATR) under complex conditions remains highly challenging. Notably, large-scale pretrained models exhibit relatively superior performance, suggesting that large pretraining could enhance cross-target recognition capabilities. This comprehensive dataset and experimental benchmark establish a new research platform for SAR ATR.

## User Manual

The specific acquisition conditions of the ATRNet-STAR dataset are detailed in the table below, and additional data and model weights will be progressively released based on community needs. 

<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">Table 1: SOC and EOC settings of ATRBench derived from ATRNet-STAR. The imaging conditions in SOC are similar, while EOC considers variations in a single imaging condition. Simple scenes are sandstone and bare soil, and complex scenes are urban, factory, and woodland. The separate labeling of the ground distance and slant distance images results in their numbers of annotations not being strictly corresponding. \# Types.: Number of object types. 
Dep.: Depression angle. Azi.: Target azimuth angle. Pol.: Polarization. # Img. (Ground): Number of images in ground range. # Img. (Slant): Number of images in slant range.</div>


| Scene     | *\# Types* | Platform | Mode    | ***Band*** | ***Res. (m)*** | ***Pol.*** | Dep. (°)       | Azi. (°) | *Img. Size* | *\# Img.* |
| --------- | ---------- | -------- | ------- | ---------- | -------------- | ---------- | -------------- | -------- | ----------- | --------- |
| City      | 40         | airborne | strimap | X          | 0.12~0.15      | quad       | 15, 30, 45, 60 | 5        | 128         | 83,465    |
| Factory   | 40         | airborne | strimap | X/Ku       | 0.12~0.15      | quad       | 15, 30, 45, 60 | 30       | 128         | 63,597    |
| Sandstone | 40         | airborne | strimap | X/Ku       | 0.12~0.15      | quad       | 15, 30, 45, 60 | 30       | 128         | 30,720    |
| Woodland  | 11         | airborne | strimap | X/Ku       | 0.12~0.15      | quad       | 15, 30, 45, 60 | 30       | 128         | 8,094     |
| Bare soil | 11         | airborne | strimap | X/Ku       | 0.12~0.15      | quad       | 15, 30, 45, 60 | 30       | 128         | 8,448     |

The current open-source version comprises two data formats:

- **Ground-range 8-bit amplitude data** (processed for direct analysis)
- **Complex slant-range float data** (raw phase-preserving SAR signals)

Each compressed package is organized according to the 7 experimental configurations in robust recognition of ATRBench, with pre-split training and test sets. The hierarchical structure follows:  

- Experimental configuration (e.g., SOC-40) → Dataset split (train/test) → Target class (e.g., Buick_Excelle_GT)  → Image files  (naming convention: *band_polarization_depression angle_azimuth angle_ID*)  

The combined SOC-40 training and test sets constitute the full open-source data in this release. We provide **pre-converted** **COCO annotations** and **COCO-format conversion codes** to support object detection research. 

<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">Table 2: XML annotations.</div>

| Annotation             |                                         | demo                       |
| ---------------------- | --------------------------------------- | -------------------------- |
| filename               | filename of data                        | KU_HH_15_0_253993.tif      |
| height                 | image height                            | 128                        |
| width                  | image width                             | 128                        |
| depth                  | image depth                             | 1                          |
| format                 | image format                            | ground_range_uint8         |
| range_dimension        | range dimension of SAR image            | 1_height                   |
| cross_range_dimension  | cross range dimension of SAR image      | 2_width                    |
| target_id              | id of target in 40 objects              | 13                         |
| class                  | target class                            | Car                        |
| subclass               | target subclass                         | Medium_SUV                 |
| type                   | target type                             | Changfeng_Cheetah_CFA6473C |
| length                 | target size                             | 4.8m                       |
| width                  |                                         | 1.79m                      |
| height                 |                                         | 1.88m                      |
| xmin                   | target location                         | 25                         |
| xmax                   |                                         | 65                         |
| ymin                   |                                         | 26                         |
| ymax                   |                                         | 69                         |
| scene_name             | scene name                              | sandstone                  |
| platform               | platform                                | airborne                   |
| imaging_mode           | imaging mode                            | strimap                    |
| band                   | band                                    | KU                         |
| polarization           | polarization                            | HH                         |
| range_resolution       | range resolution                        | 0.15m                      |
| cross_range_resolution | cross range resolution                  | 0.15m                      |
| depression_angle       | depression angle                        | 15°                        |
| target_azimuth_angle   | azimuth angle between target and sensor | 0°                         |

## Motivation

Synthetic Aperture Radar (SAR) imaging is capable of generating high-resolution imagery irrespective of lighting conditions and weather and has become an indispensable tool for Earth observation. SAR Remote Sensing (RS) data enables analysis and recognition of objects and scenes, which has become a valuable complement to other RS imagery. Consequently, as a fundamental and challenging field in RS image analysis, SAR Automatic Target Recognition (ATR), which autonomously detects and classifies objects of interest (e.g., vehicles, ships, aircraft, and buildings), has become an active research area for several decades. SAR ATR has a wide range of civilian and military applications, including global surveillance, military reconnaissance, urban management, disaster assessment, and emergency rescue. Despite the remarkable achievements over the past several decades in the field of SAR ATR, the accurate, robust, and efficient recognition of any target in an open world remains unresolved.

<figure>
<div align="left">
<img src=Figure/fig1_motivation.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">Motivation of our ATRNet-STAR. Subfigure (a) depicts the most frequent keywords in 21,780 journal papers published in remote sensing (TGRS, JSTARS, GRSL, ISPRS Journal, and JAG) from 2020 to 2024. The size of each word is proportional to its frequency, highlighting that concepts such as Synthetic Aperture Radar (SAR), image classification, and object detection have garnered substantial attention.
Subfigure (b) focuses on the number of publications related to SAR Automatic Target Recognition (ATR) over the past five years, a cross-area of the concepts highlighted in Subfigure (a). As the pioneering dataset for SAR target classification, MSTAR has long served as the predominant benchmark due to its unique data diversity and accumulated benchmarks. However, the lack of large-scale datasets has significantly limited the growth of this research field in recent years.</div>
</div>
</figure>

**Need for ATRNet -** The advent of big data has propelled the evolution of RS pre-training foundation models where large-scale pre-training enables efficient cross-task adaptation with minimal finetuning. However, the scarcity of large-scale standardized datasets restricts the development of generalizable data-driven SAR ATR methods compared to the success of foundation models in other RS sensors. The data sensitivity, acquisition costs, annotation difficulty, and complexity of SAR imaging hinder the establishment of open large-scale data ecosystems: More than 50% of SAR ATR studies still rely on the 1990s-era Moving and Stationary Target Acquisition and Recognition (MSTAR) dataset. In addition, non-standardized evaluation protocols across existing benchmarks impede objective algorithm comparison. Therefore, building a large SAR ATR dataset and benchmark is necessary to unlock new model capabilities in this field. We aim for **ATRNet**, a massive, diverse, and standard SAR target dataset benchmark for model and recognize target characteristics. 

<figure>
<div align="left">
<img src=Figure/fig3_Timeline.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;">Timeline of SAR classification dataset. Compared to other datasets focusing on target slices in simple scenes, we provide diverse target samples from different scenes to study the more difficult issues, and it is a much larger SAR ATR dataset.</div>
</div>
</figure>

**Need for ATRNet-STAR -** Researchers have struggled to construct many SAR target datasets for ATR tasks. Many SAR classification datasets have significantly improved data diversity. In particular, new SAR detection datasets (SARDet-100K, FAIR-CSAR) have emerged with 100,000 images. However, our previous research on SAR foundation models revealed that collecting public datasets yields fewer than 200,000 available target samples due to severe sample imbalance\textemdash, mainly ship detection datasets. As the inaugural phase of ATRNet, we focus on vehicle targets due to:

1. Recent datasets predominantly are targets based on spaceborne SAR with constrained imaging geometries, whereas airborne platforms offer higher resolution (0.1-0.3m) and flexible imaging conditions in complex scenes.
2. Vehicle recognition has driven SAR ATR research for three decades since MSTAR in the 1990s, establishing mature research systems.
3. The 1990s-era MSTAR dataset, despite its seminal role, suffers from idealized acquisition conditions and saturated performance (near 99% accuracy), failing to reflect real-world complexities and support innovation in the 2020s.

## Data Acquisition

Here, we present our data acquisition pipeline in terms of imaging, annotation, and product.

<figure>
<div align="left">
<img src=Figure/fig_acquisition_00.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;"> Illustrations of data acquisition.
We annotate and cut to build target slices with corresponding metadata information. The range dimension of slant range complex data is in the line of sight direction, which results in the deformation of the target shape in this dimension. Therefore, we also provide ground range images after ground projecting based on slant range amplitude data.</div>
</div>
</figure>

**Imaging -** We use the Unmanned Aerial Vehicle (UAV) platform to carry sensor equipment and collect data. Two antennas acquire quad polarization radar echo in the X and Ku bands. Besides, the Position and Orientation System (POS) device provides the Global Positioning System (GPS) and Inertial Measurement Unit (IMU) information for motion compensation. After imaging processing, we have slant range complex images. Ground range data is obtained by ground projection based on slant range amplitude images and POS information.

**Annotation -** Target classes and coordinates are annotated using rectangular box labels based on optical reference images and deployment records. Because of focusing on individual target signatures, all objects maintain specified separation distances during placement. Besides our vehicles, other vehicles in the scene are labeled as ``other''. After labeling, we acquired the target slices and added a random offset. 

**Product -** We offer data products in two coordinate systems. The distance dimension of the slant range data is the line of sight direction, and we provide original complex data. The ground range images is projected to the ground truth distance and is processed with nonlinear quantization. The corresponding annotation files include basic image information and target, scene, and sensor parameters.

## Statistical Analysis

We present key characteristics and comparative statistics of our dataset relative to others.

<figure>
<div align="left">
<img src=Figure/fig_target_category_00.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;"> Taxonomic systems.
For civilian vehicles, the taxonomy is based on Chinese and European vehicle classification standard, according to the vehicle's purpose, structure, size, and mass. For military vehicles, we followed the MSTAR taxonomic system. 
(a) Taxonomic systems of ATRNet-STAR.
Our dataset is a comprehensive civilian vehicle map covering 4 classes, 21 subclasses, and 40 types. We provide a detailed illustration of the histogram distribution for these 40 vehicle types, demonstrating the breadth and depth of our dataset. Its sufficient and balanced samples make it well-equipped to meet various experimental settings and studies. 
(b) Target classes of SAR vehicle datasets.
We statistically analyze the number of civilian and military vehicle classes and types in SAR vehicle datasets. Our dataset greatly enhances the civilian vehicle richness for the SAR ATR, compared to the measured dataset Gotcha and the simulated datasets CVDomes and SARSim.
(c) List of vehicle abbreviations.</div>
</div>
</figure>

**Class distribution -** ATRNet-STAR dataset includes 4 classes, 21 subclasses and 40 types of vehicles with balanced samples. Relative to other SAR target datasets, our dataset's enhanced diversity introduces novel formidable challenges to the SAR target fine-grained recognition research. Moreover, adequate samples guarantee the availability of a substantial array of target classes across experimental settings, facilitating more diverse investigations.

<figure>
<div align="left">
<img src=Figure/fig7_target_info_00.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;"> Demo images, classes, types, and sizes of 40 vehicles in ATRNet-STAR.
Here are the SAR and corresponding RGB images. As we can see, the obvious difference between the vehicles in the SAR images is the scattering characteristic variation due to size and structure. Therefore, we classify various vehicle types based on their size and structure. Their size (length * width * height) measurements are listed in meters (m). Besides, it is worth noting that our SAR images are not in the ideal situation where the target is located right in the center of the demo image, just as in the MSTAR dataset, but has a random offset similar to the QinetiQ dataset.</div>
</div>
</figure>

**Object size -** The length, width, and height of our 40 objects are listed in Figure, and we can see that they have various structures, different sizes, and similar size ratios. Compared to another civilian vehicle dataset (Gotcha), our dataset has a wider range of target classes and size distribution.

**Reference target -** Besides deploying corner reflectors for resolution measurements, we produce and place a reference target of multiple geometries as the ``SLICY'' target in the MSTAR dataset for scattering research.

**Non-centered location -** Most SAR target datasets place the target in the image center or contain only the target region. Since remote sensing differs from the human eye view that customarily centers the object of interest, the overhead view requires searching targets with more interference. We randomly added position offsets to increase the recognition difficulty with detecting non-centered target locations. 

<figure>
<div align="center">
<img src=Figure/fig_scene_00.jpg width="55%">
<div align="left">
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;"> Influence of different scenes on target characteristics in SAR images.
Most current SAR target classification datasets are collected in simple scenes. (a) MSTAR is collected in a flat grass scene with a clean background and pseudo-correlation between target and background. While (b) our ATRNet-STAR samples are collected under different positions in various scenes. In Fig. (b), it is clear that the same target under the same imaging angle has obvious signature changes due to different scenes. For example, other objects higher up in front of and behind the target can create occlusions and layovers, reducing reflected energy and increasing non-target scattering. Besides, the target shadow in difficult scenes is not as obvious as the MSTAR data. (c) Occlusion and layover.
Compared to the bare soil scene without interfering objects, the factory and woodland scenarios indicate occlusion and layover. Shadows from interfering objects in the former may obscure targets, while trees in the latter are likely to increase non-target scattering. We illustrate this problem with a single target demo at the same angle across scenes, but occlusion and layover result from a combination of target, interference, and imaging geometry. These statistics are from an SUV Chang'an CSCS75 Plus (Vehicle 12) at different image angles in the 3 scenes's ground range images.</div>
</div>
</figure>

**Occlusion in scenes -** Whereas previous vehicle datasets collect target samples in ideal environments, we systematically acquire samples across diverse scenes. Figure shows their different influences with occlusions and layovers in different scene regions. The shadow of buildings and roadside trees in the factory may obscure targets and reduce target scattering, and the nearby trees in woodland have more obvious occlusion and increase non-target scattering.

<figure>
<div align="center">
<img src=Figure/target_azimuth_00.jpg width="55%">
<div align="left">
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;"> Target azimuth angle distribution of (a) MSTAR and (b) ATRNet-STAR. 
Compared to the MSTAR dataset, which has depression angles mainly publicized at 17° and 15° with incomplete angles for most target classes, our ATRNet-STAR dataset provides balance and comprehensive angles for all targets. However, the sampling interval of target angles is sparser due to strip imaging. </div>
</div>
</figure>

**Imaging angle -** Different imaging angles affect the target scattering characteristics, and the imaging geometry relationship can change the interference in the scene. Our dataset has various imaging angles with balance distribution in different depression angles across scenes. However, azimuth angle sampling density exhibits scene-dependent due to stripmap mode constraints and operational cost limitations.

<figure>
<div align="center">
<img src=Figure/fig_band_00.jpg width="55%">
<div align="left">
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;"> Ratio of strong and weak points in different bands and polarization.
It can be noticed that the band and polarization can have a statistically significant effect on the target signatures. These statistical data are the target region inside the vehicle's rectangular box of sandstone scenes. Points with pixel values greater than or equal to 128 are treated as strong scattering points for 8-bit ground range images.</div>
</div>
</figure>

**Band and polarization -** Most vehicle datasets with the MSTAR dataset's X-band and HH polarization settings. However, Figure demonstrates that this setting is not the only one to consider, as band and polarization have a noticeable effect on target scattering. We provide two bands and quad polarization to support discussing these sensor parameters.

**Complex data -** Feature extraction based on complex data is a hot research concern, so we provide complex data in slant distance coordinates stored in *.mat* format.

**Number of image bits -** RGB images are stored in 8-bit format, whereas SAR images have a larger range of digital data values. Therefore, we provide different bit formats to investigate quantification for enhancing weaker target scattering points.

**Cross-platform problem -** We collect a small amount of satellite data synchronously under the same scenes and targets to investigate the difficulty across platforms for future versions.

**Quality control -** To ensure high-quality data, we considered challenges such as target and site hire, airspace applications, weather conditions, target placing, and data checking. Eight people completed the data acquisition over six months. In addition, 14 labelers and two inspectors performed the data annotation for about four months.  The entire project cost nearly two years.

## Dataset Value

Based on the diversity of target classes, scenes, imaging conditions, and data format with detailed annotation, we recommend that this dataset be used for SAR studies in the following areas.

**Robustness recognition -** Our dataset encompasses diverse acquisition scenarios, with each image accompanied by exhaustive acquisition metadata. Therefore, we can discuss in detail the robustness of the SAR recognition algorithms when the acquisition conditions vary between the training and test sets. If we consider data from different acquisition conditions as domains, the training set can include multiple domains to learn domain invariant features, the domain generalization. 

**Few-shot learning -** Besides acquisition condition variation, the inherent challenge of limited training samples due to high SAR collection costs also presents critical research opportunities. The few-shot setting can be combined with robust recognition, and our dataset's large number of target types can enrich the task construction of meta-learning settings in existing few-shot learning.

**Transfer learning -** Our dataset contains a large diversity of samples that can be used to study the transfer problem of the pre-training model with different SAR target datasets. In addition, the large number of samples can further increase the volume of SAR target samples to advance the study of self-supervised learning and foundation models.

**Incremental learning -** The diversity of our dataset supports investigations into domain incremental learning and class incremental learning. The domain incremental learning can improve the algorithm's robustness with a dynamic process, and class incremental learning can progressively increase the ability to recognize or reject new classes. These incremental capacities are essential for SAR ATR in open environments.

**Physical deep learning -** SAR images have unique properties, such as complex phase and polarization. Our dataset provides multi-format data to facilitate recognition studies leveraging these intrinsic attributes instead of only relying on quantized SAR magnitude images. In addition, detailed metadata can also provide more gains for SAR ATR.

**Generative Models - **Beyond recognition tasks, our dataset enables controllable generation of target samples under diverse imaging conditions, as well as estimation of target parameters across varying acquisition scenarios.

We encourage researchers to propose new experimental settings and research issues based on this dataset. \emph{Don't hesitate to contact us if you get new ideas.

## ATRBench

We consider 7 experimental settings with 2 data formats as classification and detection benchmarks from this dataset, named **ATRBench**. These experimental settings include 2 SOC settings sampled from similar distributions and 5 EOC settings with obvious distribution shifts. The data formats consist of magnitude images in the ground range coordinate system and complex images in the slant range coordinate system. 

Please refer to <a href="ATRBench/README.md">readme</a>. 

## Updates

- [ ] Release of aligned quad polarizable data (20%). 
- [ ] Constructing rotated box detection and multi-resolution data (5%).
- [ ] Collecting more target samples from various classes (10%).

## Statement
- If you have any questions, please contact us at lwj2150508321@sina.com. 

- If you find our work is useful, please give us 🌟 in GitHub and cite our paper in the following BibTex format:

```
@misc{liu2025atrnet,
	title={{ATRNet-STAR}: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild}, 
	author={Yongxiang Liu and Weijie Li and Li Liu and Jie Zhou and Bowen Peng and Yafei Song and Xuying Xiong and Wei Yang and Tianpeng Liu and Zhen Liu and Xiang Li},
	year={2025},
	eprint={2501.13354},
	archivePrefix={arXiv},
	primaryClass={cs.CV},
	url={https://arxiv.org/abs/2501.13354}, 
}
