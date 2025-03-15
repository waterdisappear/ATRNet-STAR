<h1 align="center"> ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild </h1> 

<h5 align="center"><em> Yongxiang Liu, Weijie Li, Li Liu, Jie Zhou, Bowen Peng, Yafei Song, Xuying Xiong, Wei Yang, Tianpeng Liu, Zhen Liu, Xiang Li </em></h5>

<p align="center">
  <a href="#Introduction">Introduction</a> |
  <a href="#Updates">Updates</a> |
  <a href="#ATRBench">ATRBench</a> |
  <a href="#Statement">Statement</a>
</p >
<p align="center">
<a href="https://arxiv.org/abs/2501.13354"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>
</p>

## Introduction

This is the official repository for the dataset “ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild”.

**Abstract:** 
The absence of publicly available, large-scale, high-quality datasets for Synthetic Aperture Radar Automatic Target Recognition (SAR ATR) has significantly hindered the application of rapidly advancing deep learning techniques, which hold huge potential to unlock new capabilities in this field. This is primarily because collecting large volumes of diverse target samples from SAR images is prohibitively expensive, largely due to privacy concerns, the characteristics of microwave radar imagery perception, and the need for specialized expertise in data annotation. Throughout the history of SAR ATR research, there have been only a number of small datasets, mainly including targets like ships, airplanes, buildings, etc. There is only one vehicle dataset MSTAR collected in the 1990s, which has been a valuable source for SAR ATR. To fill this gap, this paper introduces a large-scale, new dataset named ATRNet-STAR with 40 different vehicle categories collected under various realistic imaging conditions and scenes. It marks a substantial advancement in dataset scale and diversity, comprising over 190,000 well-annotated samples, 10 times larger than its predecessor, the famous MSTAR. Building such a large dataset is a challenging task, and the data collection scheme will be detailed. Secondly, we illustrate the value of ATRNet-STAR via extensively evaluating the performance of 15 representative methods with 7 different experimental settings on challenging classification and detection benchmarks derived from the dataset. Finally, based on our extensive experiments, we identify valuable insights for SAR ATR and discuss potential future research directions in this field. We hope that the scale, diversity, and benchmark of ATRNet-STAR can significantly facilitate the advancement of SAR ATR.

<figure>
<div align="center">
<img src=Figure/FigureBeforeAbstract_00.jpg width="95%">
</div>
</figure>

## Updates

- [ ] 🚀Initializing the homepage. (30%). 
- [ ] 🚀Release of datasets and benchmark codes (35%). 
- [ ] Release of aligned quad polarizable data (20%). 
- [ ] Constructing rotated box detection and multi-resolution data (0%).
- [ ] Collecting more target samples from various classes (10%).

## ATRBench

Here, we describe the experimental settings in Table 1, including 2 SOC settings sampled from similar distributions and 5 EOC settings with obvious condition and distribution shifts.

**SOC and EOC settings -** SOC settings are those where the training and test sets have similar imaging conditions, and we have SOC-40 and SOC-50 settings. EOC settings are those where there is a significant domain shift between the training set and the test set, such as variation in imaging conditions and target state. 

*SOC-40* is created from a random data sample with a 7:3 training-to-test ratio. 

*SOC-50* randomly selects data with a similar amount of MSTAR ten classes, and we combine our dataset with MSTAR. 

*EOC-Scene* uses simple scenes (sandstone and bare soil) as the training set and complex scenes (urban, factory, and woodland) as the test set.

*EOC-Depression* is the training set with a 15° depression angle and the test set with 30°, 45° and 60°. 
*EOC-Azimuth* means that training has limited target azimuth angles within 0°~60°, and the test has other angles. Imaging angle and geometry variation can change target signatures and background clutter. 
*EOC-band* and *EOC-polarization* are designed to test the effect of different bands and change from HH-polarization to other polarizations on recognition, respectively.

Table 1: **SOC and EOC settings of ATRBench derived from ATRNet-STAR.** The imaging conditions in SOC are similar, while EOC considers variations in a single imaging condition. Simple scenes are sandstone and bare soil, and complex scenes are urban, factory, and woodland. The separate labeling of the ground distance and slant distance images results in their numbers of annotations not being strictly corresponding. \# Types.: Number of object types. 
Dep.: Depression angle. Azi.: Target azimuth angle. Pol.: Polarization. # Img. (Ground): Number of images in ground range. # Img. (Slant): Number of images in slant range. 

| Setting          | Set   | \# Types | Scene         | Dep.               | Azi.       | Band   | Pol.      | # Img. (Ground) | # Img. (Slant) |
| ---------------- | ----- | -------- | ------------- | ------------------ | ---------- | ------ | --------- | --------------- | -------------- |
| SOC-40           | train | **40**   | all           | 15, 30, 45, 60     | 0~360      | X/Ku   | quad      | 68,091          | 67,780         |
|                  | Test  | **40**   | all           | 15, 30, 45, 60     | 0~360      | X/Ku   | quad      | 29,284          | 29,169         |
| SOC-50           | train | **50**   | all           | 15, 17, 30, 45, 60 | 0~360      | X      | quad      | 18,071          | 18,071         |
|                  | test  | **50**   | all           | 15, 30, 45, 60     | 0~360      | X      | quad      | 17,603          | 17,613         |
| EOC-Scene        | train | 40       | **simple**    | 15, 30, 45, 60     | 0~360      | X/Ku   | quad      | 19,584          | 19,584         |
|                  | test  | 40       | **difficult** | 15, 30, 45, 60     | 0~360      | X/Ku   | quad      | 77,791          | 77,365         |
| EOC-Depression   | train | 40       | all           | **15**             | 0~360      | X/Ku   | quad      | 24,361          | 22,206         |
|                  | test  | 40       | all           | **30, 45, 60**     | 0~360      | X/Ku   | quad      | 73,014          | 74,743         |
| EOC-Azimuth      | train | 40       | all           | 15, 30, 45, 60     | **0~60**   | X      | quad      | 18,636          | 18,592         |
|                  | test  | 40       | all           | 15, 30, 45, 60     | **60~360** | X      | quad      | 78,739          | 78,357         |
| EOC-Band         | train | 40       | except city   | 15, 30, 45, 60     | 0~360      | **X**  | quad      | 27,711          | 27,653         |
|                  | test  | 40       | except city   | 15, 30, 45, 60     | 0~360      | **Ku** | quad      | 27,763          | 27,732         |
| EOC-Polarization | train | 40       | all           | 15, 30, 45, 60     | 0~360      | X      | **HH**    | 24,361          | 24,246         |
|                  | test  | 40       | all           | 15, 30, 45, 60     | 0~360      | X      | **other** | 73,014          | 72,703         |

### Classsification

For classification based on amplitude data, we have chosen six computer vision methods ([VGG16](https://arxiv.org/abs/1409.1556), [ResNet18](https://arxiv.org/abs/1512.03385), [ResNet34](https://arxiv.org/abs/1512.03385), [ConvNeXt](https://arxiv.org/abs/2201.03545)) and two SAR ATR methods ([HDANet](https://ieeexplore.ieee.org/document/10283916), [SARATR-X](https://ieeexplore.ieee.org/document/10856784)). We provide the codes and hyperparameter settings in the Pytorch environment, and the corresponding weights and result files can be downloaded via the dataset link.

Table 2: **Classification results.** We use overall accuracy (%) as a metric, *i.e.*, the number of correctly classified samples in proportion to the total number of samples. **Bolded** text indicates the best result, while <u>underlined</u> text is the next best result.

| Setting          | VGG16 | HDANet   | ResNet18 | ResNet34 | ConvNeXt    | ViT         | HiViT       | SARATR-X    |
| ---------------- | ----- | -------- | -------- | -------- | ----------- | ----------- | ----------- | ----------- |
| SOC-40           | 88.8  | 89.1     | 90.6     | 91.7     | <u>96.0</u> | 76.4        | 86.8        | **96.4**    |
| SOC-50           | 72.9  | 63.7     | 71.2     | 72.9     | <u>81.6</u> | 59.2        | 68.0        | **85.2**    |
| EOC-Scene        | 21.6  | **33.6** | 16.1     | 18.0     | 16.5        | 12.9        | 15.8        | 19.5        |
| – (city)         | 22.7  | **33.7** | 18.1     | 18.9     | 17.9        | 11.6        | 13.9        | 20.4        |
| – (factory)      | 20.5  | **34.3** | 13.7     | 16.8     | 14.4        | 15.3        | 17.8        | 18.4        |
| – (woodland)     | 6.8   | **27.6** | 4.2      | 5.4      | 4.2         | 1.97        | 3.8         | 2.7         |
| EOC-Depression   | 33.2  | 32.9     | 33.9     | 37.2     | **43.1**    | 30.4        | 31.4        | <u>39.9</u> |
| – (30)           | 52.4  | 54.0     | 52.2     | 55.9     | **63.2**    | 43.7        | 47.3        | <u>58.1</u> |
| – (45)           | 33.5  | 32.7     | 34.2     | 38.3     | **45.1**    | 31.0        | 32.0        | <u>41.2</u> |
| – (60)           | 14.0  | 12.3     | 15.4     | 17.4     | **21.3**    | 16.7        | 15.0        | <u>20.5</u> |
| EOC-Azimuth      | 15.7  | 16.4     | 14.9     | 16.5     | 21.1        | **29.0**    | 22.8        | <u>26.4</u> |
| – (60)           | 20.7  | 19.4     | 18.0     | 22.1     | 26.2        | **34.4**    | 27.2        | <u>28.9</u> |
| – (120)          | 8.2   | 9.5      | 8.3      | 8.5      | 13.5        | **26.4**    | 14.2        | <u>22.8</u> |
| – (180)          | 17.6  | 17.9     | 16.4     | 17.4     | 19.9        | 20.8        | <u>22.2</u> | **23.0**    |
| – (240)          | 7.11  | 7.0      | 7.4      | 7.9      | 10.6        | **20.3**    | 11.1        | <u>12.3</u> |
| – (300)          | 26.0  | 30.4     | 25.5     | 27.5     | 36.6        | <u>44.4</u> | 40.9        | **46.7**    |
| EOC-Band         | 78.8  | 79.5     | 83.1     | 83.8     | <u>88.4</u> | 65.7        | 70.7        | **89.2**    |
| – inverse        | 74.5  | 79.8     | 76.1     | 79.0     | <u>84.6</u> | 62.3        | 78.4        | **89.1**    |
| EOC-Polarization | 72.5  | 63.1     | 71.4     | 70.5     | <u>83.1</u> | 53.6        | 67.1        | **84.6**    |
| – VV             | 72.4  | 63.0     | 72.9     | 72.1     | <u>84.4</u> | 55.2        | 69.1        | **87.5**    |
| – HV             | 72.3  | 62.7     | 70.2     | 69.4     | <u>82.4</u> | 52.4        | 65.9        | **83.1**    |
| – VH             | 72.8  | 63.5     | 71.0     | 70.0     | <u>82.7</u> | 55.1        | 66.5        | **83.3**    |

### Detection



## Statement
- The first version will release after 2025.3.20.
- If you have any questions, please contact us at lwj2150508321@sina.com. 

