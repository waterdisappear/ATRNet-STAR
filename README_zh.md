<p align="right">
  <a href="./README.md">English</a> | <b>中文</b>
</p>

<h1 align="center">ATRNet-STAR：面向开放环境遥感目标识别的大规模数据集与基准</h1>
<figure>
<div align="center">
<img src=Figure/fig_logo.png width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
</div>
</div>
</figure>

<h5 align="center"><em>刘永祥、李玮杰、刘丽、周洁、彭渤文、宋娅菲、熊旭莹、杨威、刘天鹏、刘振、黎湘</em></h5>

<p align="center">
    <a href="#简介">📖 简介</a> |
    <a href="#用户手册">📘 用户手册</a> |
    <a href="#研究动机">💡 研究动机</a> |
    <a href="#数据采集">🛰️ 数据采集</a> |
    <a href="#统计分析">📊 统计分析</a> |
    <a href="#数据集价值">🏆 数据集价值</a> |
    <a href="#ATRBench">⚙️ ATRBench</a> |
    <a href="#声明">📜 声明</a>
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

本仓库为数据集 **「ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild」** 的官方发布页。若对您的研究有帮助，欢迎在 GitHub 上点亮 Star ⭐，并按文末 BibTeX 格式引用相关论文。

<figure>
<div align="center">
<img src=Figure/FigureBeforeAbstract_00.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
ATRNet-STAR 包含 40 类目标，旨在替代沿用已久但仍被广泛使用的 MSTAR 数据集，推动 SAR 自动目标识别（ATR）研究取得实质性进展。
</div>
</div>
</figure>

---

## 📢 更新

### 2026-10-15

- 补充并更正 **Raw_data** 说明：磁盘 RAW 帧长、XML 距离/方位窗与聚焦 AUX 尺寸的区别、`azbias`、`detdelay`/`GPSLeap`、MoCo/PGA 配置标志与已发布 `Result` 产品的关系，以及相位相关说明。详见 [Raw_data.md](Documentation/Raw_data.md) 与 [沙地 Origin](Documentation/Origin_zh.md)。

### 2026-09-26

### ✅ 已发布数据

- **对齐全极化复数数据（complex float）**  
  路径：`.\Slant_Range\complex_float_quad`  
  *提供对齐全极化复数数据的[演示示意](Figure/fig_quad.png)。*

- **RGB 辅助数据**  
  路径：`.\Auxiliary_data\RGB.7z`  
  *提供 RGB 数据的[演示示意](Figure/fig_RGB.png)。*

- **原始全场景 SAR 数据**  
  路径：`.\Raw_data`  
  *提供沙地与城区两类未切片全场景 SAR 影像，含辅助文件、原始回波及 GeoTIFF 成像产品。详见 [沙地](Documentation/Origin_zh.md) / [城区](Documentation/Origin_City_zh.md)。*

### 🚧 进行中

- **旋转框检测数据** — 约完成 35%  
- **补充更多类别目标样本** — 约完成 10%

### 📌 致研究者

为便于进一步交流，建议研究者先查阅相关文档，并就具体数据需求与我们讨论。我们将在不违反保密约定的前提下，评估所请求数据是否可以提供。  
有关原始数据发布范围与说明，请参阅 [Raw Data Documentation](Documentation/Raw_data.md)。

---

## 📖 简介

作为构建大规模 SAR 目标数据库的第一步，**ATRNet-STAR** 相对此前车辆目标基准 **MSTAR** 取得了显著突破。团队历时近两年完成方案设计、数据采集与处理，以及评测基准建设。ATRNet-STAR 共包含近 **20 万** 幅带完整标注的目标图像，主要特点如下：

- **4 大类、21 亚类、40 型**民用车辆（如轿车、SUV、皮卡、货车、客车、罐车等）
- **多样场景**，覆盖城区、工业区、林地、裸土与沙地等
- **多种成像条件**，包含不同角度、波段与极化方式
- **双数据格式**，包括浮点复数原始数据与处理后的 8 比特幅度数据

该数据集是目前公开可获取的**最大规模 SAR 车辆识别数据集**，规模约为著名前身 MSTAR 的 10 倍。充足样本量可支撑生成、检测与分类等方向的系统研究。

为促进方法创新与公平对比，我们构建了严谨设计的评测基准 **ATRBench**，包括：

- **7 种实验设置**，面向稳健识别、少样本学习与迁移学习
- **15 种代表性方法**，覆盖当前主流与先进途径

实验结果表明，复杂条件下的 SAR 自动目标识别（ATR）仍极具挑战。值得注意的是，大规模预训练模型表现相对更优，提示大规模预训练有望增强跨目标识别能力。这一完整数据集与实验基准，为 SAR ATR 研究提供了新的平台。

---

## 📘 用户手册

ATRNet-STAR 数据集的具体采集条件见下表。后续将根据社区需求逐步发布更多数据与模型权重。

<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>表 1：</strong>开源 ATRNet-STAR 的采集参数。<br>
我们以固定尺寸切片并加入随机偏移截取目标，以考察其特性，并面向不同采集条件下的稳健识别。城区与工厂场景中目标位置有所变化，工厂场景亦进行了多次采集。此外，部分目标因严重遮挡无法标注。<br>
X 波段载频为 9.6 GHz，Ku 波段载频为 14.6 GHz，带宽均为 1200 MHz。平台飞行速度为 10 m/s，脉冲重复频率为 124.998 Hz。飞行高度与不同俯视角相对应。<br>
<em># Types：</em>目标类型数。<em>Res.：</em>分辨率。<em>Pol.：</em>极化。<em>Dep.：</em>俯视角。<em>Azi.：</em>目标方位角采样间隔。<em>Img. Size：</em>图像尺寸。<em># Img.：</em>地距与斜距坐标系下的图像数量。
</div>

| Scene     | *# Types* | Platform | Mode    | ***Band*** | ***Res. (m)*** | ***Pol.*** | Dep. (°)       | Altitude (m)       | Azi. (°) | *Img. Size* | *# Img.* |
| --------- | --------- | -------- | ------- | ---------- | -------------- | ---------- | -------------- | ------------------ | -------- | ----------- | --------- |
| City      | 40        | airborne | strimap | X          | 0.12~0.15      | quad       | 15, 30, 45, 60 | 150, 300, 300, 400 | 5        | 128         | 83,465    |
| Factory   | 40        | airborne | strimap | X/Ku       | 0.12~0.15      | quad       | 15, 30, 45, 60 | 120, 300, 320, 400 | 30       | 128         | 63,597    |
| Sandstone | 40        | airborne | strimap | X/Ku       | 0.12~0.15      | quad       | 15, 30, 45, 60 | 120, 300, 300, 300 | 30       | 128         | 30,720    |
| Woodland  | 11        | airborne | strimap | X/Ku       | 0.12~0.15      | quad       | 15, 30, 45, 60 | 120, 300, 300, 300 | 30       | 128         | 8,094     |
| Bare soil | 11        | airborne | strimap | X/Ku       | 0.12~0.15      | quad       | 15, 30, 45, 60 | 120, 300, 300, 300 | 30       | 128         | 8,448     |

当前开源版本包含两种数据格式：

- **地距 8 比特幅度数据**（便于直接分析使用）
- **斜距复数浮点数据**（保留相位的原始 SAR 信号）

各压缩包按 ATRBench 稳健识别中的 7 种实验配置组织，并预先划分训练集与测试集。目录层级为：

- 实验配置（如 SOC‑40）→ 数据划分（train/test）→ 目标类别（如 Buick_Excelle_GT）→ 图像文件（命名约定：*波段_极化_俯视角_方位角_ID*）

本次开源中，SOC‑40 的训练与测试集合并即构成完整开源数据。我们同时提供**预转换的 COCO 标注**及 **COCO 格式转换代码**，以支持目标检测研究。

<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>表 2：</strong>XML 标注字段。
</div>

| Annotation             | Value/Description                          | Example                       |
| ---------------------- | ------------------------------------------ | ----------------------------- |
| filename               | 数据文件名                                 | KU_HH_15_0_253993.tif         |
| height                 | 图像高度                                   | 128                           |
| width                  | 图像宽度                                   | 128                           |
| depth                  | 图像深度                                   | 1                             |
| format                 | 图像格式                                   | ground_range_uint8            |
| range_dimension        | SAR 图像距离维                             | 1_height                      |
| cross_range_dimension  | SAR 图像方位维                             | 2_width                       |
| target_id              | 40 类目标中的编号                          | 13                            |
| class                  | 目标大类                                   | Car                           |
| subclass               | 目标亚类                                   | Medium_SUV                    |
| type                   | 目标型号                                   | Changfeng_Cheetah_CFA6473C    |
| length                 | 目标长度                                   | 4.8m                          |
| width                  | 目标宽度                                   | 1.79m                         |
| height                 | 目标高度                                   | 1.88m                         |
| xmin                   | 边界框左边界                               | 25                            |
| xmax                   | 边界框右边界                               | 65                            |
| ymin                   | 边界框上边界                               | 26                            |
| ymax                   | 边界框下边界                               | 69                            |
| scene_name             | 场景名称                                   | sandstone                     |
| platform               | 平台                                       | airborne                      |
| imaging_mode           | 成像模式                                   | strimap                       |
| band                   | 波段                                       | KU                            |
| polarization           | 极化                                       | HH                            |
| range_resolution       | 距离向分辨率                               | 0.15m                         |
| cross_range_resolution | 方位向分辨率                               | 0.15m                         |
| depression_angle       | 俯视角                                     | 15°                           |
| target_azimuth_angle   | 目标与传感器之间的方位角                   | 0°                            |

---

## 💡 研究动机

合成孔径雷达（SAR）成像可在不分昼夜、不惧天气的条件下获取高分辨率影像，是对地观测不可或缺的手段。SAR 遥感数据可用于目标与场景的分析识别，是对其他遥感影像的重要补充。因此，作为遥感图像分析中既基础又具挑战性的方向，**SAR 自动目标识别（ATR）**——对车辆、舰船、飞机、建筑物等感兴趣目标进行自动检测与分类——数十年来始终是活跃研究领域。SAR ATR 在全球监视、军事侦察、城市管理、灾害评估与应急救援等军民应用中均有广阔空间。尽管过去数十年已取得显著进展，开放世界中任意目标的准确、稳健、高效识别仍是未解难题。

<figure>
<div align="left">
<img src=Figure/fig1_motivation.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>ATRNet-STAR 的研究动机。</strong>子图 (a) 统计 2020–2024 年遥感领域期刊（TGRS、JSTARS、GRSL、ISPRS Journal、JAG）共 21,780 篇论文中的高频关键词，词云大小与词频成正比，可见 “Synthetic Aperture Radar (SAR)”“image classification”“object detection” 等概念受到广泛关注。子图 (b) 聚焦近五年与 SAR ATR 相关的发文量——正是 (a) 中高亮概念的交叉领域。作为 SAR 目标分类的开创性数据集，MSTAR 因其独特的数据多样性与长期积累的评测体系，长期占据主导基准地位；但近年来大规模数据的缺失，显著制约了该研究方向的进一步发展。
</div>
</div>
</figure>

**为何需要 ATRNet** — 大数据时代推动了遥感预训练基础模型的演进：大规模预训练使模型仅需少量微调即可高效适配跨任务需求。然而，与其他遥感传感器上基础模型的成功相比，缺乏大规模、标准化数据集，制约了可泛化的数据驱动 SAR ATR 方法发展。数据敏感性、采集成本、标注难度以及 SAR 成像本身的复杂性，阻碍了开放的大规模数据生态建设：超过 50% 的 SAR ATR 研究仍依赖 20 世纪 90 年代的 **Moving and Stationary Target Acquisition and Recognition（MSTAR）** 数据集。此外，既有基准评测协议不统一，也妨碍了算法的客观比较。因此，构建大规模 SAR ATR 数据集与基准，是释放该领域新模型能力的必要一步。我们以 **ATRNet** 为目标——面向目标特性建模与识别的海量、多样、标准化 SAR 目标数据集与基准。

<figure>
<div align="left">
<img src=Figure/fig3_Timeline.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>SAR 分类数据集时间线。</strong>相较其他多聚焦于简单场景目标切片的数据集，我们提供来自不同场景的多样目标样本，以研究更具难度的问题。本数据集是规模显著更大的 SAR ATR 数据集。
</div>
</div>
</figure>

**为何需要 ATRNet-STAR** — 研究者们已为 ATR 任务构建了多种 SAR 目标数据集，许多分类数据集显著提升了数据多样性。尤其是新的 SAR 检测数据集（SARDet‑100K、FAIR‑CSAR）已达到约 10 万量级图像。然而，我们此前关于 SAR 基础模型的研究表明：汇总公开数据后，可用目标样本仍不足 20 万，主因是样本严重不平衡（多集中于舰船检测数据集）。作为 ATRNet 的首期工作，我们聚焦**车辆目标**，原因有三：

1. 近期数据集多为星载 SAR 目标，成像几何受限；而机载平台可提供更高分辨率（0.1–0.3 m），并在复杂场景中实现更灵活的成像条件。
2. 自 20 世纪 90 年代 MSTAR 起，车辆识别已驱动 SAR ATR 研究约三十年，形成了较为成熟的研究体系。
3. 尽管 MSTAR 意义重大，但其采集条件偏理想、性能近乎饱和（准确率接近 99%），难以反映现实复杂性，也难以支撑 2020 年代的方法创新。

---

## 🛰️ 数据采集

以下从成像、标注与数据产品三方面介绍采集流程。

<figure>
<div align="left">
<img src=Figure/fig_acquisition_00.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>数据采集示意。</strong>我们基于标注与裁剪构建带元数据的目标切片。斜距复数数据的距离维沿视线方向，目标外形在该维会发生形变；因此，我们还基于斜距幅度数据经地距投影，提供地距图像。
</div>
</div>
</figure>

**成像** — 采用无人机（UAV）平台挂载传感器采集数据。双天线获取 X / Ku 波段全极化雷达回波；定位定向系统（POS）提供 GPS 与 IMU 信息，用于运动补偿。成像处理后得到斜距复数图像；地距数据则基于斜距幅度图像与 POS 信息经地距投影获得。

**标注** — 结合光学参考影像与布设记录，以矩形框标注目标类别与坐标。由于关注单体目标特性，布设时各目标保持规定间距。除本项目车辆外，场景中其他车辆标注为 “other”。标注完成后截取目标切片，并加入随机偏移。

**产品** — 提供两种坐标系下的数据产品。斜距数据的距离维为视线方向，并提供原始复数数据；地距图像投影至真实地距，并经非线性量化。对应标注文件包含基本图像信息，以及目标、场景与传感器参数。

---

## 📊 统计分析

以下给出本数据集的主要特征，以及相对其他数据集的对比统计。

<figure>
<div align="left">
<img src=Figure/fig_target_category.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>分类体系。</strong>民用车辆分类参考中国与欧洲相关车辆分类标准，按用途、结构、尺寸与质量划分；军用车辆则沿用 MSTAR 分类体系。<br>
(a) ATRNet-STAR 分类体系。本数据集构成覆盖 4 大类、21 亚类、40 型的民用车辆图谱。我们给出 40 型车辆的直方图分布，以体现数据广度与深度；样本充足且较为均衡，可支撑多样实验设置与研究。<br>
(b) SAR 车辆数据集的目标类别。我们统计了 SAR 车辆数据集中民用与军用车辆的类别与型号数量。相较实测 Gotcha 以及仿真 CVDomes、SARSim，本数据集显著丰富了 SAR ATR 可用的民用车辆种类。<br>
(c) 车辆缩写一览。
</div>
</div>
</figure>

**类别分布** — ATRNet-STAR 包含 4 大类、21 亚类、40 型车辆，样本较为均衡。相对其他 SAR 目标数据集，更高的多样性为细粒度 SAR 目标识别带来新的挑战；同时充足样本量保证了在各类实验设置下均可获得相当规模的目标类别，便于开展更丰富的研究。

<figure>
<div align="left">
<img src=Figure/fig7_target_info_00.jpg width="95%">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>ATRNet-STAR 中 40 类车辆的示意图像、亚类、型号与尺寸。</strong>图中给出 SAR 图像与对应目标。可见车辆在 SAR 图像中的显著差异主要来自体量与结构引起的散射特性变化，因此我们按尺寸与结构划分型号。尺寸（长×宽×高）以米（m）计。需要说明的是，本数据集的 SAR 图像并非如 MSTAR 那样目标理想地位于示意图像正中心，而是类似 QinetiQ 数据集存在随机偏移。
</div>
</div>
</figure>

**目标尺寸** — 40 类目标的长、宽、高见图示，可见结构多样、尺寸各异，且尺寸比例相近。相较另一民用车辆数据集 Gotcha，本数据集在目标类别与尺寸分布上更为宽广。

**参考目标** — 除布设角反射器用于分辨率测量外，我们还制作并布设了多几何结构的参考目标（类似 MSTAR 中的 “SLICY”），以支撑散射特性研究。

**非居中位置** — 多数 SAR 目标数据集将目标置于图像中心，或仅包含目标区域。遥感俯视视角不同于人眼习惯的“兴趣目标居中”，往往需要在更强干扰中搜索目标。我们随机加入位置偏移，通过非居中目标检测提高识别难度。

<figure>
<div align="center">
<img src=Figure/fig_scene_00.jpg width="55%">
<div align="left">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>不同场景对 SAR 图像中目标特性的影响。</strong>当前多数 SAR 目标分类数据集采集于简单场景。(a) MSTAR 采集于平坦草地，背景干净，目标与背景存在伪相关。(b) ATRNet-STAR 样本采集于多种场景的不同位置。由图 (b) 可见，同一目标在相同成像角度下，因场景不同会出现明显特征变化。例如，目标前后较高地物可造成遮挡与叠掩，削弱目标反射能量并增加非目标散射；困难场景中目标阴影亦不如 MSTAR 明显。(c) 遮挡与叠掩。相较无干扰物的裸土场景，工厂与林地更易出现遮挡与叠掩：前者中干扰物阴影可能遮蔽目标，后者中树木更易增强非目标散射。我们以同一角度、跨场景的单目标示意说明该问题；遮挡与叠掩实际由目标、干扰与成像几何共同决定。上述统计来自一款 SUV（长安 CSCS75 Plus，车辆 12）在三种场景地距图像中不同成像角度下的结果。
</div>
</div>
</figure>

**场景遮挡** — 既往车辆数据集多在理想环境中采集，而我们系统性地在多样场景中获取样本。图中展示了不同场景区域的影响，包括遮挡与叠掩：工厂中建筑与路侧树木阴影可能遮蔽目标并减弱目标散射；林地中邻近树木造成更明显遮挡，并增加非目标散射。

<figure>
<div align="center">
<img src=Figure/target_azimuth_00.jpg width="55%">
<div align="left">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>(a) MSTAR 与 (b) ATRNet-STAR 的目标方位角分布。</strong>相较 MSTAR（公开俯视角以 17°、15° 为主，且多数目标类别方位角不完整），ATRNet-STAR 为全部目标提供了均衡、完整的角度覆盖；但由于条带成像，方位角采样间隔相对更稀疏。
</div>
</div>
</figure>

**成像角度** — 不同成像角度影响目标散射特性，成像几何关系也会改变场景中的干扰。本数据集在不同俯视角与场景下具有多样且较为均衡的成像角度分布；但受条带模式约束与作业成本限制，方位角采样密度随场景而异。

<figure>
<div align="center">
<img src=Figure/fig_band_00.jpg width="55%">
<div align="left">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
<strong>不同波段与极化下强弱散射点比例。</strong>可见波段与极化对目标特征具有统计上显著的影响。统计数据取自沙地场景中车辆矩形框内的目标区域；对 8 比特地距图像，将像素值 ≥ 128 的点视为强散射点。
</div>
</div>
</figure>

**波段与极化** — 多数车辆数据集沿用 MSTAR 的 X 波段与 HH 极化设置。但图示表明，该设置并非唯一需要考虑的配置，波段与极化对目标散射有明显影响。我们提供双波段与全极化，以支持对传感器参数的讨论。

**复数数据** — 基于复数数据的特征提取是热点方向，因此我们提供以 `*.mat` 格式存储的斜距坐标系复数数据。

**图像比特数** — RGB 图像通常为 8 比特，而 SAR 图像数值动态范围更大。因此我们提供不同比特格式，以研究量化对增强弱散射点的影响。

**跨平台问题** — 我们在相同场景与目标下同步采集了少量卫星数据，以支撑后续版本中跨平台难度的研究。

**质量控制** — 为保证数据质量，我们统筹考虑了目标与场地租赁、空域申请、气象条件、目标布设与数据核查等挑战。8 人历时约六个月完成采集；另有 14 名标注人员与 2 名质检人员开展约四个月标注。整个项目历时近两年。

---

## 🏆 数据集价值

基于目标类别、场景、成像条件与数据格式的多样性，以及详尽标注，我们建议本数据集可用于以下 SAR 研究方向。

- **稳健识别** — 本数据集覆盖多样采集场景，每幅图像附带完备采集元数据，可细致讨论训练与测试采集条件变化时算法的稳健性。若将不同采集条件视为不同域，训练集可包含多域以学习域不变特征，即域泛化。

- **少样本学习** — 除采集条件变化外，SAR 采集成本高导致训练样本有限，也是关键研究议题。少样本设置可与稳健识别结合；本数据集大量目标类型亦可丰富既有少样本学习中元学习任务的构建。

- **迁移学习** — 样本多样性可用于研究预训练模型在不同 SAR 目标数据集之间的迁移问题；同时，大规模样本可进一步扩充 SAR 目标数据体量，推动自监督学习与基础模型研究。

- **增量学习** — 数据多样性支撑域增量学习与类别增量学习研究。域增量学习可提升动态过程中的算法稳健性；类别增量学习可逐步增强识别或拒识新类的能力。这些增量能力对开放环境中的 SAR ATR 至关重要。

- **物理信息深度学习** — SAR 图像具有复相位、极化等独特属性。本数据集提供多格式数据，便于利用这些内禀属性开展识别研究，而非仅依赖量化后的 SAR 幅度图；详尽元数据亦有望进一步提升 SAR ATR 效果。

- **生成模型** — 除识别任务外，本数据集还可支撑多样成像条件下目标样本的可控生成，以及跨采集场景的目标参数估计。

我们鼓励研究者基于本数据集提出新的实验设置与研究问题。如有新想法，欢迎随时与我们联系。

---

## ⚙️ ATRBench

我们基于本数据集构建了包含 **7 种实验设置**、**2 种数据格式** 的分类与检测评测基准，命名为 **ATRBench**。其中 2 种 SOC 设置采样自相近分布，5 种 EOC 设置存在明显分布偏移。数据格式包括地距坐标系下的幅度图像，以及斜距坐标系下的复数图像。

详情请参阅 <a href="ATRBench/README.md">ATRBench README</a>。

---

## 📜 声明

- 如有问题，请联系：**lwj2150508321@sina.com**
- 若本工作对您有帮助，欢迎在 GitHub 点亮 Star ⭐，并使用下列 BibTeX 引用相关论文：

```bibtex
@ARTICLE{liu2026atrnet,
  author={Liu, Yongxiang and Li, Weijie and Liu, Li and Zhou, Jie and Peng, Bowen and Song, Yafei and Xiong, Xuying and Yang, Wei and Liu, Tianpeng and Liu, Zhen and Li, Xiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={{ATRNet-STAR}: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild}, 
  year={2026},
  volume={48},
  number={6},
  pages={6735-6753},
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
```
