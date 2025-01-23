<h1 align="center"> NUDT4MSTAR: A New Dataset and Benchmark\\Towards SAR Target Recognition in the Wild </h1> 

<h5 align="center"><em> Yongxiang Liu, Weijie Li, Li Liu, Jie Zhou, Xuying Xiong, Bowen Peng, Yafei Song, Wei Yang, Tianpeng Liu, Zhen Liu, Xiang Li </em></h5>

<p align="center">
  <a href="#Introduction">Introduction</a> |
  <a href="#Updates">Updates</a> |
  <a href="#Statement">Statement</a>
</p >

## Introduction

This is the official repository for the dataset “NUDT4MSTAR: A New Dataset and Benchmark Towards SAR Target Recognition in the Wild”.

**Abstract:** 
Synthetic Aperture Radar (SAR) stands as an indispensable sensor for Earth observation, owing to its unique capability for all-day imaging. Nevertheless, in a data-driven era, the scarcity of large-scale datasets poses a significant bottleneck to advancing SAR automatic target recognition (ATR) technology. This paper introduces NUDT4MSTAR, a large-scale SAR dataset for vehicle target recognition in the wild, including 40 target types and a wide array of imaging conditions across 5 different scenes. NUDT4MSTAR represents a significant leap forward in dataset scale, containing over 190,000 images—tenfold the size of its predecessors. To enhance the utility of this dataset, we meticulously annotate each image with detailed target information and imaging conditions. We also provide data in both processed magnitude images and original complex formats. Then, we construct a comprehensive benchmark consisting of 7 experiments with 15 recognition methods focusing on the stable and effective ATR issues. Besides, we conduct transfer learning experiments utilizing various models trained on NUDT4MSTAR and applied to three other target datasets, thereby demonstrating its substantial potential to the broader field of ground objects ATR. Finally, we discuss this dataset's application value and ATR's significant challenges. To the best of our knowledge, this work marks the first-ever endeavor to create a large-scale dataset benchmark for fine-grained SAR recognition in the wild, featuring an extensive collection of exhaustively annotated vehicle images. We expect that the open source of NUDT4MSTAR will facilitate the development of SAR ATR and attract a wider community of researchers.


## Updates
- [ ] Initial release of homepage. 🚀
- [ ] Release of datasets and benchmark codes. 
- [ ] Publication of aligned quad polarizable data. 
- [ ] Constructing rotated box detection and multi-resolution data
      
## Statement

- If you have any questions, please contact us at lwj2150508321@sina.com. 
- If you find our work is useful, please give us 🌟 in GitHub and cite our paper in the following BibTex format:

```
@article{li2024saratr,
  title={SARATR-X: Towards Building A Foundation Model for SAR Target Recognition},
  author={Li, Weijie and Yang, Wei and Hou, Yuenan and Liu, Li and Liu, Yongxiang and Li, Xiang},
  journal={arXiv preprint},
  url={https://arxiv.org/abs/2405.09365},
  year={2024}
}

@article{li2024predicting,
  title = {Predicting gradient is better: Exploring self-supervised learning for SAR ATR with a joint-embedding predictive architecture},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {218},
  pages = {326-338},
  year = {2024},
  issn = {0924-2716},
  doi = {https://doi.org/10.1016/j.isprsjprs.2024.09.013},
  url = {https://www.sciencedirect.com/science/article/pii/S0924271624003514},
  author = {Li, Weijie and Yang, Wei and Liu, Tianpeng and Hou, Yuenan and Li, Yuxuan and Liu, Zhen and Liu, Yongxiang and Liu, Li},
}
```
