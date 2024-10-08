# HATs: Hierarchical Adaptive Taxonomy Segmentation for Panoramic Pathology Image Analysis

### [[Project Page]](https://github.com/hrlblab/HATs)   [[CVPR 2024 Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Deng_PrPSeg_Universal_Proposition_Learning_for_Panoramic_Renal_Pathology_Segmentation_CVPR_2024_paper.html) [[MICCAI 2024 paper]](https://papers.miccai.org/miccai-2024/374-Paper1451.html)<br />


This is the official implementation of HATs: Hierarchical Adaptive Taxonomy Segmentation for Panoramic Pathology Image Analysis. <br />

**CVPR 2024 Paper** <br />
> [PrPSeg: Universal Proposition Learning for Panoramic Renal Pathology Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Deng_PrPSeg_Universal_Proposition_Learning_for_Panoramic_Renal_Pathology_Segmentation_CVPR_2024_paper.html) <br />
> Ruining Deng, Quan Liu, Can Cui, Tianyuan Yao, Jialin Yue, Juming Xiong, Lining Yu, Yifei Wu, Mengmeng Yin, Yu Wang, Shilin Zhao, Yucheng Tang, Haichun Yang, Yuankai Huo. <br />
> *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition * <br />

**MICCAI 2024 Paper** <br />
> [HATs: Hierarchical Adaptive Taxonomy Segmentation for Panoramic Pathology Image Analysis](https://papers.miccai.org/miccai-2024/374-Paper1451.html) <br />
>  Deng, Ruining and Liu, Quan and Cui, Can and Yao, Tianyuan and Xiong, Juming and Bao, Shunxing and Li, Hao and Yin, Mengmeng and Wang, Yu and Zhao, Shilin and Tang, Yucheng and Yang, Haichun and Huo, Yuankai. <br />
> *Proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024* <br />

## Abstract
Panoramic image segmentation in computational pathology presents a remarkable challenge due to the morphologically complex and variably scaled anatomy. For instance, the intricate organization in kidney pathology spans multiple layers, from regions like the cortex and medulla to functional units such as glomeruli, tubules, and vessels, down to various cell types. In this paper, we propose a novel Hierarchical Adaptive Taxonomy Segmentation (HATs) method, which is designed to thoroughly segment panoramic views of kidney structures by leveraging detailed anatomical insights. 

Our approach entails <br />
(1) the innovative HATs technique which translates spatial relationships among 15 distinct object classes into a versatile “plug-and-play” loss function that spans across regions, functional units, and cells, <br />
(2) the incorporation of anatomical hierarchies and scale considerations into a unified simple matrix representation for all panoramic entities, <br />
(3) the adoption of the latest AI foundation model (EfficientSAM) as a feature extraction tool to boost the model’s adaptability, yet eliminating the need for manual prompt generation in conventional segment anything model (SAM). Experimental findings demonstrate that the HATs method offers an efficient and effective strategy for integrating clinical insights and imaging precedents into a unified segmentation model across more than 15 categories. 

## Model Training
1. Use [Dataset_save_csv.py](https://github.com/hrlblab/HATs/blob/main/Dataset_save_csv.py) to generate data list csv.
1. Use [train_EfficientSAM_HATs.py](https://github.com/hrlblab/HATs/blob/main/train_EfficientSAM_HATs.py) to train the model.
2. Use [Testing_EfficientSAM.py](https://github.com/hrlblab/HATs/blob/main/Testing_EfficientSAM.py) to test the model.

## How to define the segmentation model
We provide two model backbones: <br />
1. A dynamic EfficientSAM backbone from HATs: <br />
```python
import os, sys
sys.path.append("/Data4/HATs/EfficientSAM_token_dynamichead_logits")
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

model = build_efficient_sam_vits(task_num = 15, scale_num = 4)
model.image_encoder.requires_grad_(False)
```

2. a token-based CNN backbone from PrPSeg: <br />
```
from unet2D_Dodnet_scale_token import UNet2D as UNet2D_scale

model = UNet2D_scale(num_classes=15, num_scale = 4, weight_std=False)
```


## Citation
```
@InProceedings{Deng_2024_CVPR,
    author    = {Deng, Ruining and Liu, Quan and Cui, Can and Yao, Tianyuan and Yue, Jialin and Xiong, Juming and Yu, Lining and Wu, Yifei and Yin, Mengmeng and Wang, Yu and Zhao, Shilin and Tang, Yucheng and Yang, Haichun and Huo, Yuankai},
    title     = {PrPSeg: Universal Proposition Learning for Panoramic Renal Pathology Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {11736-11746}
}

@InProceedings{Den_HATs_MICCAI2024,
        author = { Deng, Ruining and Liu, Quan and Cui, Can and Yao, Tianyuan and Xiong, Juming and Bao, Shunxing and Li, Hao and Yin, Mengmeng and Wang, Yu and Zhao, Shilin and Tang, Yucheng and Yang, Haichun and Huo, Yuankai},
        title = { { HATs: Hierarchical Adaptive Taxonomy Segmentation for Panoramic Pathology Image Analysis } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
        year = {2024},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15004},
        month = {October},
        page = {pending}
}

```
