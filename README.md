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
```python
from unet2D_Dodnet_scale_token import UNet2D as UNet2D_scale

model = UNet2D_scale(num_classes=15, num_scale = 4, weight_std=False)
```

## How to define the Hierarchical Adaptive Taxonomy matrices
1. Hierarchical Scale Matrix 
```python
Area = np.zeros((15))
Area[0] = 2.434
Area[1] = 2.600
Area[2] = 1.760
Area[3] = 1.853
Area[4] = 1.844
Area[5] = 0.097
Area[6] = 0.360
Area[7] = 0.619
Area[8] = 0.466
Area[9] = 0.083
Area[10] = 0.002
Area[11] = 0.012
Area[12] = 0.001
Area[13] = 0.001
Area[14] = 0.002

Area_ratio = np.zeros((15, 15))
for xi in range(0,15):
    for yi in range(0,15):
        Area_ratio[xi,yi] = division_ratio(Area[xi], Area[yi])
```

2. Hierarchical Taxonomy Matrix
```python
HATs_matrix = np.zeros((15, 15))
'0_medulla'
HATs_matrix[0, 1] = 2  # 1_cortex
HATs_matrix[0, 2] = 2  # 2_cortexin
HATs_matrix[0, 3] = 2  # 3_cortexmiddle
HATs_matrix[0, 4] = 2  # 4_cortexout

HATs_matrix[0, 7] = 2  #7_cap
HATs_matrix[0, 8] = 2  #8_tuft
HATs_matrix[0, 10] = 2  #10_ptc
HATs_matrix[0, 11] = 1  #11_mv    medulla cover mv

HATs_matrix[0, 12] = 2  #12_pod
HATs_matrix[0, 13] = 2  #13_mes

'1_cortex'
HATs_matrix[1, 0] = 2  #0_medulla
HATs_matrix[1, 2] = 1  # 2_cortexin cortex cover cortexin
HATs_matrix[1, 3] = 1  # 3_cortexmiddle cortex cover cortexmiddle
HATs_matrix[1, 4] = 1  # 4_cortexout cortex cover cortexout

HATs_matrix[1, 7] = 1  # 7_cap cortex cover cap
HATs_matrix[1, 8] = 1  # 8_tuft cortex cover tuft
HATs_matrix[1, 10] = 1  # 10_ptc cortex cover ptc
HATs_matrix[1, 11] = 2  # 11_mv

HATs_matrix[1, 12] = 1  #12_pod cortex cover pod
HATs_matrix[1, 13] = 1  #13_mes cortex cover mes

......

'7_cap'
HATs_matrix[7, 0] = 2  #0_medulla
HATs_matrix[7, 1] = -1  #1_cortex  cap is covered by cortex but don't know between in/middle/out

HATs_matrix[7, 5] = 2  #5_dt
HATs_matrix[7, 6] = 2  #6_pt
HATs_matrix[7, 8] = 1  #8_tuft  cap covers tuft
HATs_matrix[7, 9] = 2  #9_art
HATs_matrix[7, 10] = 2  #10_ptc
HATs_matrix[7, 11] = 2  #11_mv

HATs_matrix[7, 12] = 1  #12_pod   cap cover pod
HATs_matrix[7, 13] = 1  #13_mes   cap cover mes
HATs_matrix[7, 14] = 2  #14_smooth

'8_tuft'
HATs_matrix[8, 0] = 2  #0_medulla
HATs_matrix[8, 1] = -1  #1_cortex  tuft is covered by cortex but don't know between in/middle/out

HATs_matrix[8, 5] = 2  #5_dt
HATs_matrix[8, 6] = 2  #6_pt
HATs_matrix[8, 7] = -1  #7_cap  tuft is covered by cap
HATs_matrix[8, 9] = 2  #9_art
HATs_matrix[8, 10] = 2  #10_ptc
HATs_matrix[8, 11] = 2  #11_mv

HATs_matrix[8, 12] = 1  #12_pod   tuft cover pod
HATs_matrix[8, 13] = 1  #13_mes   tuft cover mes
HATs_matrix[8, 14] = 2  #14_smooth

......

'12_pod'
HATs_matrix[12, 0] = 2  #0_medulla
HATs_matrix[12, 1] = -1  #1_cortex  pod is covered by cortex but don't know between in/middle/out

HATs_matrix[12, 5] = 2  #5_dt
HATs_matrix[12, 6] = 2  #6_pt
HATs_matrix[12, 7] = -1  #7_cap     pod is covered by cap
HATs_matrix[12, 8] = -1  #8_tuft    pod is covered by tuft
HATs_matrix[12, 9] = 2  #9_art
HATs_matrix[12, 10] = 2  #10_ptc
HATs_matrix[12, 11] = 2  #11_mv

HATs_matrix[12, 13] = 2  #13_mes
HATs_matrix[12, 14] = 2  #14_smooth

'13_mes'
HATs_matrix[13, 0] = 2  #0_medulla
HATs_matrix[13, 1] = -1  #1_cortex  pod is covered by cortex but don't know between in/middle/out

HATs_matrix[13, 5] = 2  #5_dt
HATs_matrix[13, 6] = 2  #6_pt
HATs_matrix[13, 7] = -1  #7_cap     med is covered by cap
HATs_matrix[13, 8] = -1  #8_tuft    med is covered by tuft
HATs_matrix[13, 9] = 2  #9_art
HATs_matrix[13, 10] = 2  #10_ptc
HATs_matrix[13, 11] = 2  #11_mv

HATs_matrix[13, 12] = 2  #12_pod
HATs_matrix[13, 14] = 2  #14_smooth

......
```

# How to define Anatomy Loss
```python
def HATs_learning(images, labels, batch_size, scales, model, now_task, weight, loss_seg_DICE, loss_seg_CE, term_seg_Dice, term_seg_BCE, term_all, HATs_matrix, semi_ratio, area_ratio):

	for ii in range(len(HATs_matrix[1])):
		now_task_semi = ii
		if now_task_semi == now_task:
			continue
		now_relative = HATs_matrix[now_task][now_task_semi]
		now_area_ratio = area_ratio[now_task][now_task_semi]

		if now_relative == 0:
			continue

		semi_preds = model(images, torch.ones(batch_size).cuda() * now_task_semi, scales)

		'Only use dice rather than bce in semi-supervised learning'

		if now_relative == 1:
			semi_labels = 1 - labels                        # Background from this label should not have any overlap with the pred, --> 0
			semi_labels = one_hot_3D(semi_labels.long())
			semi_seg_Dice, semi_seg_BCE, semi_all = get_loss(images, semi_preds, semi_labels, weight, loss_seg_DICE, loss_seg_CE)
			term_seg_Dice -= semi_ratio * semi_seg_Dice * now_area_ratio
			term_all -= semi_ratio * semi_seg_Dice * now_area_ratio


		elif now_relative == -1:
			semi_labels = labels
			semi_preds = semi_labels.unsqueeze(1).repeat(1,2,1,1) * semi_preds           # Only supervised the regions which have label  --> 1
			semi_labels = one_hot_3D(semi_labels.long())
			semi_seg_Dice, semi_seg_BCE, semi_all = get_loss(images, semi_preds, semi_labels, weight, loss_seg_DICE, loss_seg_CE)
			term_seg_Dice += semi_ratio * semi_seg_Dice * now_area_ratio
			term_all += semi_ratio * semi_seg_Dice * now_area_ratio

		elif now_relative == 2:
			semi_labels = labels                            # Foreground from this label should not have any overlap with the pred, --> 0
			semi_labels = one_hot_3D(semi_labels.long())
			semi_seg_Dice, semi_seg_BCE, semi_all = get_loss(images, semi_preds, semi_labels, weight, loss_seg_DICE, loss_seg_CE)
			term_seg_Dice -= semi_ratio * semi_seg_Dice * now_area_ratio
			term_all -= semi_ratio * semi_seg_Dice * now_area_ratio

	return term_seg_Dice, term_seg_BCE, term_all
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
