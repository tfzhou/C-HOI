# Cascaded Human-Object Interaction Recognition

This repository contains the PyTorch implementation for [CVPR 2020](http://cvpr2020.thecvf.com/) Paper "Cascaded Human-Object Interaction Recognition" by [Tianfei Zhou](https://www.tfzhou.com/), [Wenguan Wang](https://sites.google.com/view/wenguanwang/), [Siyuan Qi](http://web.cs.ucla.edu/~syqi/), [Haibin Ling](https://www3.cs.stonybrook.edu/~hling/), [Jianbing Shen](https://scholar.google.com/citations?user=_Q3NTToAAAAJ&hl=en).

Our proposed method reached the __1st__ place in ICCV-2019 Person in Context Challenge (PIC19 Challenge), on both [_Human-Object Interaction in the Wild (HOIW)_](http://picdataset.com/challenge/leaderboard/hoi2019) and [_Person in Context (PIC)_](http://picdataset.com/challenge/leaderboard/pic2019) tracks.

![](../master/framework.png)

---
## Prerequisites
This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please follow [INSTALL.md](https://github.com/open-mmlab/mmdetection/blob/v1.0rc0/INSTALL.md) for installation.

## Prepare Dataset
For now, we only provide pre-trained weights for [PIC v2.0](http://picdataset.com/challenge/dataset/download/) and [HOIW](http://picdataset.com/challenge/dataset/download/) datasets. Please download these two datasets first.

## Download pre-trained weights
Will be added later.

## Testing

1. Run testing on the validation set of PIC v2.0
```python tools/test_pic.py configs/pic_v2.0/htc_rel_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_train_rel_dcn_semantichead.py pic_latest.pth --json_out det_result.json```

2. Run testing on the validation set of HOIW
```python tools/test_hoiw.py configs/hoiw/cascade_rcnn_x101_64x4d_fpn_1x_4gpu_rel.py hoiw_latest.pth --json_out det_result.json --hoiw_out hoiw_result.json```

## Citation
```
@inproceedings{zhou2020cascaded,
  title={Cascaded human-object interaction recognition},
  author={Zhou, Tianfei and Wang, Wenguan and Qi, Siyuan and Ling, Haibin and Shen, Jianbing},
  booktitle=CVPR,
  pages={4263--4272},
  year={2020}
}
```
