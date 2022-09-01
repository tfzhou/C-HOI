# Cascaded Human-Object Interaction Recognition

This repository contains the PyTorch implementation for [CVPR 2020](http://cvpr2020.thecvf.com/) Paper "Cascaded Human-Object Interaction Recognition" by [Tianfei Zhou](https://www.tfzhou.com/), [Wenguan Wang](https://sites.google.com/view/wenguanwang/), [Siyuan Qi](http://web.cs.ucla.edu/~syqi/), [Haibin Ling](https://www3.cs.stonybrook.edu/~hling/), [Jianbing Shen](https://scholar.google.com/citations?user=_Q3NTToAAAAJ&hl=en).

Our proposed method reached the __1st__ place in ICCV-2019 Person in Context Challenge (PIC19 Challenge), on both [_Human-Object Interaction in the Wild (HOIW)_](http://picdataset.com/challenge/leaderboard/hoi2019) and [_Person in Context (PIC)_](http://picdataset.com/challenge/leaderboard/pic2019) tracks.

![](../master/framework.png)

---

**Update #1**: A new branch (pytorch-1.5.0) is created, with some bugs fixed. The branch will be easier to use. p.s. you will still see a warning on missing keys (e.g., sa.g.conv.bias), and I did not solve it yet but will try to figure it out later.

**Update #2**: The score of our model (i.e., 66.04%) on HOIW reported in our paper is obtained by an ensemble of multiple models. Here I only provided the best single model that I have, so it is reasonable that the model does not deliver a close score. I am running the evaluation on HOIW test set, and expect to report my performance for reference this week (hopefully 02.09.2022).


## Prerequisites
This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please follow [INSTALL.md](https://github.com/open-mmlab/mmdetection/blob/v1.0rc0/INSTALL.md) for installation.

The code will work for pytorch=1.5.0, mmdet=1.0rc0+65c1842, and mmcv=0.4.3. 

If you encounter problems on *.so files (e.g., undefined symbol in *.so), please try to delete all existing *.so files and rebuild mmdet. 

## Prepare Dataset

Please find the dataset from the PIC challenge website: [http://picdataset.com:8000/challenge/task/download/](http://picdataset.com:8000/challenge/task/download/)

Please download converted json files from [google drive](https://drive.google.com/file/d/1hjED1c0E3JWGn8MijpHrVmAs_gFxQew8/view?usp=sharing), and put them in the top-most directory.

## Download pre-trained weights
Download from [Google Drive](https://drive.google.com/drive/folders/1STX6aad2qxNS4wZkS1G5TuA8tyFqDcGY). 

Results on PIC and HOIW datasets are also provided.

## Testing

1. Run testing on the validation set of PIC v2.0

```python tools/test_pic.py configs/pic_v2.0/htc_rel_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_train_rel_dcn_semantichead.py pic_latest.pth --json_out det_result.json```

2. Run testing on the validation set of HOIW

```python tools/test_hoiw.py configs/hoiw/cascade_rcnn_x101_64x4d_fpn_1x_4gpu_rel.py hoiw_latest.pth --json_out det_result.json --hoiw_out hoiw_result.json```

## Citation
```
@article{zhou2021cascaded,
  title={Cascaded parsing of human-object interaction recognition},
  author={Zhou, Tianfei and Qi, Siyuan and Wang, Wenguan and Shen, Jianbing and Zhu, Song-Chun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={6},
  pages={2827--2840},
  year={2021},
  publisher={IEEE}
}

@inproceedings{zhou2020cascaded,
  title={Cascaded human-object interaction recognition},
  author={Zhou, Tianfei and Wang, Wenguan and Qi, Siyuan and Ling, Haibin and Shen, Jianbing},
  booktitle=CVPR,
  pages={4263--4272},
  year={2020}
}
```
