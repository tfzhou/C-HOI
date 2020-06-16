from pycocotools.coco import COCO
from .custom import CustomDataset

import mmcv
import json
import warnings
import os.path as osp
import numpy as np
from imagecorruptions import corrupt
from mmcv.parallel import DataContainer as DC

from .utils import random_scale, to_tensor

from .registry import DATASETS


@DATASETS.register_module
class VisualGenomeDataset(CustomDataset):
    CLASSES = ["airplane", "animal", "arm", "bag", "banana", "basket", "beach", "bear", "bed", "bench", "bike",
               "bird", "board", "boat", "book", "boot", "bottle", "bowl", "box", "boy", "branch", "building",
               "bus", "cabinet", "cap", "car", "cat", "chair", "child", "clock", "coat", "counter", "cow", "cup",
               "curtain", "desk", "dog", "door", "drawer", "ear", "elephant", "engine", "eye", "face", "fence",
               "finger", "flag", "flower", "food", "fork", "fruit", "giraffe", "girl", "glass", "glove", "guy",
               "hair", "hand", "handle", "hat", "head", "helmet", "hill", "horse", "house", "jacket", "jean",
               "kid", "kite", "lady", "lamp", "laptop", "leaf", "leg", "letter", "light", "logo", "man", "men",
               "motorcycle", "mountain", "mouth", "neck", "nose", "number", "orange", "pant", "paper", "paw",
               "people", "person", "phone", "pillow", "pizza", "plane", "plant", "plate", "player", "pole", "post",
               "pot", "racket", "railing", "rock", "roof", "room", "screen", "seat", "sheep", "shelf", "shirt",
               "shoe", "short", "sidewalk", "sign", "sink", "skateboard", "ski", "skier", "sneaker", "snow",
               "sock", "stand", "street", "surfboard", "table", "tail", "tie", "tile", "tire", "toilet", "towel",
               "tower", "track", "train", "tree", "truck", "trunk", "umbrella", "vase", "vegetable", "vehicle",
               "wave", "wheel", "window", "windshield", "wing", "wire", "woman", "zebra"]

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 corruption=None,
                 corruption_severity=1,
                 skip_img_without_anno=True,
                 skip_img_without_rel=True,
                 test_mode=False,
                 ann_file_rel=None,
                 ann_file_pred=None,
                 with_rel=False):
        super(VisualGenomeDataset, self).__init__(ann_file,
                                                  img_prefix,
                                                  img_scale,
                                                  img_norm_cfg,
                                                  multiscale_mode,
                                                  size_divisor,
                                                  proposal_file,
                                                  num_max_proposals,
                                                  flip_ratio,
                                                  with_mask,
                                                  with_crowd,
                                                  with_label,
                                                  with_semantic_seg,
                                                  seg_prefix,
                                                  seg_scale_factor,
                                                  extra_aug,
                                                  resize_keep_ratio,
                                                  corruption,
                                                  corruption_severity,
                                                  skip_img_without_anno,
                                                  test_mode)

        self.ann_file_rel = ann_file_rel
        self.ann_file_pred = ann_file_pred
        self.with_rel = with_rel
        self.skip_img_without_rel = skip_img_without_rel

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # corruption
        if self.corruption is not None:
            img = corrupt(
                img,
                severity=self.corruption_severity,
                corruption_name=self.corruption)
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        gt_rels = ann['rels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0 and self.skip_img_without_anno:
            warnings.warn('Skip the image "%s" that has no valid gt bbox' %
                          osp.join(self.img_prefix, img_info['filename']))
            return None

        if len(gt_rels) == 0 and self.skip_img_without_rel:
            warnings.warn('Skip the image "%s" that has no valid rel' %
                          osp.join(self.img_prefix, img_info['filename']))
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix,
                         img_info['filename'].replace('jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack([proposals, scores
                                   ]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        return data

    def prepare_test_img(self, idx):
        pass

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask, self.with_rel)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=True, with_rel=False):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        if with_rel:
            with open(self.ann_file_rel) as f:
                ann['rels'] = json.load(f)
            with open(self.ann_file_pred) as f:
                ann['pred_cats'] = json.load(f)

        return ann
