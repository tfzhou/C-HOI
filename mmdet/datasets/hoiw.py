import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .coco import CocoDataset
from .registry import DATASETS

import json

@DATASETS.register_module
class HoiwDataset(CocoDataset):

    CATEGORIES = [
        {
            "name": "person",
            "id": 1
        },
        {
            "name": "cellphone",
            "id": 2
        },
        {
            "name": "cigarette",
            "id": 3
        },
        {
            "name": "drink",
            "id": 4
        },
        {
            "name": "food",
            "id": 5
        },
        {
            "name": "bicyle",
            "id": 6
        },
        {
            "name": "motorcycle",
            "id": 7
        },
        {
            "name": "horse",
            "id": 8
        },
        {
            "name": "ball",
            "id": 9
        },
        {
            "name": "document",
            "id": 10
        },
        {
            "name": "computer",
            "id": 11
        }
    ]

    CLASSES = [c['name'] for c in CATEGORIES]

    REL_CATEGORIES = [
    {
        "name": "smoking",
        "id": 1
    },
    {
        "name": "call",
        "id": 2
    },
    {
        "name": "play(mobile phone)",
        "id": 3
    },
    {
        "name": "eat",
        "id": 4
    },
    {
        "name": "drink",
        "id": 5
    },
    {
        "name": "ride",
        "id": 6
    },
    {
        "name": "hold",
        "id": 7
    },
    {
        "name": "kick(ball)",
        "id": 8
    },
    {
        "name": "read",
        "id": 9
    },
    {
        "name": "play(computer)",
        "id": 10
    }
]

    REL_CLASSES = [c['name'] for c in REL_CATEGORIES]

    def load_annotations(self, ann_file, rel_ann_file=False):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }

        if rel_ann_file is not None:
            with open(rel_ann_file) as f:
                rel_infos_eval = json.load(f)
            rel_info = {}
            for rels in rel_infos_eval:
                rs = []
                name = rels['file_name']
                relations = rels['hoi_annotation']
                for rel in relations:
                    rs.append([rel['subject_id'], rel['object_id'],
                               rel['category_id']])
                rel_info[name] = rs

        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            if rel_ann_file is not None:
                if info['filename'] not in rel_info:
                    print(info['filename'])
                else:
                    info['rel'] = rel_info[info['filename']]
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask)

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

    def _parse_ann_info(self, ann_info, with_mask=True):
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
        gt_instance = []
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
                gt_instance.append(ann['ins_id'])
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

        gt_instance = np.array(gt_instance)
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            instance_id=gt_instance)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens

        return ann
