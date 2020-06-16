import numpy as np
from pycocotools.coco import COCO

from .registry import DATASETS
from .coco import CocoDataset

import json


@DATASETS.register_module
class PicDatasetV20(CocoDataset):

    STUFF = []

    CATEGORIES = [
        {
            "name": "human",
            "id": 1
        },
        {
            "name": "hat",
            "id": 2
        },
        {
            "name": "racket",
            "id": 3
        },
        {
            "name": "plant",
            "id": 4
        },
        {
            "name": "flag",
            "id": 5
        },
        {
            "name": "food",
            "id": 6
        },
        {
            "name": "cushion",
            "id": 7
        },
        {
            "name": "tent",
            "id": 8
        },
        {
            "name": "stick",
            "id": 9
        },
        {
            "name": "bag",
            "id": 10
        },
        {
            "name": "pot",
            "id": 11
        },
        {
            "name": "flower",
            "id": 12
        },
        {
            "name": "rug",
            "id": 13
        },
        {
            "name": "blackboard",
            "id": 14
        },
        {
            "name": "window",
            "id": 15
        },
        {
            "name": "phone",
            "id": 16
        },
        {
            "name": "car",
            "id": 17
        },
        {
            "name": "ball",
            "id": 18
        },
        {
            "name": "PC",
            "id": 19
        },
        {
            "name": "instrument",
            "id": 20
        },
        {
            "name": "fan",
            "id": 21
        },
        {
            "name": "rope",
            "id": 22
        },
        {
            "name": "electronics",
            "id": 23
        },
        {
            "name": "kitchen_island",
            "id": 24
        },
        {
            "name": "pillar",
            "id": 25
        },
        {
            "name": "horse",
            "id": 26
        },
        {
            "name": "basket",
            "id": 27
        },
        {
            "name": "book",
            "id": 28
        },
        {
            "name": "poke",
            "id": 29
        },
        {
            "name": "lamp",
            "id": 30
        },
        {
            "name": "guardrail",
            "id": 31
        },
        {
            "name": "floor",
            "id": 32
        },
        {
            "name": "scissor",
            "id": 33
        },
        {
            "name": "stairs",
            "id": 34
        },
        {
            "name": "kitchenware",
            "id": 35
        },
        {
            "name": "decoration",
            "id": 36
        },
        {
            "name": "document",
            "id": 37
        },
        {
            "name": "pen",
            "id": 38
        },
        {
            "name": "curtain",
            "id": 39
        },
        {
            "name": "microphone",
            "id": 40
        },
        {
            "name": "bottle",
            "id": 41
        },
        {
            "name": "towel",
            "id": 42
        },
        {
            "name": "brand",
            "id": 43
        },
        {
            "name": "digital",
            "id": 44
        },
        {
            "name": "tableware",
            "id": 45
        },
        {
            "name": "certificate",
            "id": 46
        },
        {
            "name": "box",
            "id": 47
        },
        {
            "name": "barrel",
            "id": 48
        },
        {
            "name": "umbrella",
            "id": 49
        },
        {
            "name": "bicycle",
            "id": 50
        },
        {
            "name": "pillow",
            "id": 51
        },
        {
            "name": "luggage",
            "id": 52
        },
        {
            "name": "tool",
            "id": 53
        },
        {
            "name": "toy",
            "id": 54
        },
        {
            "name": "cup",
            "id": 55
        },
        {
            "name": "cigarette",
            "id": 56
        },
        {
            "name": "door",
            "id": 57
        },
        {
            "name": "stalls",
            "id": 58
        },
        {
            "name": "money_coin",
            "id": 59
        },
        {
            "name": "building",
            "id": 60
        },
        {
            "name": "cabin",
            "id": 61
        },
        {
            "name": "ice",
            "id": 62
        },
        {
            "name": "stone",
            "id": 63
        },
        {
            "name": "track",
            "id": 64
        },
        {
            "name": "train",
            "id": 65
        },
        {
            "name": "prop",
            "id": 66
        },
        {
            "name": "road",
            "id": 67
        },
        {
            "name": "street_light",
            "id": 68
        },
        {
            "name": "body_building_apparatus",
            "id": 69
        },
        {
            "name": "military_equipment",
            "id": 70
        },
        {
            "name": "glass",
            "id": 71
        },
        {
            "name": "parachute",
            "id": 72
        },
        {
            "name": "ground",
            "id": 73
        },
        {
            "name": "snow",
            "id": 74
        },
        {
            "name": "amusement_facilities",
            "id": 75
        },
        {
            "name": "motorcycle",
            "id": 76
        },
        {
            "name": "net",
            "id": 77
        },
        {
            "name": "sidewalk",
            "id": 78
        },
        {
            "name": "shovel",
            "id": 79
        },
        {
            "name": "property",
            "id": 80
        },
        {
            "name": "wood",
            "id": 81
        },
        {
            "name": "beach",
            "id": 82
        },
        {
            "name": "water",
            "id": 83
        },
        {
            "name": "paddle",
            "id": 84
        },
        {
            "name": "straw",
            "id": 85
        },
        {
            "name": "skis",
            "id": 86
        },
        {
            "name": "field",
            "id": 87
        },
        {
            "name": "animal",
            "id": 88
        },
        {
            "name": "bridge",
            "id": 89
        },
        {
            "name": "bench",
            "id": 90
        },
        {
            "name": "grass",
            "id": 91
        },
        {
            "name": "mountain",
            "id": 92
        },
        {
            "name": "surfboard",
            "id": 93
        },
        {
            "name": "wall",
            "id": 94
        },
        {
            "name": "aircraft",
            "id": 95
        },
        {
            "name": "bulletin",
            "id": 96
        },
        {
            "name": "tree",
            "id": 97
        },
        {
            "name": "hoe",
            "id": 98
        },
        {
            "name": "bucket",
            "id": 99
        },
        {
            "name": "steps",
            "id": 100
        },
        {
            "name": "swimming_things",
            "id": 101
        },
        {
            "name": "fishing_rod",
            "id": 102
        },
        {
            "name": "table",
            "id": 103
        },
        {
            "name": "skateboard",
            "id": 104
        },
        {
            "name": "laptop",
            "id": 105
        },
        {
            "name": "radiator",
            "id": 106
        },
        {
            "name": "refrigerator",
            "id": 107
        },
        {
            "name": "painting/poster",
            "id": 108
        },
        {
            "name": "emblem",
            "id": 109
        },
        {
            "name": "stool",
            "id": 110
        },
        {
            "name": "handcart",
            "id": 111
        },
        {
            "name": "nameplate",
            "id": 112
        },
        {
            "name": "showcase",
            "id": 113
        },
        {
            "name": "lighter",
            "id": 114
        },
        {
            "name": "sculpture",
            "id": 115
        },
        {
            "name": "shelf",
            "id": 116
        },
        {
            "name": "chair",
            "id": 117
        },
        {
            "name": "cabinet",
            "id": 118
        },
        {
            "name": "clothes",
            "id": 119
        },
        {
            "name": "sink",
            "id": 120
        },
        {
            "name": "apparel",
            "id": 121
        },
        {
            "name": "gun",
            "id": 122
        },
        {
            "name": "stand",
            "id": 123
        },
        {
            "name": "sofa",
            "id": 124
        },
        {
            "name": "bed",
            "id": 125
        },
        {
            "name": "sled",
            "id": 126
        },
        {
            "name": "bird",
            "id": 127
        },
        {
            "name": "cat",
            "id": 128
        },
        {
            "name": "pram",
            "id": 129
        },
        {
            "name": "plate",
            "id": 130
        },
        {
            "name": "blender",
            "id": 131
        },
        {
            "name": "remote_control",
            "id": 132
        },
        {
            "name": "vase",
            "id": 133
        },
        {
            "name": "toaster",
            "id": 134
        },
        {
            "name": "boat",
            "id": 135
        },
        {
            "name": "blanket",
            "id": 136
        },
        {
            "name": "camel",
            "id": 137
        },
        {
            "name": "dog",
            "id": 138
        },
        {
            "name": "vegetation",
            "id": 139
        },
        {
            "name": "display",
            "id": 140
        },
        {
            "name": "banner",
            "id": 141
        },
        {
            "name": "elephant",
            "id": 142
        },
        {
            "name": "squirrel",
            "id": 143
        }
    ]

    CLASSES = [item['name'] for item in CATEGORIES]

    REL_CATEGORIES = [
    {
        "name": "in front of",
        "id": 1
    },
    {
        "name": "behind",
        "id": 2
    },
    {
        "name": "talk",
        "id": 3
    },
    {
        "name": "next to",
        "id": 4
    },
    {
        "name": "hold",
        "id": 5
    },
    {
        "name": "drink",
        "id": 6
    },
    {
        "name": "sit on",
        "id": 7
    },
    {
        "name": "stand on",
        "id": 8
    },
    {
        "name": "look",
        "id": 9
    },
    {
        "name": "touch",
        "id": 10
    },
    {
        "name": "wear",
        "id": 11
    },
    {
        "name": "carry",
        "id": 12
    },
    {
        "name": "in",
        "id": 13
    },
    {
        "name": "others",
        "id": 14
    },
    {
        "name": "use",
        "id": 15
    },
    {
        "name": "lie down",
        "id": 16
    },
    {
        "name": "eat",
        "id": 17
    },
    {
        "name": "hit",
        "id": 18
    },
    {
        "name": "with",
        "id": 19
    },
    {
        "name": "drive",
        "id": 20
    },
    {
        "name": "ride",
        "id": 21
    },
    {
        "name": "squat",
        "id": 22
    },
    {
        "name": "pull",
        "id": 23
    },
    {
        "name": "on the top of",
        "id": 24
    },
    {
        "name": "on",
        "id": 25
    },
    {
        "name": "kick",
        "id": 26
    },
    {
        "name": "throw",
        "id": 27
    },
    {
        "name": "play",
        "id": 28
    },
    {
        "name": "push",
        "id": 29
    },
    {
        "name": "feed",
        "id": 30
    }
]

    REL_CLASSES = [item['name'] for item in REL_CATEGORIES]

    def load_annotations(self, ann_file, rel_ann_file=None):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()

        if rel_ann_file is not None:
            with open(rel_ann_file) as f:
                rel_infos_eval = json.load(f)
            rel_info = {}
            for rels in rel_infos_eval:
                rs = []
                name = rels['name']
                relations = rels['relations']
                for rel in relations:
                    rs.append([rel['subject'], rel['object'], rel['relation']])
                rel_info[name] = rs

        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            if rel_ann_file is not None:
                info['rel'] = rel_info[info['filename']]
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask)

    def get_rel_info(self, rel_info):
        pass

    def _filter_imgs(self, min_size=32):
        error_list = []
        val_error_list = []

        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if img_info['filename'] in error_list:
                continue
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
            else:
                print(img_info['filename'], img_info['width'],
                      img_info['height'])
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

        ann = dict(bboxes=gt_bboxes,
                   labels=gt_labels,
                   bboxes_ignore=gt_bboxes_ignore,
                   instance_id=gt_instance)
        #ann = dict(bboxes=gt_bboxes,
        #           labels=gt_labels,
        #           bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def _augment_relations(self, rel_info, gt_labels, gt_instid):
        """
        <subject A, in front of, subject B> ==> <subject B, behind, subject A>
        """

        matched_pairs = {'in front of': 'behind',
                         'behind': 'in front of',
                         'next to': 'next to'}

        rel_info_augmented = rel_info.copy()
        for rel in rel_info:
            sbj_inst_id, obj_inst_id, rel_id = rel

            rel_name = PicDatasetV20.REL_CLASSES[rel_id-1]
            if rel_name in matched_pairs.keys():
                annotation_idx = gt_instid.tolist().index(obj_inst_id)
                semantic_label = gt_labels[annotation_idx]

                if semantic_label == 1:  # human
                    aug_rel_name = matched_pairs[rel_name]
                    aug_rel_id = \
                        PicDatasetV20.REL_CLASSES.index(aug_rel_name) + 1
                    aug_rel = [obj_inst_id, sbj_inst_id, aug_rel_id]
                    if aug_rel not in rel_info_augmented:
                        rel_info_augmented.append([obj_inst_id, sbj_inst_id,
                                                   aug_rel_id])

        return rel_info_augmented


