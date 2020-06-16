from functools import reduce
import numpy as np
import cv2
import json
import os
import sys
import glob
from tqdm import trange

import matplotlib.pyplot as plt

categories = [
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

class PIC(object):
    def __init__(self, semantic_path, instance_path, relation_json, mode='gt', size=None, top_k=100):
        assert mode in ('gt', 'pred')
        self.semantic_path = semantic_path
        self.instance_path = instance_path
        self.mode = mode
        self.size = size
        self.top_k = top_k
        self.img2rels = dict()
        semantic_names = [name[:-4] for name in os.listdir(semantic_path) if name.endswith('.png')]
        instance_names = [name[:-4] for name in os.listdir(instance_path) if name.endswith('.png')]
        assert semantic_names == instance_names
        semantic_names.sort(key=str.lower)
        self.img_names = semantic_names
        self.img_names = [x for x in self.img_names if x != '27090']
        img_relations = json.load(open(relation_json, 'r'))
        assert type(img_relations) == list, 'relation file format {} not supported'.format(type(img_relations))
        self.img_relations = img_relations
        self.create_index()

    def create_index(self):
        for img_relation in self.img_relations:
            if self.mode == 'gt':
                rel_numpy = np.empty((0, 3), dtype=np.int32)
                for index, rel in enumerate(img_relation['relations']):
                    temp = np.array([[rel['subject'], rel['object'], rel['relation']]], dtype=np.int32)
                    rel_numpy = np.concatenate((rel_numpy, temp), axis=0)
                self.img2rels[img_relation['name']] = rel_numpy
            elif self.mode == 'pred':
                rels = []
                scores = []
                for index, rel in enumerate(img_relation['relations']):
                    temp = np.array([[rel['subject'], rel['object'], rel['relation']]], dtype=np.int32)
                    rels.append(temp)
                    scores.append(np.array([rel['score']], dtype=np.float64))
                if len(rels) != 0:
                    rel_numpy = np.concatenate(rels, axis=0)
                    score_numpy = np.concatenate(scores, axis=0)
                    # descending sort by score
                    self.img2rels[img_relation['name']] = rel_numpy[np.argsort(-score_numpy)][:self.top_k]
                else:
                    # there is no pred_rels in this img and [-1, -1, -1] is added for convenience
                    self.img2rels[img_relation['name']] = np.array([[-1, -1, -1]], dtype=np.int32)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_semantic = cv2.imread(
            os.path.join(self.semantic_path, img_name+'.png'),
            flags=cv2.IMREAD_GRAYSCALE)
        img_instance = cv2.imread(
            os.path.join(self.instance_path, img_name+'.png'),
            flags=cv2.IMREAD_GRAYSCALE)
        if self.size is not None:
            img_semantic = cv2.resize(img_semantic, dsize=self.size,
                                      interpolation=cv2.INTER_NEAREST)
            img_instance = cv2.resize(img_instance, dsize=self.size,
                                      interpolation=cv2.INTER_NEAREST)
        entry = {
            'semantic': img_semantic.astype(np.int32),
            'instance': img_instance.astype(np.int32),
            'relations': self.img2rels[img_name + '.jpg']
        }
        return entry

    def __len__(self):
        return len(self.img_names)

def compute_bbox_iou(target_mask, query_masks):
    pass


def compute_iou(target_mask, query_masks):
    N = query_masks.shape[0]
    target_masks = np.repeat(target_mask[None], N, axis=0)
    target_masks = target_masks.astype(np.int32)
    query_masks = query_masks.astype(np.int32)
    I = target_masks & query_masks
    I = I.sum(axis=2).sum(axis=1)
    U = target_masks | query_masks
    U = U.sum(axis=2).sum(axis=1) + sys.float_info.min
    return I/U


def intersect_2d(x1, x2):

    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res


def triplet(rels, semantic, instance):
    subs = rels[:, 0]
    objs = rels[:, 1]
    rel_cats = rels[:, 2]
    subs_mask = instance == subs[:, None, None]
    objs_mask = instance == objs[:, None, None]
    subs_semantic = subs_mask * semantic
    objs_semantic = objs_mask * semantic
    subs_class = subs_semantic.reshape(subs_semantic.shape[0], -1).max(axis=1)
    objs_class = objs_semantic.reshape(objs_semantic.shape[0], -1).max(axis=1)
    triplet_rels = np.concatenate((subs_class[:, None], objs_class[:, None], rel_cats[:, None]), axis=1)
    triplet_masks = np.concatenate((subs_mask[:, None, :, :], objs_mask[:, None, :, :]), axis=1)
    return triplet_rels, triplet_masks


def evaluate_from_dict(gt_entry, pred_entry, result_dict, iou_threshes, rel_cats, geometric_rel_cats):

    gt_rels = gt_entry['relations']
    gt_semantic = gt_entry['semantic']
    gt_instance = gt_entry['instance']
    gt_rels_nums = [0 for x in range(len(rel_cats))]
    for rel in gt_rels:
        gt_rels_nums[rel[2]-1] += 1
        if rel[2] in geometric_rel_cats.keys():
            gt_rels_nums[-2] += 1
        else:
            gt_rels_nums[-1] += 1

    pred_rels = pred_entry['relations']
    pred_semantic = pred_entry['semantic']
    pred_instance = pred_entry['instance']

    gt_triplet_rels, gt_triplet_masks = triplet(gt_rels, gt_semantic, gt_instance)
    pred_triplet_rels, pred_triplet_masks = triplet(pred_rels, pred_semantic, pred_instance)
    keeps = intersect_2d(gt_triplet_rels, pred_triplet_rels)

    gt_has_match = keeps.any(1)
    pred_to_gt = {}
    for iou_thresh in iou_threshes:
        pred_to_gt[iou_thresh] = {}
        for rel_cat_id, rel_cat_name in rel_cats.items():
            pred_to_gt[iou_thresh][rel_cat_name] = [[] for x in range(pred_rels.shape[0])]
    for gt_ind, gt_mask, keep_inds in zip(np.where(gt_has_match)[0], gt_triplet_masks[gt_has_match], keeps[gt_has_match]):
        masks = pred_triplet_masks[keep_inds]
        sub_iou = compute_iou(gt_mask[0], masks[:, 0])
        obj_iou = compute_iou(gt_mask[1], masks[:, 1])
        for iou_thresh in iou_threshes:
            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)
            for i in np.where(keep_inds)[0][inds]:
                for rel_cat_id, rel_cat_name in rel_cats.items():
                    if gt_triplet_rels[int(gt_ind), 2] == rel_cat_id:
                        pred_to_gt[iou_thresh][rel_cat_name][i].append(int(gt_ind))
                if gt_triplet_rels[int(gt_ind), 2] in geometric_rel_cats.keys():
                    pred_to_gt[iou_thresh]['geometric_rel'][i].append(int(gt_ind))
                else:
                    pred_to_gt[iou_thresh]['non_geometric_rel'][i].append(int(gt_ind))
    for iou_thresh in iou_threshes:
        for rel_cat_id, rel_cat_name in rel_cats.items():
            match = reduce(np.union1d, pred_to_gt[iou_thresh][rel_cat_name])
            if gt_rels_nums[rel_cat_id-1] == 0:
                # None means this rel_cat_id does not appear in gt_rels in this img
                rec_i = None
            else:
                rec_i = float(len(match)) / float(gt_rels_nums[rel_cat_id-1])
            result_dict[iou_thresh][rel_cat_name].append(rec_i)


def convert_npy(relation_npy_dir, image_dir, pred_root):
    semantic_folder = os.path.join(pred_root, 'semantic')
    instance_folder = os.path.join(pred_root, 'instance')
    if not os.path.exists(semantic_folder):
        os.makedirs(semantic_folder)
    if not os.path.exists(instance_folder):
        os.makedirs(instance_folder)

    npy_files = sorted(glob.glob(os.path.join(relation_npy_dir, '*.npy')))

    print('number of files: ', relation_npy_dir, len(npy_files))

    json_relations = []
    for jj in trange(len(npy_files)):
        npy_file = npy_files[jj]

        imagename = os.path.basename(npy_file)[:-4]

        if imagename == '27090.jpg':
            continue

        relation_results = np.load(npy_file)
        boxes_i = relation_results.item().get('ret_bbox')
        masks_i = relation_results.item().get('ret_mask')
        objs_labels_i = relation_results.item().get('ret_label')
        relations = relation_results.item().get('ret_relation')
        scale_factor = relation_results.item().get('scale_factor')
        ori_shape = relation_results.item().get('ori_shape')
        det_scores = boxes_i[:, -1]

        semantic_result_file = os.path.join(semantic_folder, imagename[:-4] + '.png')
        instance_result_file = os.path.join(instance_folder, imagename[:-4] + '.png')

        imagefile = os.path.join(image_dir, imagename)
        image = cv2.imread(imagefile)
        im_h = image.shape[0]
        im_w = image.shape[1]

        labels = objs_labels_i + 1 # object label begins from 1
        boxes_i = np.round(boxes_i).astype(np.int32)

        inds = np.where(det_scores > 0.3)[0]
        boxes_i = boxes_i[inds, :]
        labels = labels[inds]
        masks_i = masks_i[inds, :, :, :]

        # sort by object sizes
        object_sizes = []
        for i in range(len(labels)):
            bbox = boxes_i[i, :4]
            x_0 = max(bbox[0], 0)
            x_1 = min(bbox[2] + 1, im_w)
            y_0 = max(bbox[1], 0)
            y_1 = min(bbox[3] + 1, im_h)
            w = x_1 - x_0
            h = y_1 - y_0
            mask = masks_i[i, labels[i], :, :]
            mask = cv2.resize(mask, (w, h))
            mask = np.array(mask >= 0.5, dtype=np.uint8)
            object_size = np.sum(mask)
            object_sizes.append(object_size)
        object_order = np.argsort(object_sizes)[::-1]

        pred_semantic = np.zeros((im_h, im_w), dtype=np.uint8)
        pred_instance = np.zeros((im_h, im_w), dtype=np.uint8)

        #fig, ax = plt.subplots(1, 1, figsize=(10,10))
        #ax.imshow(image[:, :, ::-1])

        det_id_to_ins_id = {}
        for i, instance_id in enumerate(object_order):
            det_id_to_ins_id[inds[instance_id]] = instance_id + 1

            bbox = boxes_i[instance_id, :4]
            label = labels[instance_id]

            x_0 = max(bbox[0], 0)
            x_1 = min(bbox[2] + 1, im_w)
            y_0 = max(bbox[1], 0)
            y_1 = min(bbox[3] + 1, im_h)
            w = x_1 - x_0
            h = y_1 - y_0

            # debug
            #ax.add_artist(plt.Rectangle((bbox[0], bbox[1]), w, h, fill=False, color='r'))
            #text = categories[label-1]['name']
            #ax.add_artist(plt.Text(bbox[0], bbox[1], text, size='x-large', color='r'))

            mask_pred_ = masks_i[instance_id, label, :, :]
            mask = cv2.resize(mask_pred_, (w, h))
            mask = np.array(mask > 0.5, dtype=np.uint8)
            mask_category = mask * int(label)
            mask_instance = mask * (instance_id + 1)

            nonbk = mask != 0
            pred_instance[y_0:y_1, x_0:x_1][nonbk] = mask_instance[
                                                     (y_0 - bbox[1]):(y_1 - bbox[1]),
                                                     (x_0 - bbox[0]):(x_1 - bbox[0])
                                                     ][nonbk]
            pred_semantic[y_0:y_1, x_0:x_1][nonbk] = mask_category[
                                                     (y_0 - bbox[1]):(y_1 - bbox[1]),
                                                     (x_0 - bbox[0]):(x_1 - bbox[0])
                                                     ][nonbk]

        pred_relations = relations

        rels = {
            'image_id': jj,
            'name': imagename,
            'relations': []
        }

        for rel in pred_relations:
            sbj_id = int(rel[0])
            obj_id = int(rel[1])
            rel_ids = rel[2]
            rel_ids = list(map(int, rel_ids))
            rel_scores = rel[3]
            rel_scores = list(map(float, rel_scores))

            if sbj_id in inds and obj_id in inds:
                sbj_ins_id = int(det_id_to_ins_id[sbj_id])
                obj_ins_id = int(det_id_to_ins_id[obj_id])
                if sbj_ins_id == obj_ins_id:
                    continue
                for kk, rel_id in enumerate(rel_ids):
                    score = det_scores[sbj_id] * det_scores[obj_id] * rel_scores[kk]
                    rels['relations'].append({
                        'subject': sbj_ins_id,
                        'object':  obj_ins_id,
                        'relation': rel_id,
                        'score': score
                    })

        json_relations.append(rels)

        cv2.imwrite(semantic_result_file, pred_semantic)
        cv2.imwrite(instance_result_file, pred_instance)

        #image_result_file = os.path.join(semantic_folder, imagename[:-4] + '.jpg')
        #ax.axis('off')
        #plt.savefig(image_result_file, dpi=100)
        #plt.close()

    json_file = os.path.join(pred_root, 'relations.json')
    with open(json_file, 'w') as f:
        json.dump(json_relations, f)


def main(eval_result, mode='val', ):
    top_ks = [20]
    size = (640, 480)
    gt_root = '/raid/tfzhou/PIC_v2.0/'
    if mode == 'vcoco':
        image_root = '../data/vcoco/test2014/'
    else:
        image_root = os.path.join(gt_root, 'image/{}'.format(mode))
    pred_root = '/raid/tfzhou/PIC_v2.0/result/{}'.format(eval_result)
    relation_npy_dir = os.path.join(gt_root, 'relation_npy/{}'.format(eval_result))

    #convert_npy(relation_npy_dir, image_root, pred_root)

    if mode == 'test' or mode == 'vcoco':
        print('Cheers! Let us submit the results!')
        return

    final_scores = []
    for top_k in top_ks:
        gt = PIC(semantic_path=os.path.join(gt_root, 'semantic/val'),
                 instance_path=os.path.join(gt_root, 'instance/val'),
                 relation_json=os.path.join(gt_root, 'relations_val.json'),
                 mode='gt', size=size)
        pred = PIC(semantic_path=os.path.join(pred_root, 'semantic'),
                   instance_path=os.path.join(pred_root, 'instance'),
                   relation_json=os.path.join(pred_root, 'relations.json'),
                   mode='pred', size=size, top_k=top_k)
        assert gt.img_names == pred.img_names

        # rel_cats are 1-30. 30: geometric_rel and 31: non_geometric_rel are added into rel_cats for convenience
        rel_cats = {1: 'in front of', 2: 'behind', 3: 'talk', 4: 'next to', 5: 'hold', 6: 'drink', 7: 'sit on',
                    8: 'stand on', 9: 'look', 10: 'touch', 11: 'wear', 12: 'carry', 13: 'in', 14: 'others', 15: 'use',
                    16: 'lie down', 17: 'eat', 18: 'hit', 19: 'with', 20: 'drive', 21: 'ride', 22: 'squat', 23: 'pull',
                    24: 'on the top of', 25: 'on', 26: 'kick', 27: 'throw', 28: 'play', 29: 'push', 30: 'feed',
                    31: 'geometric_rel', 32: 'non_geometric_rel'}
        geometric_rel_cats = {1: 'in front of', 4: 'next to', 24: 'on the top of', 2: 'behind', 25: 'on', 13: 'in'}

        iou_threshes = [0.25, 0.5, 0.75]
        result_dict = {iou_thresh: {rel_cat_name: [] for rel_cat_name in rel_cats.values()} for iou_thresh in iou_threshes}
        for index in trange(len(gt)):
            evaluate_from_dict(gt[index], pred[index], result_dict,
                               iou_threshes=iou_threshes,
                               rel_cats=rel_cats,
                               geometric_rel_cats=geometric_rel_cats)

        for iou_thresh in iou_threshes:
            print('----------IoU: {:.2f}(R@{})----------'.format(iou_thresh, top_k))
            for rel_cat_id, rel_cat_name in rel_cats.items():
                recalls = result_dict[iou_thresh][rel_cat_name]
                while None in recalls:
                    recalls.remove(None)
                if len(recalls) != 0:
                    recall_mean = float('%.4f' % np.mean(recalls))
                    result_dict[iou_thresh][rel_cat_name] = recall_mean
                    print('%s: %.4f' % (rel_cat_name, recall_mean))
                # if all of recalls are None, it means that rel_cat_id does not appear in all imgs
                else:
                    result_dict[iou_thresh][rel_cat_name] = None
                    print('%s does not appear in gt_rels' % rel_cat_name)

        print('----------Final Result(R@{})----------'.format(top_k))
        final_result_iou_25 = (result_dict[0.25]['geometric_rel'] + result_dict[0.25]['non_geometric_rel']) / 2
        final_result_iou_50 = (result_dict[0.5]['geometric_rel'] + result_dict[0.5]['non_geometric_rel']) / 2
        final_result_iou_75 = (result_dict[0.75]['geometric_rel'] + result_dict[0.75]['non_geometric_rel']) / 2
        print('IoU(0.25): %.4f' % final_result_iou_25)
        print('IoU(0.5): %.4f' % final_result_iou_50)
        print('IoU(0.75): %.4f' % final_result_iou_75)
        print('Average: %.4f' % ((final_result_iou_25 + final_result_iou_50 +
                                  final_result_iou_75) / 3))

        final_score = (final_result_iou_25 + final_result_iou_50 +
                       final_result_iou_75) / 3
        final_scores.append(final_score)

    print(final_scores)


if __name__ == '__main__':
    mode = 'val'

    eval_result = 'relation_npy_{}_r50_joint_training'.format(mode)

    main(mode=mode, eval_result=eval_result)


