import numpy as np
import cv2
import os.path as osp
from .pic_eval import evaluate_from_dict
size = (640, 480)


def eval_impl(det_res, dataset, mode, result_dict, iou_threshes, rel_cats, geometric_rel_cats):
    assert mode in ['val', 'test']
    img_infos = dataset.img_infos
    img_prefix = dataset.img_prefix
    img_names = img_infos['filename']
    gt_rels = img_infos['rel_eval']

    for j in range(len(det_res)):
        img_name = img_names[j]

        gt_instance = cv2.imread(osp.join(img_prefix, 'segmentation/' + mode + '/instance/' + img_name + '.png'), 0)
        im_h = gt_instance.shape[0]
        im_w = gt_instance.shape[1]
        gt_instance = cv2.resize(gt_instance, dsize=size, interpolation=cv2.INTER_NEAREST)
        gt_semantic = cv2.imread(osp.join(img_prefix, 'segmentation/' + mode + '/semantic/' + img_name + '.png'), 0)
        gt_semantic = cv2.resize(gt_semantic, dsize=size, interpolation=cv2.INTER_NEAREST)

        gt_relations = gt_rels[img_name].copy()

        gt_entry = {
            'semantic': gt_semantic.astype(np.int32),
            'instance': gt_instance.astype(np.int32),
            'relations': gt_relations
        }

        boxes_i = det_res[j]['ret_bbox']
        masks_i = det_res[j]['ret_mask']
        objs_labels_i = det_res[j]['ret_label']
        relations = det_res[j]['ret_relation']

        if True:
            boxes_i = np.round(boxes_i).astype(np.int32)
            pred_semantic = np.zeros((im_h, im_w), dtype=np.uint8)  # np.zeros_like(gt_semantic)
            pred_instance = np.zeros((im_h, im_w), dtype=np.uint8)  # np.zeros_like(gt_instance)
            obj_order = np.argsort(objs_labels_i)[::-1]
            pred_relations = []
            for i, instance_id in enumerate(obj_order):
                ref_box = boxes_i[instance_id, :]
                w = ref_box[2] - ref_box[0] + 1
                h = ref_box[3] - ref_box[1] + 1
                w = np.maximum(w, 1)
                h = np.maximum(h, 1)
                padded_mask = np.zeros((28 + 2, 28 + 2), dtype=np.float32)
                padded_mask[1:-1, 1:-1] = masks_i[instance_id, :, :]
                mask = cv2.resize(padded_mask, (w, h))
                mask = np.array(mask >= 0.5, dtype=np.uint8)
                mask_category = mask * int(objs_labels_i[instance_id])
                # instance_id begins from 1
                mask_instance = mask * (instance_id + 1)
                x_0 = max(ref_box[0], 0)
                x_1 = min(ref_box[2] + 1, im_w)
                y_0 = max(ref_box[1], 0)
                y_1 = min(ref_box[3] + 1, im_h)

                nonbk = mask != 0
                pred_instance[y_0:y_1, x_0:x_1][nonbk] = mask_instance[
                                                         (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                                                         (x_0 - ref_box[0]):(x_1 - ref_box[0])
                                                         ][nonbk]
                pred_semantic[y_0:y_1, x_0:x_1][nonbk] = mask_category[
                                                         (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                                                         (x_0 - ref_box[0]):(x_1 - ref_box[0])
                                                         ][nonbk]
                pred_relations.append(relations[instance_id, :])

            #pred_relations[:, 0:2] += 1
            pred_semantic = cv2.resize(pred_semantic, dsize=size, interpolation=cv2.INTER_NEAREST)
            pred_instance = cv2.resize(pred_instance, dsize=size, interpolation=cv2.INTER_NEAREST)
            pred_entry = {
                'semantic': pred_semantic.astype(np.int32),
                'instance': pred_instance.astype(np.int32),
                'relations': pred_relations
            }

        evaluate_from_dict(gt_entry, pred_entry, result_dict, iou_threshes=iou_threshes,
                           rel_cats=rel_cats, geometric_rel_cats=geometric_rel_cats)


def pic_eval(det_res, dataset):
    rel_cats = {1: 'in front of', 2: 'behind', 3: 'talk', 4: 'next to', 5: 'hold', 6: 'drink', 7: 'sit on',
                8: 'stand on', 9: 'look', 10: 'touch', 11: 'wear', 12: 'carry', 13: 'in', 14: 'others', 15: 'use',
                16: 'lie down', 17: 'eat', 18: 'hit', 19: 'with', 20: 'drive', 21: 'ride', 22: 'squat', 23: 'pull',
                24: 'on the top of', 25: 'on', 26: 'kick', 27: 'throw', 28: 'play', 29: 'push', 30: 'feed',
                31: 'geometric_rel', 32: 'non_geometric_rel'}

    geometric_rel_cats = {1: 'in front of', 4: 'next to', 24: 'on the top of', 2: 'behind', 25: 'on', 13: 'in'}
    iou_threshes = [0.25, 0.5, 0.75]

    result_dict = {iou_thresh: {rel_cat_name: [] for rel_cat_name in rel_cats.values()} for iou_thresh in iou_threshes}

    eval_impl(det_res, dataset, 'val', result_dict, iou_threshes, rel_cats, geometric_rel_cats)

    for iou_thresh in iou_threshes:
        print('----------IoU: %.2f(R@100)----------' % iou_thresh)

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

    print('----------Final Result(R@100)----------')
    final_result_iou_25 = (result_dict[0.25]['geometric_rel'] + result_dict[0.25]['non_geometric_rel']) / 2
    final_result_iou_50 = (result_dict[0.5]['geometric_rel'] + result_dict[0.5]['non_geometric_rel']) / 2
    final_result_iou_75 = (result_dict[0.75]['geometric_rel'] + result_dict[0.75]['non_geometric_rel']) / 2
    final_result_iou_average = (final_result_iou_25 + final_result_iou_50 + final_result_iou_75) / 3
    print('IoU(0.25): %.4f' % final_result_iou_25)
    print('IoU(0.5): %.4f' % final_result_iou_50)
    print('IoU(0.75): %.4f' % final_result_iou_75)
    print('Average: %.4f' % final_result_iou_average)
    return final_result_iou_average
