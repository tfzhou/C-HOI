import os
import cv2
import json
import numpy as np
from skimage import measure

import pycocotools.mask as mask

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def convert(split='train'):
    # v1.0
    error_list = ['indoor_05407.jpg', 'indoor_04289.jpg', 'indoor_01782.jpg',
                  'indoor_01775.jpg', 'indoor_00099.jpg', 'indoor_02584.jpg']
    # v2.0
    error_list = ['23382.png', '23441.png', '20714.png', '20727.png',
                  '23300.png', '21200.png']

    relation_file = 'data/pic/annotations/relations_{}_v2.0.json'.format(split)
    with open(relation_file) as f:
        rels = json.load(f)

    semantic_folder = '/raid/tfzhou/PIC_v2.0/semantic/{}'.format(split)
    instance_folder = '/raid/tfzhou/PIC_v2.0/instance/{}'.format(split)

    new_dict = {}
    for i, item in enumerate(rels):
        filename = item['name']
        image_id = item['image_id']
        relations = item['relations']
        num_relation = len(relations)

        if filename in error_list:
            continue

        print('{} {:06d} {}'.format(filename, image_id, num_relation))

        semantic_file = os.path.join(semantic_folder, filename[:-4] + '.png')
        instance_file = os.path.join(instance_folder, filename[:-4] + '.png')

        ins = cv2.imread(instance_file, 0)
        sem = cv2.imread(semantic_file, 0)

        new_dict[filename] = []
        for rel in relations:
            predicate_id = rel['relation']
            object_id = rel['object']

            sem_copy = sem.copy()
            sem_copy[ins != object_id] = 0
            id_list, counts = np.unique(sem_copy, return_counts=True)

            max_cnt = 0
            max_id = -1
            for id, c in zip(id_list, counts):
                if c > max_cnt and id != 0:
                    max_cnt = c
                    max_id = id

            object_category = max_id
            subject_category = 1

            assert object_category > 0

            subject_binary_mask = ins.copy()
            subject_binary_mask[ins != rel['subject']] = 0
            subject_binary_mask_encoded = mask.encode(
                np.asfortranarray(subject_binary_mask.astype(np.uint8)))
            subject_bounding_box = mask.toBbox(subject_binary_mask_encoded)
            subject_segmentation = binary_mask_to_polygon(
                subject_binary_mask, 2)

            object_binary_mask = ins.copy()
            object_binary_mask[ins != rel['object']] = 0
            object_binary_mask_encoded = mask.encode(
                np.asfortranarray(object_binary_mask.astype(np.uint8)))
            object_bounding_box = mask.toBbox(object_binary_mask_encoded)
            object_segmentation = binary_mask_to_polygon(object_binary_mask, 2)

            predicate = int(predicate_id)
            subject = {'category': int(subject_category),
                       'bbox': subject_bounding_box.tolist(),
                       'segm': subject_segmentation}
            object = {'category': int(object_category),
                      'bbox': object_bounding_box.tolist(),
                      'segm': object_segmentation}

            new_dict[filename].append({'predicate': predicate,
                                       'subject': subject,
                                       'object': object})

    with open('data/pic/annotations/new_relations_{}_v2.0.json'.format(split),
              'w') as f:
        json.dump(new_dict, f)


if __name__ == '__main__':
    convert('train')
    convert('val')
