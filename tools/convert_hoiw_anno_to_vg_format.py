import os
import cv2
import json
import numpy as np
from skimage import measure

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to
        approximated polygonal chain.
        If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant',
                                constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1
        # we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def convert():

    relation_file = '../data/hoiw/annotations/trainval.json'
    with open(relation_file) as f:
        rels = json.load(f)

    new_dict = {}

    for i, item in enumerate(rels):
        filename = item['file_name']
        annotations = item['annotations']
        hoi_annotations = item['hoi_annotation']

        new_dict[filename] = []

        for hoi_anno in hoi_annotations:
            subject_id = hoi_anno['subject_id']
            object_id = hoi_anno['object_id']
            category_id = hoi_anno['category_id']

            try:
                subject_category = int(annotations[subject_id]['category_id'])
                object_category = int(annotations[object_id]['category_id'])
                subject_bbox = annotations[subject_id]['bbox']
                object_bbox = annotations[object_id]['bbox']

                w = subject_bbox[2] - subject_bbox[0]
                h = subject_bbox[3] - subject_bbox[1]
                x = subject_bbox[0] + w / 2
                y = subject_bbox[1] + h / 2
                subject_bbox = [x, y, w, h]

                w = object_bbox[2] - object_bbox[0]
                h = object_bbox[3] - object_bbox[1]
                x = object_bbox[0] + w / 2
                y = object_bbox[1] + h / 2
                object_bbox = [x, y, w, h]
            except:
                continue

            if subject_category != 1 or object_category == 1:
                continue

            predicate = int(category_id)
            subject = {'category': int(subject_category),
                       'bbox': subject_bbox}
            object = {'category': int(object_category),
                      'bbox': object_bbox}

            new_dict[filename].append({'predicate': predicate,
                                       'subject': subject,
                                       'object': object})

    with open('../data/hoiw/annotations/relations_all.json', 'w') as f:
        json.dump(new_dict, f)


if __name__ == '__main__':
    convert()
