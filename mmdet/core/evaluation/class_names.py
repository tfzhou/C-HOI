import mmcv


def wider_face_classes():
    return ['face']


def voc_classes():
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]


def imagenet_det_classes():
    return [
        'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo',
        'artichoke', 'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam',
        'banana', 'band_aid', 'banjo', 'baseball', 'basketball', 'bathing_cap',
        'beaker', 'bear', 'bee', 'bell_pepper', 'bench', 'bicycle', 'binder',
        'bird', 'bookshelf', 'bow_tie', 'bow', 'bowl', 'brassiere', 'burrito',
        'bus', 'butterfly', 'camel', 'can_opener', 'car', 'cart', 'cattle',
        'cello', 'centipede', 'chain_saw', 'chair', 'chime', 'cocktail_shaker',
        'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',
        'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper',
        'digital_clock', 'dishwasher', 'dog', 'domestic_cat', 'dragonfly',
        'drum', 'dumbbell', 'electric_fan', 'elephant', 'face_powder', 'fig',
        'filing_cabinet', 'flower_pot', 'flute', 'fox', 'french_horn', 'frog',
        'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',
        'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger',
        'hammer', 'hamster', 'harmonica', 'harp', 'hat_with_a_wide_brim',
        'head_cabbage', 'helmet', 'hippopotamus', 'horizontal_bar', 'horse',
        'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',
        'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
        'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk_can',
        'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck_brace',
        'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener', 'perfume',
        'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza',
        'plastic_bag', 'plate_rack', 'pomegranate', 'popsicle', 'porcupine',
        'power_drill', 'pretzel', 'printer', 'puck', 'punching_bag', 'purse',
        'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator',
        'remote_control', 'rubber_eraser', 'rugby_ball', 'ruler',
        'salt_or_pepper_shaker', 'saxophone', 'scorpion', 'screwdriver',
        'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
        'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa', 'spatula',
        'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
        'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',
        'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie',
        'tiger', 'toaster', 'traffic_light', 'train', 'trombone', 'trumpet',
        'turtle', 'tv_or_monitor', 'unicycle', 'vacuum', 'violin',
        'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft',
        'whale', 'wine_bottle', 'zebra'
    ]


def imagenet_vid_classes():
    return [
        'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
        'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
        'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
        'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle',
        'watercraft', 'whale', 'zebra'
    ]


def coco_classes():
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]


def cityscapes_classes():
    return [
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


def pic_classes():
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

    return CLASSES


dataset_aliases = {
    'voc': ['voc', 'pascal_voc', 'voc07', 'voc12'],
    'imagenet_det': ['det', 'imagenet_det', 'ilsvrc_det'],
    'imagenet_vid': ['vid', 'imagenet_vid', 'ilsvrc_vid'],
    'coco': ['coco', 'mscoco', 'ms_coco'],
    'wider_face': ['WIDERFaceDataset', 'wider_face', 'WDIERFace'],
    'cityscapes': ['cityscapes'],
    'pic': ['pic']
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels
