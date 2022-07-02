import json
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


WN_MAPPING = {}  # 'wnid' -> wn_name (for all wnid)
WN_TO_OURS_NAME = {}  # wn_name -> ours_name
OURS_NAME_TO_LABEL = {}  # ours_name -> ours_label
OURS_LABEL_TO_NAME = {}

WN_TO_OURS_NAME = {
    'chair': 'chair',
    'table': 'table',
    'cabinet': 'cabinet',
    'ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin': 'bin',
    'bookshelf': 'bookshelf',
    'display,video display': 'display',
    'sofa,couch,lounge': 'sofa',
    'bathtub,bathing tub,bath,tub': 'bath',
    'bed': 'bed',
    'file,file cabinet,filing cabinet': 'cabinet',
    'bag,traveling bag,travelling bag,grip,suitcase': 'others',
    'printer,printing machine': 'others',
    'lamp': 'others',
    'microwave,microwave oven': 'others',
    'stove': 'others',
    'basket,handbasket': 'others',
    'laptop,laptop computer': 'others',
    'washer,automatic washer,washing machine': 'others',
    'bench': 'sofa',
    'computer keyboard,keypad': 'others',
    'dishwasher,dish washer,dishwashing machine': 'others',
    'pot,flowerpot': 'others',
    'piano,pianoforte,forte-piano': 'others',
    'guitar': 'others',
    'faucet,spigot': 'others',
    'clock': 'others',
    'bowl': 'others',
    'pillow': 'others',
    'motorcycle,bike': 'others',
    'loudspeaker,speaker,speaker unit,loudspeaker system,speaker system': 'others',
    'bottle': 'others',
    'jar': 'others',
    'can,tin,tin can': 'others',
    'telephone,phone,telephone set': 'others',
    'cap': 'others',
}
"""
chair, mean size: [0.551 0.579 0.849]
table, mean size: [1.245 0.725 0.662]
cabinet, mean size: [0.889 0.561 0.956]
bin, mean size: [0.366 0.279 0.456]
bookshelf, mean size: [1.051 0.337 1.347]
display, mean size: [0.607 0.164 0.475]
sofa, mean size: [1.643 0.856 0.746]
bath, mean size: [0.853 0.516 0.439]
bed, mean size: [1.371 2.06  1.122]
bag, mean size: [0.366 0.289 0.5  ]
printer, mean size: [0.521 0.694 0.704]
lamp, mean size: [0.336 0.345 0.708]
others, mean size: [0.573 0.477 0.615]
"""
OURS_NAME_TO_LABEL = {
    'chair': 0,
    'table': 1,
    'cabinet': 2,
    'bin': 3,
    'bookshelf': 4,
    'display': 5,
    'sofa': 6,
    'bath': 7,
    'bed': 8,
    'others': 9,
}
N_CAT = len(OURS_NAME_TO_LABEL.items())

OURS_LABEL_TO_NAME = {v: k for k, v in OURS_NAME_TO_LABEL.items()}

jjd = json.load(open(os.path.join(BASE_DIR, "taxonomy.json")))
for d in jjd:
    id = d['synsetId']
    name = d['name']
    WN_MAPPING[id] = name

whos_your_parent = dict()
for d in jjd:
    Id = d["synsetId"]
    children = d["children"]
    for child in children:
        assert child not in whos_your_parent
        whos_your_parent[child] = Id

def find_your_parent(Id):
    while Id in whos_your_parent:
        Id = whos_your_parent[Id]
    return Id, WN_MAPPING[Id]

def wnid2name(wnid):
    _, wn_name = find_your_parent(wnid)
    return WN_TO_OURS_NAME[wn_name]

def wnid2label(wnid):
    name = wnid2name(wnid)
    return OURS_NAME_TO_LABEL[name]

# appearance = {}
# cad_app = json.load(open('cad_appearances.json'))  # dict of dict
# for scene in cad_app:
#     scene_dict = cad_app[scene]
#     for model in scene_dict:
#         synid = model[:8]
#         n_ap_this_model = scene_dict[model]
#         cls_idx, cls_name = find_your_parent(synid)
#         if cls_name in appearance:
#             appearance[cls_name] += 1
#         else:
#             appearance[cls_name] = 1
#
# appearance = {k: v for k, v in sorted(appearance.items(), key=lambda item: item[1], reverse=True)}
# x = []
# for k in appearance:
#     n_ap = appearance[k]
#     x.append(n_ap)
#     print('{}: {}'.format(k, n_ap))
# from scipy.io import savemat
# savemat('app_dist', {'dist': x})

"""
chair: 1988
table: 1938
cabinet: 1163
ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin: 795
bookshelf: 626
display,video display: 463
sofa,couch,lounge: 458
bathtub,bathing tub,bath,tub: 406
bed: 286
file,file cabinet,filing cabinet: 222
bag,traveling bag,travelling bag,grip,suitcase: 143
printer,printing machine: 116
lamp: 107
microwave,microwave oven: 99
stove: 92
basket,handbasket: 70
laptop,laptop computer: 50
washer,automatic washer,washing machine: 47
bench: 42
computer keyboard,keypad: 35
dishwasher,dish washer,dishwashing machine: 33
pot,flowerpot: 32
piano,pianoforte,forte-piano: 30
guitar: 19
faucet,spigot: 18
clock: 15
bowl: 12
pillow: 5
motorcycle,bike: 4
loudspeaker,speaker,speaker unit,loudspeaker system,speaker system: 3
bottle: 2
jar: 1
can,tin,tin can: 1
telephone,phone,telephone set: 1
cap: 1
"""

# NYU40_LABEL2NAME = {
#     0: "unlabeled",
#     1: "wall",
#     2: "floor",
#     3: "cabinet",
#     4: "bed",
#     5: "chair",
#     6: "sofa",
#     7: "table",
#     8: "door",
#     9: "window",
#     10: "bookshelf",
#     11: "picture",
#     12: "counter",
#     13: "blinds",
#     14: "desk",
#     15: "shelves",
#     16: "curtain",
#     17: "dresser",
#     18: "pillow",
#     19: "mirror",
#     20: "floor mat",
#     21: "clothes",
#     22: "ceiling",
#     23: "books",
#     24: "refridgerator",
#     25: "television",
#     26: "paper",
#     27: "towel",
#     28: "shower curtain",
#     29: "box",
#     30: "whiteboard",
#     31: "person",
#     32: "night stand",
#     33: "toilet",
#     34: "sink",
#     35: "lamp",
#     36: "bathtub",
#     37: "bag",
#     38: "otherstructure",
#     39: "otherfurniture",
#     40: "otherprop",
# }

# NYU40_NAME2LABEL = {v: k for k, v in NYU40_LABEL2NAME.items()}
#
# COMPATIBLE_MAP = {
#     'chair': ['chair', 'sofa'],
#     'table': ['table', 'desk', 'otherfurniture'],
#     'cabinet': ['cabinet', 'night stand', 'table'],
#     'bin': ['box', 'otherprop', 'otherfurniture'],
#     'bookshelf': ['bookshelf', 'shelves'],
#     'display': ['mirror', 'television', 'whiteboard', 'window', 'blinds', 'curtain', 'otherprop', 'otherfurniture'],
#     'sofa': ['sofa', 'chair'],
#     'bath': ['bathtub'],
#     'bed': ['bed'],
#     'bag': ['bag'],
#     'printer': ['otherfurniture', 'otherprop', 'shelves', 'cabinet', 'bookshelf'],
#     'lamp': ['lamp', 'otherprop'],
#     'others': ['otherfurniture', 'otherprop', 'picture', 'counter', 'pillow', 'refridgerator', 'box', 'sink']
# }
#
# COMPATIBLE_LABEL_MAP = {}
# for ours_name, nyu40_namelist in COMPATIBLE_MAP.items():
#     ours_id = OURS_NAME_TO_LABEL[ours_name]
#     nyu40_idlist = [NYU40_NAME2LABEL[nyu40_name] for nyu40_name in nyu40_namelist]
#     COMPATIBLE_LABEL_MAP[ours_id] = nyu40_idlist
#
# def check_compatible(nyu40_labels, ours_label):
#     """
#     Args:
#         nyu40_labels: np.array shaped (N,), int
#         ours_label: int
#     """
#     compatible_idx = nyu40_labels == 1e10  # all False, of course
#     unique_labels_this = np.unique(nyu40_labels)
#     print('checking: unique labels: {} ({}), ours label: {} ({})'.format(unique_labels_this, [NYU40_LABEL2NAME[l] for l in unique_labels_this], ours_label, OURS_LABEL_TO_NAME[ours_label]))
#     compatible_cls = COMPATIBLE_LABEL_MAP[ours_label]
#     print('compatible cls: {} ({})'.format(compatible_cls, [NYU40_LABEL2NAME[l] for l in compatible_cls]))
#     for c in compatible_cls:
#         compatible_with_this = nyu40_labels == c
#         compatible_idx = compatible_idx | compatible_with_this
#
#     print('compatible rate: {}'.format(compatible_idx.mean()))
#     if compatible_idx.mean() < 0.5:
#         print('Warning: low compatible rate!')
#     return compatible_idx