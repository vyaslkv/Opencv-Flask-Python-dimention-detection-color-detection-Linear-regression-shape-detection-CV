import json
import numpy as np


def default(o):
    if isinstance(o, np.int32): return int(o)
    raise TypeError


def json_create_for_maskrcnn(tempxf, tempyf, cntlen, pillname):
    """
    creates the pill dataset for the maskrcnn

    :param tempxf:
    :param tempyf:
    :param cntlen:
    :param pillname:
    :return:
    """
    data = {}
    pillname = pillname + '.jpg'
    data[pillname] = {}
    data[pillname]['regions'] = {}
    for i in range(cntlen):
        data[pillname]['regions'][str(i)] = {}
        data[pillname]['regions'][str(i)]['shape_attributes'] = {}
        data[pillname]['regions'][str(i)]['shape_attributes']['all_points_x'] = tempxf[i]
        data[pillname]['regions'][str(i)]['shape_attributes']['all_points_y'] = tempyf[i]
        data[pillname]['regions'][str(i)]['shape_attributes']['name'] = 'polygon'
        data[pillname]['regions'][str(i)]['region_attributes'] = {}
        data[pillname]['regions'][str(i)]['region_attributes']['Pill_Name'] = 'pill'
        data[pillname]['file_attributes'] = {}
        data[pillname]['file_attributes']['filename'] = pillname
        # keys= str(data.keys())
        # image_name_as_keys = keys.split()[1]

    with open('pill10.json') as f:
        feeds = json.loads(f.read())
    feeds.update(data)

    with open('pill10.json', 'w') as f:
        json.dump(feeds, f, default=default)