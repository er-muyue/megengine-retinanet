import os
import json
import argparse


def convert_odgt2coco(odgtpath, targetclass):
    """
    将具备常用的ODGT格式转变成coco的格式便于复用评测代码, 格式如下:
        {
        'height': 1080,
        'width': 1920,
        'ID': '7657807,6c1463000028849ff',
        'gtboxes': [{'box': [969, 365, 40, 45], 'tag': '18duan'}]}
    对于不同格式的odgt/json文件, 你应该修改本函数保证符合coco要求
    :param odgtpath:
    :param targetclass: 目标类别是
    :return:
    """

    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []

    category_set = dict()

    targetclass2id = {}

    for (i, name) in enumerate(targetclass):
        # name = name.replace('#', '_')
        targetclass2id[name] = i

    category_item_id = len(targetclass)
    annotation_id = 1  # annotation从1开始

    def add_cat_item(name, category_item_id):
        category_item = dict()
        category_item['supercategory'] = 'none'
        category_item['id'] = category_item_id
        category_item['name'] = name
        coco['categories'].append(category_item)
        category_set[name] = category_item_id
        category_item_id += 1
        return category_item_id

    def add_img_item(file_name, size, nori_id, image_id):
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = file_name
        image_item['width'] = size['width']
        image_item['height'] = size['height']
        #         image_item['use_nori'] = True
        #         image_item['nori_id'] = nori_id
        coco['images'].append(image_item)

    def add_anno_item(object_name, image_id, category_id, bbox, annotation_id):
        # bbox is x,y,w,h
        annotation_item = dict()
        annotation_item['segmentation'] = []
        seg = []
        # left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        # left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        # right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        # right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])
        annotation_item['segmentation'].append(seg)
        annotation_item['area'] = bbox[2] * bbox[3]
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = image_id
        annotation_item['bbox'] = bbox
        annotation_item['category_id'] = category_id
        annotation_item['id'] = annotation_id
        coco['annotations'].append(annotation_item)
        annotation_id += 1
        return annotation_id

    for name in targetclass2id:
        add_cat_item(name, targetclass2id[name])

    f = open(odgtpath)
    # print(odgtpath)
    itemlist = f.readlines()
    f.close()

    for i, lineinfo in enumerate(itemlist):
        item = json.loads(lineinfo.strip())

        filename = '{}.jpg'.format(item['ID'])
        size = dict()
        size['width'] = item['width']
        size['height'] = item['height']
        size['depth'] = 3

        add_img_item(filename, size, item['ID'], i)

        if 'gtboxes' in item:
            for bbox in item['gtboxes']:
                object_name = bbox['tag']
                if object_name in targetclass2id:
                    current_category_id = targetclass2id[object_name]
                else:
                    if object_name not in category_set:
                        current_category_id = copy.deepcopy(category_item_id)
                        category_item_id = add_cat_item(object_name, category_item_id)
                    else:
                        current_category_id = category_set[object_name]

                bbox = bbox['box']
                annotation_id = add_anno_item(object_name, i, current_category_id, bbox, annotation_id)

    print(targetclass2id)

    return coco


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filename", default="demo.odgt", type=str, help="the odgt path"
    )
    args = parser.parse_args()

    target_classnames = ['chongqigongmen']
    data = convert_odgt2coco(args.filename, target_classnames)
    savepath = args.filename.replace('odgt', 'json')
    with open(savepath, 'w') as f:
        json.dump(data, f)



if __name__ == "__main__":
    main()