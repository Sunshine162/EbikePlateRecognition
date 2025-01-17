import os
import os.path as osp
import sys
import json
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from skimage import transform


# SPLIT_CONFIG = {
#     'aligin_pattern': [
#         [3, 3],
#         [197, 3],
#         [197, 97],
#         [3, 97],
#         [3, 42],
#         [197, 42],
#     ],
#     'aligin_dsize': (200, 100),
#     'p1': {'left': 50, 'right': 150, 'top': 6, 'bottom': 40},
#     'p2': {'left': 6, 'right': 194, 'top': 38, 'bottom': 92},
# }
SPLIT_CONFIG = {
    'aligin_pattern': [
        [3, 3],
        [197, 3],
        [197, 97],
        [3, 97],
        [3, 42],
        [197, 42],
    ],
    'aligin_dsize': (200, 100),
    'p1': {'left': 50, 'right': 150, 'top': 6, 'bottom': 40},
    'p2': {'left': 6, 'right': 194, 'top': 38, 'bottom': 92},
}


def parse_annotation(json_path):
    with open(json_path, 'r') as f:
        anno_info = json.load(f)
    shapes = anno_info['shapes']
    img_size = (anno_info['imageWidth'], anno_info['imageHeight'])

    bboxes = []
    keypoints = []
    group_points = []

    for shape in shapes:
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            if points[0][0] < points[1][0]:
                bboxes.append(points)
            else:
                bboxes.append(points[::-1])
        elif shape['shape_type'] == 'point':
            group_points.append(shape['points'][0])
            if len(group_points) == 6:
                keypoints.append(group_points)
                group_points = []
        else:
            raise ValueError('')

    assert len(bboxes) == len(keypoints), f"file {json_path} is invalid!"

    return bboxes, keypoints, img_size


def get_bad_plates(bad_record_txt):
    bad_plates = {}
    cnt = 0
    with open(bad_record_txt, 'r', encoding='utf-8') as f:
        for line in f:
            img_name, plate_index, _ = line.strip().split(' ')
            img_name = img_name + ".jpg"
            if img_name not in bad_plates:
                bad_plates[img_name] = set()
            bad_plates[img_name].add(int(plate_index))
            cnt += 1
    print(f"Bad plates: {cnt}")
    return bad_plates


def check(keypoints, bad_plate_indices):
    _keypoints = np.array(keypoints, np.float32)
    flags = []

    for i, group_points in enumerate(_keypoints):
        if i in bad_plate_indices:
            flags.append(False)
            continue

        d1 = np.sqrt(np.sum(np.square(group_points[0] - group_points[1])))
        d2 = np.sqrt(np.sum(np.square(group_points[2] - group_points[3])))
        d3 = np.sqrt(np.sum(np.square(group_points[0] - group_points[3])))
        d4 = np.sqrt(np.sum(np.square(group_points[1] - group_points[2])))
        if (d1 + d2) / 2 < 94 or (d3 + d4) / 2 < 47:
            flags.append(False)
            continue

        # d5 = np.sqrt(np.sum(np.square(group_points[0] - group_points[2]))).item()
        # d6 = np.sqrt(np.sum(np.square(group_points[1] - group_points[3]))).item()
        # if min(d5, d6) / max(d5, d6) < 0.6:
        #     flags.append(False)
        #     continue

        flags.append(True)

    return flags


def display_bboxes_keypoints(src_path, bboxes, keypoints, dst_path):
    colors = [(21,167,203), (203,169,21), (203,99,21), (203,41,21), (148,203,21), (21,203,73)]

    img = cv2.imread(src_path)
    thickness = max(3, round((img.shape[0] + img.shape[1]) / 2000))
    radius = max(8, round((img.shape[0] + img.shape[1]) / 2000))
    
    for i, (bbox, group_points) in enumerate(zip(bboxes, keypoints)):
        bbox = np.array(bbox).round().astype(np.int64)
        group_points = np.array(group_points).round().astype(np.int64)

        cv2.rectangle(img, bbox[0], bbox[1], colors[i], thickness)
        for point in group_points:
            cv2.circle(img, point, radius, colors[i], -1)
        cv2.line(img, group_points[0], group_points[1], colors[i], thickness)
        cv2.line(img, group_points[1], group_points[2], colors[i], thickness)
        cv2.line(img, group_points[2], group_points[3], colors[i], thickness)
        cv2.line(img, group_points[3], group_points[0], colors[i], thickness)
        cv2.line(img, group_points[4], group_points[5], colors[i], thickness)

    cv2.imwrite(dst_path, img)


def plate_align(img, landmark):
    global SPLIT_CONFIG
    landmark = landmark.astype(np.float32)
    pattern = np.array(SPLIT_CONFIG['aligin_pattern'], np.float32)
    M = cv2.getPerspectiveTransform(landmark[:4, :], pattern[:4, :])
    return cv2.warpPerspective(img, M, SPLIT_CONFIG['aligin_dsize'])


def split_image(src_path, bboxes, keypoints, flags, dst_prefix):
    img = cv2.imread(src_path)
    h, w, _ = img.shape

    for i, (bbox, group_points, flag) in enumerate(zip(bboxes, keypoints, flags)):
        bbox = np.array(bbox).round().astype(np.int64)
        group_points = np.array(group_points).round().astype(np.int64)
        _dst_prefix = dst_prefix if flag else dst_prefix.replace('good', 'bad')

        # crop plate
        all_points = np.vstack([bbox, group_points])
        l = np.clip(all_points[:, 0].min() - 10, 0, w)
        t = np.clip(all_points[:, 1].min() - 10, 0, h)
        r = np.clip(all_points[:, 0].max() + 10, 0, w)
        b = np.clip(all_points[:, 1].max() + 10, 0, h)
        plate_img = img[t:b, l:r, :]

        # draw bbox and keypoints
        thickness = max(2, round((r-l+b-t) / 200))
        radius = max(3, round((r-l+b-t) / 200))
        _plate_img = plate_img.copy()
        _bbox = bbox - np.array([[l, t]])
        cv2.rectangle(_plate_img, _bbox[0], _bbox[1], (0, 0, 255), thickness)
        _group_points = group_points - np.array([[l, t]])
        colors = [(0,255,255), (0,255,0), (255,255,0), (255,0,0), (255,0,255), (0,128,255)]
        for j, point in enumerate(_group_points):
            cv2.circle(_plate_img, point, radius, colors[j], -1)
        cv2.imwrite(f"{_dst_prefix}-plate{i}.jpg", _plate_img)

        # align plate
        aligned_img = plate_align(plate_img, _group_points)
        _aligned_img = aligned_img.copy()
        cv2.line(_aligned_img, (0, 42), (200, 42), (0, 0, 255), 2)
        cv2.imwrite(f"{_dst_prefix}-plate{i}-aligned.jpg", _aligned_img)

        # crop city name and plate codes
        global SPLIT_CONFIG
        p1cfg, p2cfg = SPLIT_CONFIG['p1'], SPLIT_CONFIG['p2']
        p1 = aligned_img[p1cfg['top']:p1cfg['bottom'], p1cfg['left']:p1cfg['right'], :]
        p2 = aligned_img[p2cfg['top']:p2cfg['bottom'], p2cfg['left']:p2cfg['right'], :]
        cv2.imwrite(f"{_dst_prefix}-plate{i}-p1.jpg", p1)
        cv2.imwrite(f"{_dst_prefix}-plate{i}-p2.jpg", p2)
        cv2.imwrite(f"{_dst_prefix}-plate{i}-p1s.jpg", cv2.resize(p1, (94, 24)))
        cv2.imwrite(f"{_dst_prefix}-plate{i}-p2s.jpg", cv2.resize(p2, (94, 24)))


def create_label_file(bboxes, keypoints, flags, img_size, save_path):
    _bboxes = [bboxes[i] for i, flag in enumerate(flags) if flag]
    if len(_bboxes) == 0:
        return 
    _keypoints = [keypoints[i] for i, flag in enumerate(flags) if flag]

    img_w, img_h = img_size

    with open(save_path, 'w', encoding='utf-8') as f:
        for (bbox, group_points) in zip(_bboxes, _keypoints):
            label = 0

            x = (bbox[0][0] + bbox[1][0]) / 2 / img_w
            y = (bbox[0][1] + bbox[1][1]) / 2 / img_h
            w = abs(bbox[0][0] - bbox[1][0]) / img_w
            h = abs(bbox[0][1] - bbox[1][1]) / img_h

            pt1x = group_points[0][0] / img_w
            pt1y = group_points[0][1] / img_h
            pt2x = group_points[1][0] / img_w
            pt2y = group_points[1][1] / img_h
            pt3x = group_points[2][0] / img_w
            pt3y = group_points[2][1] / img_h
            pt4x = group_points[3][0] / img_w
            pt4y = group_points[3][1] / img_h

            f.write(f"{label} {x} {y} {w} {h} {pt1x} {pt1y} {pt2x} {pt2y} {pt3x} {pt3y} {pt4x} {pt4y}\n")


def split_data(input_dir, dataset_dir, val_ratio=0.15, test_ratio=0.15):
    label_dir = osp.join(dataset_dir, 'labels')
    image_names = [name.replace('.txt', '.jpg') for name in os.listdir(label_dir)]
    image_names.sort()

    assert 0 < val_ratio < 1 and 0 < test_ratio < 1
    assert 0 < val_ratio + test_ratio < 1
    num_total = len(image_names)
    num_val = int(val_ratio * num_total)
    num_test = int(test_ratio * num_total)
    num_train = num_total - num_val - num_test
    print(num_train, num_val, num_test)

    image_dir = osp.join(dataset_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    train_file = open(osp.join(dataset_dir, 'train.txt'), 'w', encoding='utf-8')
    val_file = open(osp.join(dataset_dir, 'val.txt'), 'w', encoding='utf-8')
    test_file = open(osp.join(dataset_dir, 'test.txt'), 'w', encoding='utf-8')

    for i, image_name in enumerate(image_names):
        src_path = osp.join(input_dir, image_name)
        dst_path = osp.join(image_dir, image_name)
        shutil.copy(src_path, dst_path)
        if i < num_train:
            train_file.write(dst_path + '\n')
        elif num_train <= i < num_train + num_val:
            val_file.write(dst_path + '\n')
        else:
            test_file.write(dst_path + '\n')


def main():
    input_dir = 'frames'
    display_dir = 'display'
    plate_dir = 'plates/good'
    dataset_dir = 'datasets'
    bad_record_txt = 'bad_plates.txt'
    label_dir = osp.join(dataset_dir, 'labels')
    os.makedirs(display_dir, exist_ok=True)
    os.makedirs(plate_dir, exist_ok=True)
    os.makedirs(plate_dir.replace('good', 'bad'), exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    bad_plates = get_bad_plates(bad_record_txt)
    json_list = [x for x in os.listdir(input_dir) if x.endswith('.json')]

    for json_name in tqdm(json_list):
        # 获取标注结果
        json_path = osp.join(input_dir, json_name)
        bboxes, keypoints, img_size = parse_annotation(json_path)

        # 获取原图路径
        image_name = json_name.replace('.json', '.jpg')
        src_path = osp.join(input_dir, image_name)

        # 判断车牌是否有效
        flags = check(keypoints, bad_plates.get(image_name, set()))

        # 展示标注结果
        dst_path = osp.join(display_dir, image_name)
        display_bboxes_keypoints(src_path, bboxes, keypoints, dst_path)

        # 扣出车牌图片、校正、分割城市名与车牌号
        dst_prefix = osp.join(plate_dir, image_name.replace('.jpg', ''))
        split_image(src_path, bboxes, keypoints, flags, dst_prefix)

        # 创建标签文件
        label_path = osp.join(label_dir, image_name.replace('.jpg', '.txt'))
        create_label_file(bboxes, keypoints, flags, img_size, label_path)

    split_data(input_dir, dataset_dir)


if __name__ == "__main__":
    main()

