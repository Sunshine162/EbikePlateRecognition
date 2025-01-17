import argparse
import os
import os.path as osp

from PIL import ImageFont, ImageDraw, Image
import cv2
from munch import Munch
import numpy as np
import torch
from tqdm import tqdm
from yaml import safe_load

from detector_with_landmark import YOLOv5ONNXDetectorWithLandmark, YOLOv5TFLiteDetectorWithLandmark
from recognizer import PPMobileONNXPlateOCR, PPMobileTFLitePlateOCR
from recognizer import save_prefix

IMAGE_EXTENSIONS = set(['jpeg', 'jpg', 'png', 'bmp', 'gif'])

debug = os.environ.get("DEBUG", False)
debug = False if debug in [0, '0', 'False', 'false'] else debug
if debug:
    os.makedirs('./debug', exist_ok=True)


def load_config_and_models(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = Munch.fromDict(safe_load(f))

    detector = YOLOv5ONNXDetectorWithLandmark(cfg.detector)
    recognizer = PPMobileONNXPlateOCR(cfg.recognizer)
    return cfg, detector, recognizer


def draw_texts(texts, img, det_box):
    font_name = 'Deng.ttf'
    font_size = 50
    font = ImageFont.truetype(font_name, font_size)

    _det_box = det_box.copy()
    box_width = det_box[2] - det_box[0]

    for text in texts:
        (w, h), (offset_x, offset_y) = font.font.getsize(text)
        x_border = w * 0.05
        y_border = h * 0.05
        w = w + round(2 * x_border)
        h = h + round(2 * y_border)

        text_image = np.array([[[0, 0, 255]]], dtype=np.uint8).repeat(h, axis=0).repeat(w, axis=1)
        pil_img = Image.fromarray(text_image)
        draw = ImageDraw.Draw(pil_img)
        draw.text((round(x_border), round(y_border)), text=text, font=font, fill=(0, 0, 0, 0))
        text_image = np.array(pil_img)

        new_width= box_width + 1
        new_height = round(new_width * h / w)
        text_image = cv2.resize(text_image, (new_width, new_height))

        x_start = _det_box[0]
        if new_height < _det_box[1]:
            y_start = _det_box[1] - new_height
            _det_box[1] = y_start
        else:
            y_start = _det_box[3]
            _det_box += new_height
        img[y_start:y_start+new_height, x_start:x_start+new_width, :] = text_image

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="config.yaml", help='config path')
    parser.add_argument('-i', '--input_dir', type=str, help='path to input directory')
    parser.add_argument('-o', '--output_dir', type=str, help='path to output directory')
    args = parser.parse_args()

    cfg, detector, recognizer = load_config_and_models(args.config)
    # colors = [(0, 255, 255), (0, 75, 255), (255, 255, 0), (255, 0, 255)]
    colors = [(0, 255, 0)] * 4
    os.makedirs(args.output_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(args.input_dir)):
        img_stem = img_name.split('.')[0]

        ext = img_name.rsplit('.', 1)[1]
        if ext not in IMAGE_EXTENSIONS:
            continue

        src = cv2.imread(osp.join(args.input_dir, img_name))
        dst = src.copy()
        radius = max(7, int(max(dst.shape[:2]) / 1000))
        
        bboxes, keypoints, scores = detector.predict(src)
        for bbox, pt4, score in zip(bboxes, keypoints, scores):

            _bbox = bbox.round().astype(np.int64)
            _pt4 = pt4.round().astype(np.int64)
            cv2.rectangle(dst, _bbox[:2], _bbox[2:], (0, 0, 255), 2)
            for point, color in zip(_pt4, colors):
                cv2.circle(dst, point, radius, color, -1)

            result = recognizer.predict(src, pt4)

            # dst = draw_texts(result['text'], dst, _bbox)
            # score_text = f"{score.item():.2f} | {result['text_score'].item():.2f}"
            # text_size = cv2.getTextSize(score_text, 0, 1, 2)
            # cv2.putText(dst, f"{score.item():.2f}", 
            #     (_bbox[0] + 5, _bbox[1] + 5 + text_size[0][1]), 0, 1, (0, 0, 255), 2)

            # score_text = f"{score.item():.2f} | {result['text_score'].item():.2f}"
            # show_text = result['text'] + ' | ' + score_text
            # dst = draw_texts(show_text, dst, _bbox)

            score_text = f"{score.item():.2f} | {result['text_score']:.2f}     "
            show_texts = [score_text, result['text']]
            draw_texts(show_texts, dst, _bbox)

        cv2.imwrite(osp.join(args.output_dir, img_name), dst)


if __name__ == "__main__":
    main()
