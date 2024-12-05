import os
import os.path as osp

import cv2
import numpy as np
import torch
from tqdm import tqdm


IMAGE_EXTENSIONS = ['jpeg', 'jpg', 'png', 'bmp', 'gif']

CODE_CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 
    'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
    'W', 'X', 'Y', 'Z', '-'
]
CITY_CHARS = [
    '东', '中', '云', '佛', '关', '名', '圳', '头', '尾', '山', '州', '广', '庆', 
    '惠', '揭', '梅', '汕', '江', '河', '浮', '海', '深', '清', '湛', '源', '潮', 
    '珠', '肇', '茂', '莞', '远', '内', '阳', '韶', '-'
]

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


def plate_align(img):
    h, w, _ = img.shape
    landmark = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)

    global SPLIT_CONFIG
    pattern = np.array(SPLIT_CONFIG['aligin_pattern'], np.float32)

    M = cv2.getPerspectiveTransform(landmark[:4, :], pattern[:4, :])
    return cv2.warpPerspective(img, M, SPLIT_CONFIG['aligin_dsize'])


def split_plate(plate_img, do_align=True):
    if do_align:
        plate_img = plate_align(plate_img)

    # crop city name and plate codes
    global SPLIT_CONFIG
    p1cfg, p2cfg = SPLIT_CONFIG['p1'], SPLIT_CONFIG['p2']
    p1 = plate_img[p1cfg['top']:p1cfg['bottom'], p1cfg['left']:p1cfg['right'], :]
    p2 = plate_img[p2cfg['top']:p2cfg['bottom'], p2cfg['left']:p2cfg['right'], :]
    return p1, p2


def preprocess(img, img_size=(640, 384), channel_first=False):
    dst_w, dst_h = img_size
    h0, w0, _ = img.shape
    r = min(dst_w / w0, dst_h / h0)
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, img_size, interpolation=interp)

    img = img.astype(np.float32)
    img -= 127.5
    img /= 128

    if channel_first:
        img = img.transpose(2, 0, 1)
    img = img[None, ...]

    return img


def postprocess(output_data, chars):
    results = []

    for preb in output_data:
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))

        no_repeat_blank_label = list()
        code = ""
        scores = []
        pre_c = preb_label[0]
        if pre_c != len(chars) - 1:
            no_repeat_blank_label.append(pre_c)
            scores.append(preb[pre_c, 0])
            code += chars[pre_c]

        for i, c in enumerate(preb_label):
            if (pre_c == c) or (c == len(chars) - 1):
                if c == len(chars) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            scores.append(preb[c, i])
            code += chars[c]
            pre_c = c

        results.append((code, scores))

    return results


def ocr_recognize(img, inferencer):
    input_data = preprocess(
        img, img_size=inferencer.img_size, channel_first=inferencer.channel_first)
    output_data = inferencer.infer(input_data)
    code, scores = postprocess(output_data, inferencer.chars)[0]
    return code, scores


def plate_recognize(img, p1_inferencer, p2_inferencer, do_align):
    p1_img, p2_img = split_plate(img, do_align)
    city, city_scores = ocr_recognize(p1_img, p1_inferencer)
    code, code_scores = ocr_recognize(p2_img, p2_inferencer)
    return city, code, city_scores, code_scores, p1_img, p2_img


def get_inputs(inputs):
    if isinstance(inputs, list):
        return [(x, None) for x in inputs if osp.isfile(x)]

    assert isinstance(inputs, str)

    if osp.isdir(inputs):
        return [(osp.join(inputs, x), None) for x in os.listdir(inputs)]

    if osp.isfile(inputs) and inputs.endswith('.txt'):
        image_label_items = []
        with open(inputs, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                image_label_items.append((img_path, label))
        return image_label_items

    if osp.isfile(inputs) and inputs.endswith(tuple(IMAGE_EXTENSIONS)):
        return [(inputs, None)]

    raise RuntimeError("Can not get image files from:", inputs)


def load_models(model_paths, img_size, char_lists=[], split=False):
    global CODE_CHARS
    global CITY_CHARS

    default_chars = CODE_CHARS[:-1] + CITY_CHARS
    if not char_lists:
        char_lists = [default_chars] * len(model_paths)

    inferencers = []
    for model_path, chars in zip(model_paths, char_lists):
        chars = chars if chars else default_chars
        if model_path.endswith('.onnx'):
            inferencers.append(ONNXInferencer(model_path, img_size, chars))
        elif model_path.endswith('.tflite'):
            inferencers.append(TFLiteInferencer(model_path, img_size, chars))
        else:
            raise RuntimeError('Unkown model type:', model_path.rsplit('.', 1)[-1])

    if split and len(inferencers) == 1:
        inferencers = inferencers * 2

    return *inferencers


class ONNXInferencer:
    def __init__(self, model_path, img_size, chars):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.img_size = img_size
        self.channel_first = True
        self.chars = chars

    def infer(self, input_data):
        return self.session.run(self.output_names, {self.input_name: input_data})[0]


class TFLiteInferencer:
    def __init__(self, model_path, img_size, chars):
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        self.interpreter = interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        self.input_name = input_details[0]['index']
        self.output_name = output_details[0]['index']
        self.img_size = img_size
        self.channel_first = False
        self.chars = chars

    def infer(self, input_data):
        self.interpreter.set_tensor(self.input_name, input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_name)
        return output_data.transpose(0, 2, 1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', help="paths to model")
    parser.add_argument('--img-size', type=int, nargs=2, default=[94, 24], help="inference size")
    parser.add_argument('--inputs', type=str, help="path to input images")
    parser.add_argument('--threshold', type=float, default=0.5, help="object confidence threshold")
    parser.add_argument('--align', action='store_true', help="align plate")
    parser.add_argument('--split', action='store_true', help="split city name and plate code")
    parser.add_argument('--char-lists', type=str, nargs='+', help="split city name and plate code")
    parser.add_argument('--outputs', type=str, , 
        default="images_with_prediction", help="path to save recogntion results")
    args = parser.parse_args()

    assert 1 <= len(args.models) <= 2, "quantity of models is invalid!"
    if len(args.models) == 2:
        assert args.split
        split = True 
    else:
        split = args.split

    if split:
        p1_inferencer, p2_inferencer = load_models(args.models, args.img_size, split=split)
    else:
        inferencer = load_models(args.models, args.img_size, split=split)[0]

    inputs = get_inputs(args.inputs)
    os.makedirs(args.outputs, exist_ok=True)

    cnt_total, cnt_right, print_acc = 0, 0, False
    for i, (img_path, label) in enumerate(inputs):
        stem = osp.basename(img_path).rsplit('.', 1)[0]
        img = cv2.imread(img_path)
        if split:
            p1_code, p2_code, p1_scores, p2_scores, p1_img, p2_img = \
                plate_recognize(img, p1_inferencer, p2_inferencer, args.align)
            cv2.imwrite(osp.join(args.outputs, stem + f'-P1-{p1_code}.jpg'), p1_img)
            cv2.imwrite(osp.join(args.outputs, stem + f'-P1-{p2_code}.jpg'), p2_img)
            code = p1_code + p2_code
        else:
            code, scores = ocr_recognize(img, inferencer)
        
        cnt_total += 1
        if label:
            print_acc = True
            flag = 'F'
            if code == label:
                cnt_right += 1
                flag = 'T'
            print(flag, code, img_path)
        else:
            print(code, img_path)

    if print_acc:
        acc = cnt_right / cnt_total
        print("\nAccuracy:", acc)


if __name__ == "__main__":
    main()
