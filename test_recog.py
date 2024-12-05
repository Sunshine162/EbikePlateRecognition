import os
import os.path as osp

import cv2
import numpy as np
import torch
from tqdm import tqdm


IMAGE_EXTENSIONS = ['jpeg', 'jpg', 'png', 'bmp', 'gif']

CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 
    'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
    'W', 'X', 'Y', 'Z', '东', '中', '云', '佛', '关', '名', '圳', '头', '尾', '山', 
    '州', '广', '庆', '惠', '揭', '梅', '汕', '江', '河', '浮', '海', '深', '清', 
    '湛', '源', '潮', '珠', '肇', '茂', '莞', '远', '内', '阳', '韶', '-'
]


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
        img = img.transpose(img, (2, 0, 1))
    img = img[None, ...]

    return img


def postprocess(output_data):
    results = []

    for preb in output_data:
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))

        no_repeat_blank_label = list()
        code = ""
        scores = []
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
            scores.append(preb[pre_c, 0])
            code += CHARS[pre_c]

        for i, c in enumerate(preb_label):
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            scores.append(preb[c, i])
            code += CHARS[c]
            pre_c = c

        results.append((code, scores))

    return results


def get_inputs(inputs):
    if isinstance(inputs, list):
        return [(x, None) for x in inputs if if osp.isfile(x)]

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


class ONNXInferencer:
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(args.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs]

    def infer(self, input_data):
        return self.session.run(self.output_names, {self.input_name: input_data})[0]


class TFLiteInferencer:
    def __init__(self, model_path):
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        self.interpreter = interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        self.input_name = input_details[0]['index']
        self.output_name = output_details[0]['index']

    def infer(self, input_data):
        self.interpreter.set_tensor(input_name, input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(output_data)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="path to model")
    parser.add_argument('--img-size', type=int, nargs=2, default=[94, 24], help="inference size")
    parser.add_argument('--inputs', type=str, help="path to input images")
    parser.add_argument('--threshold', type=float, default=0.5, help="object confidence threshold")
    parser.add_argument('--align', action='store_true', help="align plate")
    parser.add_argument('--split', action='store_true', help="split city name and plate code")
    args = parser.parse_args()

    if args.model.endswith('.onnx'):
        inferencer = ONNXInferencer(args.model)
        channel_first = True
    elif args.model.endswith('.tflite'):
        inferencer = TFLiteInferencer(args.model)
        channel_first = False
    else:
        raise RuntimeError('Unkown model type:', args.model.rsplit('.', 1)[-1])

    inputs = get_inputs(args.inputs)
    cnt_total, cnt_right, print_acc = 0, 0, False
    for i, (img_path, label) in enumerate(inputs):
        img = cv2.imread(img_path)
        input_data = preprocess(img, img_size=args.img_size, channel_first=channel_first)
        output_data = inferencer.infer(input_data)
        
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
