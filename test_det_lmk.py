import os
import os.path.osp

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.experimental import attempt_load
from utils.general import non_max_suppression_face
from utils.torch_utils import select_device


def preprocess(img, dsize=(640, 384), channel_first=False, out_int8=False):
    dst_w, dst_h = dsize
    h0, w0, _ = img.shape

    # rescale image
    r = min(dst_w / w0, dst_h / h0)
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    w1, h1 = int(w0 * r), int(h0 * r)
    img = cv2.resize(img, (w1, h1), interpolation=interp)

    # pad image
    pl = (dst_w - w1) // 2
    pr = (dst_w - w1) - pl
    pt = (dst_h - h1) // 2
    pb = (dst_h - h1) - pt
    img = cv2.copyMakeBorder(
        img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR to RGB
    img = img[:, :, ::-1]

    # HWC to CHW
    if channel_first:
        img = img.transpose(2, 0, 1)

    img = np.ascontiguousarray(img)

    if out_int8:
        return img, {'ratio': r, 'pad': (pl, pt, pr, pb)}

    img = img.astype(np.float32)
    img /= 255.0
    return img, {'ratio': r, 'pad': (pl, pt, pr, pb)}


def postprocess(output_data, img_info, conf_thres, iou_thres):
    ratio = img_info['ratio']
    pl, pt, _, _ = img_info['pad']

    if isinstance(output_data, np.ndarray):
        output_data = torch.from_numpy(output_data)

    output_data = non_max_suppression_face(
        output_data, conf_thres=conf_thres, iou_thres=iou_thres)

    results = []
    for x in output_data:
        bboxes = x[:, :4]
        scores = x[:, 4]
        keypoints = x[:, 5:13].reshape(-1, 4, 2)

        bboxes -= np.array([pl, pt, pl, pt])
        bboxes /= ratio

        keypoints -= np.array([pl, pt])
        keypoints /= ratio

        results.append((bboxes.cpu().numpy(), keypoints.cpu().numpy(), scores.cpu.numpy()))

    return results


def draw_results(img, bboxes, keypoints, scores, save_path):
    height, width, _ = img.shape
    radius = max(7, int(max(width, height) / 1000))
    colors = [(0, 255, 255), (0, 75, 255), (255, 255, 0), (255, 0, 255)]

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    _bboxes = bboxes.round().astype(np.int64)
    _keypoints = keypoints.round().astype(np.int64)
    for bbox, five_points, score in zip(_bboxes, _keypoints, scores):
        cv2.rectangle(img, bboxes[0:2], bbox[2:4], (0, 255, 0), thickness=2)

        for point, color in zip(five_points, colors):
            cv2.circle(img, point, radius, color, -1)

        cv2.putText(img, f"{score.item()}:.2f",
                    (bbox[[0], bbox[1] - 4]), 0, 1, (0, 0, 255), 2)

    cv2.imwrite(save_path, img)


class TorchInferencer:
    def __init__(self, model_path):
        device = torch.device('cpu')
        self.model = attempt_load(model_path, map_location=device)

    def infer(self, input_data):
        input_data = torch.from_numpy(input_data).unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            torch.from_numpy(input_data).unsqueeze(0).to(device, non_blocking=True)
            output_data, _ = self.model(input_data, augment=False)


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
        interpreter.invoke()
        return interpreter.get_tensor(output_data)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="path to model")
    parser.add_argument('--img-size', type=int, nargs=2, default=[640, 384], help="inference size")
    parser.add_argument('--input-dir', type=str, help="path to input images")
    parser.add_argument('--output-dir', type=str, help="path to save results")
    parser.add_argument('--conf_thres', type=float, default=0.25, help="object confidence threshold")
    parser.add_argument('--iou_thres', type=float, default=0.6, help="IOU threshold for NMS")
    args = parser.parse_args()

    if args.model_path.endswith(('.pt', '.pth')):
        inferencer = TorchInferencer(model_path)
        channel_first = True
    elif args.model_path.endswith('.onnx'):
        inferencer = ONNXInferencer(model_path)
        channel_first = True
    elif args.model_path.endswith('.tflite'):
        inferencer = TFLiteInferencer(model_path)
        channel_first = False
    else:
        raise RuntimeError('Unkown model type:', args.model_path.rsplit('.', 1)[-1])

    os.makedirs(args.output_dir, exist_ok=True)

    for i, img_name in enumerate(tqdm(os.listdir(args.input_dir))):
        img = cv2.imread(osp.join(args.input_dir, img_name))
        input_data, info = preprocess(img, dsize=args.img_size, channel_first=channel_first)
        output_data = inferencer.infer(input_data)
        bboxes, keypoints, scores = postprocess(
            output_data, info, args.conf_thres, args.iou_thres)
        save_path = osp.join(args.output_dir, img_name)
        draw_results(img, bboxes, keypoints, scores, save_path)
