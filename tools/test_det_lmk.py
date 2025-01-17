import os
import os.path as osp

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.experimental import attempt_load
from utils.general import non_max_suppression_face
from utils.torch_utils import select_device


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

        results.append((bboxes.cpu().numpy(), keypoints.cpu().numpy(), scores.cpu().numpy()))

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
    for bbox, four_points, score in zip(_bboxes, _keypoints, scores):
        cv2.rectangle(img, bboxes[0:2], bbox[2:4], (0, 255, 0), thickness=2)

        for point, color in zip(four_points, colors):
            cv2.circle(img, point, radius, color, -1)

        cv2.putText(img, f"{score.item():.2f}",
                    (bbox[0], bbox[1] - 4), 0, 1, (0, 0, 255), 2)

    cv2.imwrite(save_path, img)


def save_results(img, keypoints, scores, flags, dst_prefix):
    h, w, _ = img.shape

    if not flags:
        flags = [True] * len(keypoints)

    for i, (group_points, score, flag) in enumerate(zip(keypoints, scores, flags)):
        group_points = np.array(group_points).round().astype(np.int64)

        # crop plate
        l = np.clip(group_points[:, 0].min(), 0, w)
        t = np.clip(group_points[:, 1].min(), 0, h)
        r = np.clip(group_points[:, 0].max(), 0, w)
        b = np.clip(group_points[:, 1].max(), 0, h)
        plate_img = img[t:b, l:r, :]
        
        group_points -= np.array([[l, t]])
        aligned_img = plate_align(plate_img, group_points)
        cv2.imwrite(f"{dst_prefix}-plate{i}.jpg", aligned_img)


def plate_align(img, landmark):
    global SPLIT_CONFIG
    landmark = landmark.astype(np.float32)
    pattern = np.array(SPLIT_CONFIG['aligin_pattern'], np.float32)
    M = cv2.getPerspectiveTransform(landmark[:4, :], pattern[:4, :])
    return cv2.warpPerspective(img, M, SPLIT_CONFIG['aligin_dsize'])


def split_image(img, bboxes, keypoints, scores, flags, dst_prefix):
    h, w, _ = img.shape

    if not flags:
        flags = [True] * len(bboxes)

    for i, (bbox, group_points, score, flag) in enumerate(zip(bboxes, keypoints, scores, flags)):
        bbox = np.array(bbox).round().astype(np.int64).reshape(2, 2)
        group_points = np.array(group_points).round().astype(np.int64)
        _dst_prefix = dst_prefix if flag else dst_prefix.replace('good', 'bad')

        # crop plate
        all_points = np.vstack([bbox, group_points])
        l = np.clip(all_points[:, 0].min() - 10, 0, w)
        t = np.clip(all_points[:, 1].min() - 10, 0, h)
        r = np.clip(all_points[:, 0].max() + 10, 0, w)
        b = np.clip(all_points[:, 1].max() + 10, 0, h)
        plate_img = img[t:b, l:r, :]

        # draw bbox, keypoints and score
        thickness = max(2, round(((r-l+b-t) / 200).item()))
        radius = max(4, round(((r-l+b-t) / 200).item()))
        _plate_img = plate_img.copy()
        _bbox = bbox - np.array([[l, t]])
        cv2.rectangle(_plate_img, _bbox[0], _bbox[1], (0, 255, 0), thickness)
        _group_points = group_points - np.array([[l, t]])
        colors = [(255, 0, 110), (255, 0, 238), (18, 0, 255), (0, 110, 255)]
        for j, point in enumerate(_group_points):
            cv2.circle(_plate_img, point, radius, colors[j], -1)
        text = f"{score.item():.2f}"
        text_size = cv2.getTextSize(text, 0, 1, 2)
        cv2.putText(_plate_img, text, 
            (_bbox[0, 0] + 5, _bbox[0, 1] + 5 + text_size[0][1]), 0, 1, (0, 0, 255), 2)

        cv2.imwrite(f"{_dst_prefix}-plate{i}.jpg", _plate_img)

        # align plate
        aligned_img = plate_align(plate_img, _group_points)
        _aligned_img = aligned_img.copy()
        cv2.line(_aligned_img, (0, 42), (200, 42), (0, 255, 0), 2)
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


class TorchInferencer:
    def __init__(self, model_path):
        self.device = torch.device('cpu')
        self.model = attempt_load(model_path, map_location=self.device)

    def infer(self, input_data):
        input_data = torch.from_numpy(input_data).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.no_grad():
            output_data, _ = self.model(input_data, augment=False)
        return output_data


class ONNXInferencer:
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

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
        self.interpreter.set_tensor(self.input_name, input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_name)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="path to model")
    parser.add_argument('--img-size', type=int, nargs=2, default=[640, 384], help="inference size")
    parser.add_argument('--input-dir', type=str, help="path to input images")
    parser.add_argument('--output-dir', type=str, help="path to save results")
    parser.add_argument('--conf_thres', type=float, default=0.5, help="object confidence threshold")
    parser.add_argument('--iou_thres', type=float, default=0.6, help="IOU threshold for NMS")
    args = parser.parse_args()

    if args.model_path.endswith(('.pt', '.pth')):
        inferencer = TorchInferencer(args.model_path)
        channel_first = True
    elif args.model_path.endswith('.onnx'):
        inferencer = ONNXInferencer(args.model_path)
        channel_first = True
    elif args.model_path.endswith('.tflite'):
        inferencer = TFLiteInferencer(args.model_path)
        channel_first = False
    else:
        raise RuntimeError('Unkown model type:', args.model_path.rsplit('.', 1)[-1])

    os.makedirs(args.output_dir, exist_ok=True)

    for i, img_name in enumerate(tqdm(os.listdir(args.input_dir))):
        img = cv2.imread(osp.join(args.input_dir, img_name))
        
        input_data, info = preprocess(img, dsize=args.img_size, channel_first=channel_first)
        output_data = inferencer.infer(input_data)
        bboxes, keypoints, scores = postprocess(
            output_data, info, args.conf_thres, args.iou_thres)[0]
        
        save_path = osp.join(args.output_dir, img_name)
        # draw_results(img, bboxes, keypoints, scores, save_path)

        dst_prefix = osp.join(args.output_dir, img_name.replace('.jpg', ''))
        # split_image(img, bboxes, keypoints, scores, None, dst_prefix)
        save_results( img, keypoints, scores, None, dst_prefix)


if __name__ == "__main__":
    main()
