import cv2
import numpy as np
import torch
from utils import xywh2xyxy, py_nms


class BaseDetectorWithLandmark:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_path = cfg.model_path
        self.input_size = self.cfg.input_size
        self.norm_mean = np.array(self.cfg.norm.mean, np.float32)
        self.norm_std = np.array(self.cfg.norm.std, np.float32)

        channel_first = cfg.get('channel_first', None)
        if channel_first is None:
            channel_first = False if self.model_path.endswith('.tflite') else True
        self.channel_first = channel_first

        self.conf_thres = cfg.conf_threshold
        self.iou_thres = cfg.nms_threshold
        self.min_plate = cfg.min_plate
        self.max_outputs = cfg.max_outputs

    def preprocess(self, img):
        dst_w, dst_h = self.input_size
        h0, w0, _ = img.shape

        # resize image
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

        # normalize
        img = img.astype(np.float32)
        img -= self.norm_mean
        img /= self.norm_std

        # HWC to CHW
        if self.channel_first:
            img = img.transpose(2, 0, 1)

        img = np.ascontiguousarray(img)

        img_info = {'ratio': r, 'pad_left': pl, 'pad_top': pt, 'src_width': w0, 
                    'src_height': h0}
        return img[None, ...], img_info

    def postprocess(self, output_data, img_info):
        ratio = img_info['ratio']
        pl, pt = img_info['pad_left'], img_info['pad_top']
        w, h = img_info['src_width'], img_info['src_height']

        output_data = xywh2xyxy(output_data)

        results = []
        for x in output_data:

            x = py_nms(x, self.iou_thres, self.conf_thres)

            bboxes = x[:, :4]
            scores = x[:, 4] * x[:, -1]
            keypoints = x[:, 5:13].reshape(-1, 4, 2)

            bboxes -= np.array([pl, pt, pl, pt])
            bboxes /= ratio
            bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], 0, w - 1)
            bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], 0, h - 1)

            keypoints -= np.array([pl, pt])
            keypoints /= ratio
            keypoints[..., 0] = np.clip(keypoints[..., 0], 0, w - 1)
            keypoints[..., 1] = np.clip(keypoints[..., 1], 0, h - 1)

            results.append((bboxes, keypoints, scores))

        return results

    def predict(self, img):
        """predict one image by end2end"""

        input_data, meta = self.preprocess(img)
        output_data = self.infer(input_data)
        bboxes_kpts_scores = self.postprocess(output_data, meta)[0]
        return bboxes_kpts_scores

    def infer(self, input_data):
        """model inference"""
        
        raise NotImplementedError()


class YOLOv5ONNXDetectorWithLandmark(BaseDetectorWithLandmark):
    def __init__(self, cfg):
        super(YOLOv5ONNXDetectorWithLandmark, self).__init__(cfg)

        import onnxruntime as ort
        self.session = ort.InferenceSession(cfg.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, input_data):
        """model inference"""
        return self.session.run(self.output_names, {self.input_name: input_data})[0]


class YOLOv5TFLiteDetectorWithLandmark(BaseDetectorWithLandmark):
    def __init__(self, cfg):
        super(YOLOv5TFLiteDetectorWithLandmark, self).__init__(cfg)

        import tensorflow as tf
        interpreter = tf.lite.Interpreter(cfg.model_path)
        interpreter.allocate_tensors()
        self.interpreter = interpreter
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        assert len(self.input_details) == 1
        self.input_name = self.input_details[0]['index']
        self.output_name = self.output_details[0]['index']

    def infer(self, input_data):
        """model inference"""
        self.interpreter.set_tensor(self.input_name, input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_name)
        return output_data
    