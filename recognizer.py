import os

import cv2
import numpy as np
from collections import Counter


save_prefix = {"text": None}
debug = os.environ.get(["DEBUG"], False)
debug = False if debug in [0, '0', 'False', 'false'] else debug


class BasePlateOCR:
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

        self.align_pattern = np.array(cfg.align.pattern, np.float32)
        self.align_dsize = cfg.align.dsize
        self.p1_box = cfg.split.p1_box
        self.p2_box = cfg.split.p2_box

        self.code_chars = set(cfg.code_chars)
        self.city_chars = set(cfg.city_chars)
        self.vocab = ['blank', *cfg.code_chars, *cfg.city_chars, ' ']
        self.city_names = set(cfg.city_names)

        start = 1
        self.code_char_indices = [0] + list(range(start, len(cfg.code_chars) + start)) + [len(self.vocab) - 1]
        self.code_vocab = ['blank', *cfg.code_chars, ' ']
        start = 1 + len(cfg.code_chars)
        self.city_char_indices = [0] + list(range(start, len(cfg.city_chars) + start)) + [len(self.vocab) - 1]
        self.city_vocab = ['blank', *cfg.city_chars, ' ']

        char_counter = Counter(list("".join(self.city_names)))
        self.city_map = {}
        for city_name in self.city_names:
            for i, char in enumerate(city_name):
                if char_counter[char] == 1:
                    self.city_map[char] = (city_name, i)

        self.city_char_thres = cfg.city_char_threshold
        self.code_char_thres = cfg.code_char_threshold

    def crop_plate(self, img, keypoints):
        h, w, _ = img.shape

        l = np.clip(keypoints[:, 0].min() - 10, 0, w).astype(np.int64)
        t = np.clip(keypoints[:, 1].min() - 10, 0, h).astype(np.int64)
        r = np.clip(keypoints[:, 0].max() + 10, 0, w).astype(np.int64)
        b = np.clip(keypoints[:, 1].max() + 10, 0, h).astype(np.int64)
        if debug:
            with open(save_prefix['text'] + '-box-points.txt', 'w', encoding='utf-8') as f:
                f.write(f"keypoints(in original image): {str(keypoints.tolist())}\n")
                f.write(f"crop box:  l={l.item()} t={t.item()} r={r.item()} b={b.item()}\n")

        plate_img = img[t:b, l:r, :]
        keypoints -= np.array([[l, t]])
        if debug:
            with open(save_prefix['text'] + '-box-points.txt', 'a', encoding='utf-8') as f:
                f.write(f"keypoints(in croped image): {str(keypoints.tolist())}\n")
        return plate_img, keypoints

    def align_plate(self, plate_img, keypoints):
        keypoints = keypoints.astype(np.float32)
        M = cv2.getPerspectiveTransform(keypoints, self.align_pattern)
        return cv2.warpPerspective(plate_img, M, self.align_dsize)

    def split_plate(self, aligned_img):
        p1b, p2b = self.p1_box, self.p2_box
        p1_img = aligned_img[p1b.top : p1b.bottom, p1b.left : p1b.right, :]
        p2_img = aligned_img[p2b.top : p2b.bottom, p2b.left : p2b.right, :]
        return p1_img, p2_img

    def preprocess(self, img, keypoints):
        plate_img, keypoints = self.crop_plate(img, keypoints)
        aligned_img = self.align_plate(plate_img, keypoints)
        p1_img, p2_img = self.split_plate(aligned_img)
        if debug:
            cv2.imwrite(save_prefix['text'] + '-A.jpg', plate_img)
            cv2.imwrite(save_prefix['text'] + '-B.jpg', aligned_img)
            cv2.imwrite(save_prefix['text'] + '-p1.jpg', p1_img)
            cv2.imwrite(save_prefix['text'] + '-p2.jpg', p2_img)
        
        inW, inH = self.input_size

        input_data = []
        for p in [p1_img, p2_img]:
            # normalize
            p = cv2.resize(p, self.input_size)
            p = p.astype(np.float32)
            p -= self.norm_mean
            p /= self.norm_std

            # HWC to CHW
            if self.channel_first:
                p = p.transpose(2, 0, 1)

            input_data.append(p[None, ...])

        return np.concatenate(input_data, axis=0)

    def get_city_name(self, probs, is_remove_duplicate=True, guess=True):
        pred_chars, pred_scores = [], []
        probs = probs[:, self.city_char_indices]
        max_indices = probs.argmax(axis=-1)
        for i, idx in enumerate(max_indices):
            if idx == 0 and idx == len(self.vocab) - 1:  # continue ignore charactor
                continue
            if (is_remove_duplicate and idx > 0 and idx == max_indices[i - 1]):
                continue

            score = probs[i, idx].item()
            if score < self.city_char_thres:
                continue
            char = self.city_vocab[idx]
            if char not in self.city_chars:
                continue
            pred_chars.append(char)
            pred_scores.append(score)

        pred_text = ''.join(pred_chars)
        # print("city:", pred_text)

        if len(pred_text) == 0 or pred_text in self.city_names:
            return pred_text, pred_scores
        if not guess:
            return "", []

        guess_names = []
        char_score = None
        char_index = None
        for char, score in zip(pred_text, pred_scores):
            if char in self.city_map:
                city_name, char_index = self.city_map[char]
                guess_names.append(city_name)
                char_score = score
        if len(set(guess_names)) == 1:
            guess_name = guess_names[0]
            pred_scores = [char_score, 0.0] if char_index == 0 else [0.0, char_score]
            return guess_name, pred_scores
        else:
            return "", []

    def get_plate_code(self, probs, is_remove_duplicate=True):
        pred_chars, pred_scores = [], []
        probs = probs[:, self.code_char_indices]
        max_indices = probs.argmax(axis=-1)
        for i, idx in enumerate(max_indices):
            if idx == 0 and idx == len(self.vocab) - 1:  # continue ignore charactor
                continue
            if (is_remove_duplicate and idx > 0 and idx == max_indices[i - 1]):
                continue

            score = probs[i, idx].item()
            if score < self.code_char_thres:
                continue
            char = self.code_vocab[idx]
            if char not in self.code_chars:
                continue
            pred_chars.append(char)
            pred_scores.append(score)

        pred_text = ''.join(pred_chars)
        # print("code:", pred_text)
        return pred_text, pred_scores
 
    def postprocess(self, output_data):
        city_name, name_scores = self.get_city_name(output_data[0, ...])
        plate_code, code_scores = self.get_plate_code(output_data[1, ...])

        pred_scores = name_scores + code_scores
        pred_score = np.array(pred_scores).mean().item() if pred_scores else 0.0

        _city_name = city_name if city_name else "++"
        _plate_code = plate_code if plate_code else "++++++"
        pred_text = _city_name + _plate_code

        if debug:
            os.rename(save_prefix['text']+'-p1.jpg', save_prefix['text']+f'-p1-{_city_name}.jpg')
            os.rename(save_prefix['text']+'-p2.jpg', save_prefix['text']+f'-p2-{_plate_code}.jpg')

        return {"text": pred_text, "text_score": pred_score, "char_scores": pred_scores,
                "city": city_name, "code": plate_code}

    def predict(self, img, keypoints):
        """predict one image by end2end"""

        input_data = self.preprocess(img, keypoints)
        output_data = self.infer(input_data)
        result = self.postprocess(output_data)
        return result

    def infer(self, input_data):
        """model inference"""
        
        raise NotImplementedError()


class PPMobileONNXPlateOCR(BasePlateOCR):
    def __init__(self, cfg):
        super(PPMobileONNXPlateOCR, self).__init__(cfg)

        import onnxruntime as ort
        self.session = ort.InferenceSession(cfg.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, input_data):
        """model inference"""
        return self.session.run(self.output_names, {self.input_name: input_data})[0]

class PPMobileTFLitePlateOCR(BasePlateOCR):
    def __init__(self, cfg):
        super(PPMobileTFLitePlateOCR, self).__init__(cfg)

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
    
