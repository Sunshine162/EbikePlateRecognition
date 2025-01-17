from PIL import Image, ImageOps

from cnocr import CnOcr
import cv2
import numpy as np
import onnxruntime as ort



def preprocess(src_path):
    img = Image.open(src_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    img = np.array(img)
    h, w, c = img.shape
    imgC, imgH, imgW = 3, 32, 320
    imgW = int(imgH * w / h)
    img = cv2.resize(img, (imgW, imgH))
    img = img.astype('float32')
    img = img.transpose((2, 0, 1)) / 255
    img -= 0.5
    img /= 0.5
    img = img[None, ...]
    return img


def postprocess(probs, vocab_path, is_remove_duplicate=True):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_list = f.read().splitlines()
    vocab_list.insert(0, 'blank')

    probs = probs.squeeze(axis=0)
    pred_chars, pred_scores = [], []
    max_indices = probs.argmax(axis=-1)
    for i, idx in enumerate(max_indices):
        if idx == 0:  # continue ignore charactor
            continue
        if (is_remove_duplicate and idx > 0 and idx == max_indices[i - 1]):
            continue

        pred_chars.append(vocab_list[idx])
        pred_scores.append(probs[i, idx])

    pred_text = ''.join(pred_chars)
    pred_score = np.array(pred_scores).mean()
    print(pred_text, pred_score, pred_scores)


def main():
    src_path = "2nd-000069-plate0-p1.jpg"
    ocr = CnOcr(rec_model_name="ch_ppocr_mobile_v2.0")
    out = ocr.ocr_for_single_line(src_path)
    print(out)

    # onnx_path = "ch_ppocr_mobile_v2.0_rec_infer.onnx"
    # vocab_path = "ppocr_keys_v1.txt"
    onnx_path = "ch_ppocr_mobile_v2.0_rec_infer_slim.onnx"
    vocab_path = "city_chars.txt"
    input_data = preprocess(src_path)
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output_data = sess.run(None, {input_name: input_data})[0]
    print(output_data.shape)
    postprocess(output_data, vocab_path)


if __name__ == "__main__":
    main()
