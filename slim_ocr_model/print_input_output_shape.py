import cv2
import numpy as np
import onnxruntime as ort


def preprocess(src_path, imgW, imgH):
    img = cv2.imread(src_path)
    img = cv2.resize(img, (imgW, imgH))
    img = img.astype('float32')
    img = img.transpose((2, 0, 1)) / 255
    img -= 0.5
    img /= 0.5
    img = img[None, ...]
    return img


def main():
    src_path = "2nd-000069-plate0-p1.jpg"
    onnx_path = "ch_ppocr_mobile_v2.0_rec_infer.onnx"
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    for input_length in range(82, 120):
        input_data = preprocess(src_path, input_length, 32)
        output_data = sess.run(None, {input_name: input_data})[0]
        print(f"InputShape={input_data.shape}   <===>   OutputShape={output_data.shape}")


if __name__ == "__main__":
    main()
