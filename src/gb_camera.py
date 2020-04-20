import argparse

import numpy as np
import cv2

from src.image_mask import ImageMask

blank = np.zeros((257, 257, 3))
blank += [0, 255, 0][::-1]  # RGBで青指定
blank = blank.astype(np.float32)


parser = argparse.ArgumentParser(description='Learning Racer command.')
parser.add_argument('-m','--model', help='Path to the model path.',default='../deeplabv3_257_mv_gpu.tflite', type=str)
parser.add_argument('-c', '--camera', help="Camera device id.", default=0, type=int)
parser.add_argument('--width', help="Define camera capture width size.", default=640, type=int)
parser.add_argument('--height', help="Define camera capture height size.", default=480, type=int)
parser.add_argument('-fps', '--fps', help="Define FPS.", default=30, type=int)

def img_to_tensor(image):
    image = cv2.resize(image, (257, 257))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    data = np.asarray(image / 255, dtype=np.float32)
    return data


def composite_mask(src, mask, width, height):
    src = cv2.resize(src, (width, height))
    src = np.asarray(src / 255, dtype=np.float32)
    mask = cv2.resize(mask, (width, height))
    background = cv2.resize(blank, (width, height))
    masked = cv2.merge((mask, mask, mask))
    fg = cv2.bitwise_and(src, src, mask=mask)
    masked = cv2.bitwise_not(masked)
    bg = cv2.bitwise_and(background, masked.astype(np.float32))
    result = cv2.bitwise_or(fg, bg)
    return result


def main():

    args = parser.parse_args()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # カメラ画像の横幅を1280に設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # カメラ画像の縦幅を720に設定

    image_mask = ImageMask()
    image_mask.load_interpreter(args.model)

    while True:
        ret, frame = cap.read()
        data = img_to_tensor(frame)
        mask = image_mask.getMask([data])
        masked_frame = composite_mask(frame, mask, 640, 320)
        cv2.imshow('green_back_screen', masked_frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
