from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite
import cv2

def image_load(path):
    image = cv2.imread(path)
    image = cv2.resize(image,(257,257))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = np.asarray(image/255, dtype=np.float32)
    return [data]

# * Label names: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
# * 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
# * 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'

blank = np.zeros((257, 257, 3))
blank += [0,255,0][::-1] #RGBで青指定
blank = blank.astype(np.float32)
class ImageMask():

    LABEL_PERSON = 15
    BACKGROUND = 0
    def __init__(self, threshold=17.0):
        self.threshold = 17.0
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def load_interpreter(self, model_path):
        self.interpreter = tflite.Interpreter(model_path = model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def getMask(self, image):

        if self.interpreter is None:
            return None
        result = self._inference(image)
        person_result = result[0,:,:,self.LABEL_PERSON]
        backgroun_score = result[0,:,:,self.BACKGROUND]
        mask = (person_result > backgroun_score).astype(np.uint8) * 255
        return mask

    def _inference(self, image):
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_details[0]['index'])
        return result


def img_to_tensor(image):
    image = cv2.resize(image,(257, 257))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    data = np.asarray(image/255, dtype=np.float32)
    return data

def composite_mask(src, mask, width, height):
    masked = cv2.merge((mask,mask,mask))
    fg = cv2.bitwise_and(src, src, mask =mask)
    masked = cv2.bitwise_not(masked)
    bg = cv2.bitwise_and(blank, masked.astype(np.float32))
    result = cv2.bitwise_or(fg, bg)
    resized_result = cv2.resize(result,(width,height))
    masked_frame = cv2.cvtColor(resized_result, cv2.COLOR_RGB2BGR)
    return masked_frame


def main():
    black = [0,0,0]
    green = [0,255,0]
    cap = cv2.VideoCapture(0)
    imageMask = ImageMask()
    imageMask.load_interpreter('../deeplabv3_257_mv_gpu.tflite')
    imageMask.threshold = 17
    while True:
        ret, frame = cap.read()
        data = img_to_tensor(frame)
        mask = imageMask.getMask([data])
        masked_frame = composite_mask(data, mask, 640, 320)

        cv2.imshow('green_back_screen', masked_frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()