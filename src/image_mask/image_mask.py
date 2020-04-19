import numpy as np
from tflite_runtime import interpreter as tflite


class ImageMask():
    LABEL_PERSON = 15
    BACKGROUND = 0
    # * Label names: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    # * 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    # * 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'

    def __init__(self, threshold=17.0):
        self.threshold = 17.0
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def load_interpreter(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def getMask(self, image):
        if self.interpreter is None:
            return None
        result = self._inference(image)
        person_result = result[0, :, :, self.LABEL_PERSON]
        backgroun_score = result[0, :, :, self.BACKGROUND]
        mask = (person_result > backgroun_score).astype(np.uint8) * 255
        return mask

    def _inference(self, image):
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_details[0]['index'])
        return result