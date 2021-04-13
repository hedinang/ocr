from detect import Detect
from ocr import Ocr
import cv2


class Process:
    def __init__(self, device):
        super(Process, self).__init__()
        self.detect = Detect(device)
        self.ocr = Ocr(device)

    def __call__(self, img_path):
        queue = self.detect(img_path)
        results = {}
        for e in queue:
            if e != None:
                result = self.ocr(e[5])
                results[e[4]] = result
        return results
