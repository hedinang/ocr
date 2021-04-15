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
        results = {
            'send': [],
            'number': [],
            'header': [],
            'date': [],
            'stamp': [],
            'quote': [],
        }
        if queue != None:
            for e in queue:
                if e != None:
                    sub_ocr = self.ocr(e[4], e[5])
                    if e[5] != 'header' or len(sub_ocr) == 1:
                        result = sub_ocr
                    else:
                        result = [sub_ocr[0]]
                        results['quote'] += sub_ocr[1:]
                    results[e[5]] = result
        else:
            results['stamp'] = True
        return results
