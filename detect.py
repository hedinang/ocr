import cv2
import torch
from numpy import random
import numpy as np
import math
import torchvision


class Detect:
    def __init__(self, device):
        super(Detect, self).__init__()
        self.device = torch.device(device)
        self.model = torch.load('1.pth')
        self.model.to(self.device)
        self.half = False
        if device == 'cuda':
            self.model.half()
            self.half = True
        self.stride = int(self.model.stride.max())
        self.conf_thres, self.iou_thres = 0.25, 0.45

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        coords[:, 0].clamp_(0, img0_shape[1])
        coords[:, 1].clamp_(0, img0_shape[0])
        coords[:, 2].clamp_(0, img0_shape[1])
        coords[:, 3].clamp_(0, img0_shape[0])
        return coords

    def xywh2xyxy(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def box_iou(self, box1, box2):
        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])
        area1 = box_area(box1.T)
        area2 = box_area(box2.T)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
                 torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                            labels=()):
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        max_wh = 4096
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        output = [torch.zeros((0, 6), device=prediction.device)
                  ] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            box = self.xywh2xyxy(x[:, :4])
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat(
                    (box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[
                    conf.view(-1) > conf_thres]
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            merge = True
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float(
                ) / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            output[xi] = x[i]
        return output

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), stride=32):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  # wh padding
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img

    def preprocess(self, path):
        origin = cv2.imread(path)  # BGR
        img = self.letterbox(origin)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img, origin

    def draw(self, area):
        color = (0, 0, 150)
        x1, y1, x2, y2 = int(area[0]), int(area[1]), int(area[2]), int(area[3])
        # cv2.rectangle(img, (x1, y1), (x2, y2), color,
        #               thickness=1, lineType=cv2.LINE_AA)
        # score = area[4].cpu().detach().numpy().item()
        # score = round(score, 2)
        # text = '{} {}'.format(area[5], score, decimals=2)
        # cv2.putText(img, text, (x1, y1-2), 0, 1,
        #             [150, 0, 0], thickness=2, lineType=cv2.LINE_AA)

        return [x1, y1, x2, y2]

    def __call__(self, img_path):
        img, origin = self.preprocess(img_path)
        # rectangle = origin.copy()
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        img = img.unsqueeze(0)
        pred = self.model(img)
        det = self.non_max_suppression(
            pred, self.conf_thres, self.iou_thres)[0]
        send = None
        number = None
        header = None
        date = None
        stamp = None
        quote = None
        if len(det):
            det[:, :4] = self.scale_coords(
                img.shape[2:], det[:, :4], origin.shape).round()
            for x1, y1, x2, y2, conf, cls in det:
                if cls == 0:
                    if send == None or send[4] < conf:
                        send = [x1, y1, x2, y2, conf, 'send']
                elif cls == 2:
                    if date == None or date[4] < conf:
                        date = [x1, y1, x2, y2, conf, 'date']
                elif cls == 3:
                    if header == None or header[4] < conf:
                        header = [x1, y1, x2, y2, conf, 'header']
                elif cls == 4:
                    if number == None or number[4] < conf:
                        number = [x1, y1, x2, y2, conf, 'number']
                elif cls == 5:
                    if quote == None or quote[4] < conf:
                        quote = [x1, y1, x2, y2, conf, 'quote']
                elif cls == 6:
                    if stamp == None or stamp[4] < conf:
                        stamp = [x1, y1, x2, y2, conf, 'stamp']
            if stamp != None:
                return None
            if send != None:
                send = self.draw(send)
                send.append(origin[send[1]:send[3], send[0]:send[2], :])
                send.append('send')
            if number != None:
                number = self.draw(number)
                number.append(
                    origin[number[1]:number[3], number[0]:number[2], :])
                number.append('number')
            if header != None:
                header = self.draw(header)
                header.append(
                    origin[header[1]:header[3], header[0]:header[2], :])
                header.append('header')
            if date != None:
                date = self.draw(date)
                date.append(origin[date[1]:date[3], date[0]:date[2], :])
                date.append('date')
            if quote != None:
                quote = self.draw(quote)
                quote.append(
                    origin[quote[1]:quote[3], quote[0]:quote[2], :])
                quote.append('quote')
        return [send, number, header, date, stamp, quote]
