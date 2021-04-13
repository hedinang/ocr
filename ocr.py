from craft import Craft
from transformer import Transformer
import torch
import os
import math
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.nn.functional import log_softmax, softmax
from torchvision import models
import numpy as np
import cv2
from beam import Beam


class Vocab:
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3
        self.chars = chars
        self.c2i = {c: i+4 for i, c in enumerate(chars)}
        self.i2c = {i+4: c for i, c in enumerate(chars)}
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]

    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent

    def __len__(self):
        return len(self.c2i) + 4

    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars


class Ocr:
    def __init__(self, device):
        super(Ocr, self).__init__()
        vocab = open('/home/dung/Project/AI/ocr/vocab.txt', 'r').readline()
        self.vocab = Vocab(vocab)
        self.device = torch.device(device)
        self.craft = Craft()
        self.craft.load_state_dict(torch.load(
            '/home/dung/Project/AI/ocr/craft.pth', map_location=device))
        self.craft.to(device)
        self.craft.eval()
        self.transformer = Transformer(len(self.vocab))
        self.transformer.load_state_dict(torch.load(
            '/home/dung/Project/AI/ocr/ocr.pth', map_location=device))
        self.transformer.to(device)
        self.transformer.eval()
        self.image_height = 32
        self.image_min_width = 32
        self.image_max_width = 512
        self.transforms = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406),
                      (0.229, 0.224, 0.225))
        ])

    def resize_aspect_ratio(self, img, square_size=2560):
        height, width, channel = img.shape
        target_size = max(height, width)
        if target_size > square_size:
            target_size = square_size
        ratio = target_size / max(height, width)
        target_h, target_w = int(height * ratio), int(width * ratio)
        img = cv2.resize(img, (target_w, target_h),
                         interpolation=cv2.INTER_LINEAR)
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resize = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resize[:target_h, :target_w, :] = img
        return resize, ratio

    def adjustResultCoordinates(self, boxes, ratio_w, ratio_h, ratio_net=2):
        if len(boxes) > 0:
            boxes = np.array(boxes)
            for i, box in enumerate(boxes):
                if box is not None:
                    boxes[i] *= (ratio_w * ratio_net, ratio_h * ratio_net)
        return boxes

    def getDetBoxes_core(self, textmap, linkmap, text_threshold, link_threshold, low_text):
        linkmap = linkmap.copy()
        textmap = textmap.copy()
        img_h, img_w = textmap.shape
        """ labeling method """
        _, text_score = cv2.threshold(textmap, low_text, 1, 0)
        _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
        text_score_comb = np.clip(text_score + link_score, 0, 1)
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            text_score_comb.astype(np.uint8), connectivity=4)
        det = []
        for k in range(1, nLabels):
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10:
                continue
            if np.max(textmap[labels == k]) < text_threshold:
                continue
            segmap = np.zeros(textmap.shape, dtype=np.uint8)
            segmap[labels == k] = 255
            segmap[np.logical_and(link_score == 1, text_score == 0)] = 0
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            if sx < 0:
                sx = 0
            if sy < 0:
                sy = 0
            if ex >= img_w:
                ex = img_w
            if ey >= img_h:
                ey = img_h
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
            np_contours = np.roll(
                np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)
            w, h = np.linalg.norm(
                box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
                t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
                box = np.array(
                    [[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
            startidx = box.sum(axis=1).argmin()
            box = np.roll(box, 4-startidx, 0)
            box = np.array(box)
            det.append(box)
        polys = [None] * len(det)
        return det, polys

    def get_textbox(self, image, text_threshold, link_threshold, low_text):
        result = []
        resize, target_ratio = self.resize_aspect_ratio(image)
        ratio_h = ratio_w = 1 / target_ratio
        x = Image.fromarray(resize.astype(np.uint8))
        x = self.transforms(x)
        x = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.craft(x)
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()
        boxes, polys = self.getDetBoxes_core(
            score_text, score_link, text_threshold, link_threshold, low_text)
        boxes = self.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = np.array(polys)
        result = []
        for i, poly in enumerate(polys):
            if poly is None:
                poly = boxes[i]
            poly = np.array(poly).astype(np.int32).reshape((-1))
            result.append(poly)
        return result

    def group_text_box(self, polys, slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=1.0, add_margin=0.05):
        # poly top-left, top-right, low-right, low-left
        horizontal_list, combined_list, merged_list = [], [], []
        for poly in polys:
            slope_up = (poly[3]-poly[1])/np.maximum(10, (poly[2]-poly[0]))
            slope_down = (poly[5]-poly[7])/np.maximum(10, (poly[4]-poly[6]))
            if max(abs(slope_up), abs(slope_down)) < slope_ths:
                x_max = max([poly[0], poly[2], poly[4], poly[6]])
                x_min = min([poly[0], poly[2], poly[4], poly[6]])
                y_max = max([poly[1], poly[3], poly[5], poly[7]])
                y_min = min([poly[1], poly[3], poly[5], poly[7]])
                horizontal_list.append(
                    [x_min, x_max, y_min, y_max, 0.5*(y_min+y_max), y_max-y_min])
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])
        # combine box
        new_box = []
        for poly in horizontal_list:
            if len(new_box) == 0:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                new_box.append(poly)
            else:
                # comparable height and comparable y_center level up to ths*height
                if (abs(np.mean(b_height) - poly[5]) < height_ths*np.mean(b_height)) and (abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths*np.mean(b_height)):
                    b_height.append(poly[5])
                    b_ycenter.append(poly[4])
                    new_box.append(poly)
                else:
                    b_height = [poly[5]]
                    b_ycenter = [poly[4]]
                    combined_list.append(new_box)
                    new_box = [poly]
        combined_list.append(new_box)

        # merge list use sort again
        for boxes in combined_list:
            if len(boxes) == 1:  # one box per line
                box = boxes[0]
                margin = int(add_margin*box[5])
                merged_list.append(
                    [box[0]-margin, box[1]+margin, box[2]-margin, box[3]+margin])
            else:  # multiple boxes per line
                boxes = sorted(boxes, key=lambda item: item[0])

                merged_box, new_box = [], []
                for box in boxes:
                    if len(new_box) == 0:
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        if abs(box[0]-x_max) < width_ths * (box[3]-box[2]):  # merge boxes
                            x_max = box[1]
                            new_box.append(box)
                        else:
                            x_max = box[1]
                            merged_box.append(new_box)
                            new_box = [box]
                if len(new_box) > 0:
                    merged_box.append(new_box)

                for mbox in merged_box:
                    if len(mbox) != 1:  # adjacent box in same line
                        x_min = min(mbox, key=lambda x: x[0])[0]
                        x_max = max(mbox, key=lambda x: x[1])[1]
                        y_min = min(mbox, key=lambda x: x[2])[2]
                        y_max = max(mbox, key=lambda x: x[3])[3]
                        margin = int(add_margin*(y_max - y_min))
                        merged_list.append(
                            [x_min-margin, x_max+margin, y_min-margin, y_max+margin])
                    else:
                        box = mbox[0]
                        margin = int(add_margin*(box[3] - box[2]))
                        merged_list.append(
                            [box[0]-margin, box[1]+margin, box[2]-margin, box[3]+margin])
        return merged_list

    def process_input(self, img, image_height=32, image_min_width=32, image_max_width=512):
        img = Image.fromarray(img)
        w, h = img.size
        new_w = int(image_height * float(w) / float(h))
        round_to = 10
        new_w = math.ceil(new_w/round_to)*round_to
        new_w = max(new_w, image_min_width)
        new_w = min(new_w, image_max_width)
        img = img.resize((new_w, image_height), Image.ANTIALIAS)
        img = np.array(img)
        img = torch.FloatTensor(img).permute(2, 0, 1)/255
        img = torch.unsqueeze(img, 0)
        return img

    def translate_beam_search(self, img):
        with torch.no_grad():
            memory = self.transformer(img)
            beam = Beam(beam_size=2, min_length=0, n_top=1,
                        ranker=None, start_token_id=1, end_token_id=2)
            for _ in range(128):
                tgt_inp = beam.get_current_state().transpose(0, 1).to(self.device)  # TxN
                decoder_outputs = self.transformer.transformer.forward_decoder(
                    tgt_inp, memory)
                log_prob = log_softmax(
                    decoder_outputs[:, -1, :].squeeze(0), dim=-1)
                beam.advance(log_prob.cpu())
                if beam.done():
                    break
            scores, ks = beam.sort_finished(minimum=1)
            hypothesises = []
            for times, k in ks:
                hypothesis = beam.get_hypothesis(times, k)
                hypothesises.append(hypothesis)
            encode = [1] + [int(i) for i in hypothesises[0][:-1]]
            return self.vocab.decode(encode)

    def __call__(self, img, min_size=20, text_threshold=0.7, low_text=0.4,
                 link_threshold=0.4, canvas_size=2560, mag_ratio=1.,
                 slope_ths=0.8, ycenter_ths=0.5, height_ths=1,
                 width_ths=1, add_margin=0.1):
        img = np.array(img)[:, :, :3]
        text_box = self.get_textbox(
            img, text_threshold, link_threshold, low_text)
        horizontal_list = self.group_text_box(text_box, slope_ths,
                                              ycenter_ths, height_ths,
                                              width_ths, add_margin)
        horizontal_list = [i for i in horizontal_list if max(
            i[1]-i[0], i[3]-i[2]) > 10]
        result = []
        for ele in horizontal_list:
            ele = [0 if i < 0 else i for i in ele]
            sub_img = img[ele[2]:ele[3], ele[0]:ele[1], :]
            sub_img = self.process_input(sub_img)
            sub_img = sub_img.to(self.device)
            result.append(self.translate_beam_search(sub_img))
        return result
