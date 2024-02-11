from ..pytorch_abc import PyTorchAbstractClass

import dlib
import cv2
import argparse, os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from src.model import model_static
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from colour import Color


parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str, help='input video path.')
#parser.add_argument('--face', type=str, help='face detection file path. dlib face detector is used when not specified')
parser.add_argument('--model_weight', type=str, help='path to model weights file', default='data/model_weights.pkl')
parser.add_argument('--jitter', type=int, help='jitter bbox n times, and average results', default=0)
parser.add_argument('-save_vis', help='saves output as video', action='store_true')
parser.add_argument('-save_text', help='saves output as text', action='store_true')
parser.add_argument('-display_off', help='do not display frames', action='store_true')

args = parser.parse_args()

CNN_FACE_MODEL = 'src/data/mmod_human_face_detector.dat' # from http://dlib.net/files/mmod_human_face_detector.dat.bz2
MAX_FRAME_COUNT = 100000



def bbox_jitter(bbox_left, bbox_top, bbox_right, bbox_bottom):
    cx = (bbox_right+bbox_left)/2.0
    cy = (bbox_bottom+bbox_top)/2.0
    scale = random.uniform(0.8, 1.2)
    bbox_right = (bbox_right-cx)*scale + cx
    bbox_left = (bbox_left-cx)*scale + cx
    bbox_top = (bbox_top-cy)*scale + cy
    bbox_bottom = (bbox_bottom-cy)*scale + cy
    return bbox_left, bbox_top, bbox_right, bbox_bottom


def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)



class Eye_Contact_CNN(PyTorchAbstractClass):

    model_weight = args.model_weight
    video_path = args.video
    save_text = args.save_text
    vis = args.save_vis
    jitter = args.jitter
    display_off = args.display_off



    # Set up data transformation. This processes the video frame
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
    def __init__(self):
        # Set up Video-Source
        self.cap = cv2.VideoCapture(self.video_path)
        if (self.cap.isOpened()== False):
            print("Error opening video stream or file")
            exit()

        # Set up output file formats
        if self.save_text:
            outtext_name = os.path.basename(self.video_path).replace('.avi','_output.txt')
            self.f = open(outtext_name, "w")
        if self.vis:
            outvis_name = os.path.basename(self.video_path).replace('.avi','_output.avi')
            imwidth = int(self.cap.get(3)); imheight = int(self.cap.get(4))
            self.outvid = cv2.VideoWriter(outvis_name,cv2.VideoWriter_fourcc('M','J','P','G'), self.cap.get(5), (imwidth,imheight))

        # Set up video settings (for drawing bounding boxes)
        red = Color("red")
        self.colors = list(red.range_to(Color("green"),10))
        self.font = ImageFont.truetype("data/arial.ttf", 40)

        # Set up Face Detection Mode
        self.facemode = 'DLIB'
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)

        # Load Model
        self.model = model_static(self.model_weight)
        self.model_dict = self.model.state_dict()
        snapshot = torch.load(self.model_weight)
        self.model_dict.update(snapshot)
        self.model.load_state_dict(self.model_dict)

        self.model.cuda()
        self.model.train(False)

    # Read frames from the video and apply any initial processing required before face detection. It will yield processed frames one by one.
    def preprocess(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_rgb

    def predict(self, frame_rgb):
        frame_cnt = next(self.frame_counter, None)  # frame_counter is initialized in run() as an iterable
        frame = Image.fromarray(frame_rgb) # Saved current frame as image memory

        bbox = []
        if self.facemode == 'DLIB':
            dets = self.cnn_face_detector(frame_rgb, 1)
            for d in dets:
                l = d.rect.left()
                r = d.rect.right()
                t = d.rect.top()
                b = d.rect.bottom()
                # expand a bit
                l -= (r-l)*0.2
                r += (r-l)*0.2
                t -= (b-t)*0.2
                b += (b-t)*0.2
                bbox.append([l, t, r, b])
        else:
            print("\nAlternate face modes not implemented yet\n")

        predictions = []
        for b in bbox:
            face = frame.crop((b))
            img = self.test_transforms(face).unsqueeze_(0) #Reminder: test_transform resizes, crops, converts to tensor, and normalizes.

            if self.jitter > 0:
                for i in range(self.jitter):
                    bj_left, bj_top, bj_right, bj_bottom = bbox_jitter(b[0], b[1], b[2], b[3])
                    bj = [bj_left, bj_top, bj_right, bj_bottom]
                    facej = frame.crop((bj))
        
                    img_jittered = self.test_transforms(facej).unsqueeze_(0)
                    img = torch.cat([img, img_jittered])

            output = self.model(img.cuda())

            #If jittering was applied, the jittered outputs are averaged
            if self.jitter > 0:
                output = torch.mean(output, 0)
            score = torch.sigmoid(output).item()
            predictions.append((b, score))
        return predictions
  
    def postprocess(self, predictions):
        frame_rgb, frame_cnt = self.current_frame, next(self.frame_counter, None)  # Update to fetch the current frame and its count appropriately
        frame_pil = Image.fromarray(frame_rgb)
        for b, score in predictions:
            coloridx = min(int(round(score*10)), 9)
            draw = ImageDraw.Draw(frame_pil)
            self.drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=self.colors[coloridx].hex, width=5)
            draw.text((b[0], b[3]), str(round(score, 2)), fill=(255, 255, 255, 128), font=self.font)
            if self.save_text:
                self.f.write("%d,%f\n" % (frame_cnt, score))

        if not self.display_off:
            frame_np = np.array(frame_pil)  # Convert back to NumPy array for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            cv2.imshow('', frame_bgr)
            if self.vis:
                self.outvid.write(frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        return True
    
    def run(self):
        self.frame_counter = iter(range(MAX_FRAME_COUNT))
        for frame_rgb in self.preprocess():
            self.current_frame = frame_rgb  # Store the current frame for access in postprocess
            predictions = self.predict(frame_rgb)
            if not self.postprocess(predictions):
                break

        if self.vis:
            self.outvid.release()
        if self.save_text:
            self.f.close()
        self.cap.release()
        print('DONE!')

