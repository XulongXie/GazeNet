import colorsys
import os
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import time
import tkinter as tk
import threading

from GazeNet_model.models.gazeNet import GazeNetModel as Network
from GazeNet_model.utils.visualization import visual
from GazeNet_model.utils.eyeDetection import If_Eye, Eye_Detec

from GazeNet_model.utils.preProcessing import GrayImg, lightRemove, gamma_trans, letter_box


pos = (0, 0)

# --------------------------------------------#
# Use the self-trained model to predict 2 parameters need to be modified
# Both model_path and classes_path need to be modified!
# If there is a shape mismatch, you must pay attention
# Modification of model_path and classes_path parameters during training
# --------------------------------------------#
class gazeNet(object):
    _defaults = {
        # here can be modified according to needs
        "model_path": '../logs/second_dataset/Epoch96_best_checkpoint.pth.tar',
        "classes_path": '../model_data/myGaze_classes.txt',
        # the size of the image and the number of channels
        "model_image_size": (64, 64, 1),
        # need to be set to true if CUDA is available
        "cuda": False,
        # this variable is used to control whether letterbox_image is used to resize the input image without distortion
        "letterbox_image": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # initialize the network
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

    # get all categories
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def load_checkpoint(self, device):
        filename = self.model_path
        print(filename)
        if not os.path.isfile(filename):
            return None
        state = torch.load(filename, map_location = device)
        return state

    # generate the model
    def generate(self):
        # set the network
        self.network = Network()

        # load the weight of the network
        print('Loading weights into state dict...')
        # leave the interface for later training with GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        saved = self.load_checkpoint(device)
        if saved:
            print('Loading checkpoint for epoch %05d ...' % (saved['epoch']))
            state = saved['model_state']
            try:
                self.network.module.load_state_dict(state)
            except:
                self.network.load_state_dict(state)
        else:
            print('Warning: Could not read checkpoint!')
        print('Finished!')

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.network = nn.DataParallel(self.network)
            self.network = self.network.cuda()

        # set different colors for the picture frame, use one kind of picture frame for each class
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    # image detection, here is the format of np
    def detect_image(self, image):
        self.network.eval()
        if (len(image.shape) == 3 and image.shape[2] != 1):
            img = GrayImg(image)
            mean = np.mean(img)
            gamma_val = math.log10(0.5) / math.log10(mean / 255)
            image_gamma = gamma_trans(image, gamma_val)
            image_gamma_correct = GrayImg(image_gamma)
            img_gamma_Remove = lightRemove(image_gamma_correct)
        else:
            img_gamma_Remove = image

        # determine whether eyes are detected
        if If_Eye(img_gamma_Remove):
            x0, y0, x1, y1, x2, y2, x3, y3 = Eye_Detec(img_gamma_Remove)
            img_gamma_Remove = Image.fromarray(np.uint8(img_gamma_Remove))
            leftEye = img_gamma_Remove.crop([x0, y0, x1, y1])
            rightEye = img_gamma_Remove.crop([x2, y2, x3, y3])
            # add gray bars to the image to achieve undistorted resize, or you can directly resize for recognition
            if self.letterbox_image:
                leftEye = np.array(letter_box(leftEye, [self.model_image_size[0], self.model_image_size[1]]))
                rightEye = np.array(letter_box(rightEye, [self.model_image_size[0], self.model_image_size[1]]))
            else:
                leftEye = leftEye.resize((self.model_image_size[0], self.model_image_size[1]), Image.BICUBIC)
                rightEye = rightEye.resize((self.model_image_size[0], self.model_image_size[1]), Image.BICUBIC)

            # Normalization is very important here, because the network results have a lot to do with correct standardization, 
            # so you can do without normalization for self-testing
            leftEye = np.array(leftEye, dtype=np.float32)
            rightEye = np.array(rightEye, dtype=np.float32)

            # add batch_size and channel
            leftEye = leftEye.reshape((1, leftEye.shape[0], leftEye.shape[1]))
            rightEye = rightEye.reshape((1, rightEye.shape[0], rightEye.shape[1]))
            leftEyes = [leftEye]
            rightEyes = [rightEye]

            with torch.no_grad():
                leftEyes = torch.from_numpy(np.asarray(leftEyes))
                rightEyes = torch.from_numpy(np.asarray(rightEyes))
                if self.cuda:
                    leftEyes = leftEyes.cuda()
                    rightEyes = rightEyes.cuda()
                # put the image into the network for prediction!
                eyeCorners = np.array([x0, y0, x1, y1, x2, y2, x3, y3])
                eyeCorners = torch.from_numpy(eyeCorners)
                eyeCorners = eyeCorners.reshape(1, 8)
                imEyeL = torch.autograd.Variable(leftEyes, requires_grad=False)
                imEyeR = torch.autograd.Variable(rightEyes, requires_grad=False)
                EyeCorner = torch.autograd.Variable(eyeCorners.float(), requires_grad=False)
                outputs = self.network(imEyeL, imEyeR, EyeCorner)
                pred = outputs.argmax(dim=1).item()
                return pred
        else:
            return None

def move_win():
    global pos
    window = tk.Tk()
    # On the top
    window.attributes("-topmost", True)
    #window.attributes("-alpha", 0.7)
    window.overrideredirect(True)
    # Fill with red
    Full_color = tk.Label(window, bg='red', width = 480, height = 270)
    Full_color.pack()
    while True:
        #print(pos)
        window.geometry("%dx%d+%d+%d" % (480, 270, pos[0] - 240, pos[1] - 135))
        # Update the window
        window.update()

def prediction():
    global pos
    gaze_net = gazeNet()
    capture = cv2.VideoCapture(0)
    fps = 0.0
    while (True):
        t1 = time.time()
        # read a frame
        ref, frame = capture.read()
        # flip
        frame = cv2.flip(frame, 1)
        first_point = (128, 80)
        last_point = (512, 460)
        # cropping
        crop = frame[first_point[1]:last_point[1], first_point[0]:last_point[0]]
        # detection
        pred = np.array(gaze_net.detect_image(crop))
        print(pred)
        # compute the coordinates
        pos = visual(pred)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))

        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict_thread = threading.Thread(target = prediction)
    visual_thread = threading.Thread(target = move_win)
    # start multi-processing
    predict_thread.start()
    visual_thread.start()



