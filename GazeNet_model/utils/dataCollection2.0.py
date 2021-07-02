import tkinter as tk
import time
import cv2
import os
import threading
from .preProcessing import GrayImg, lightRemove, gamma_trans, letter_box
from .eyeDetection import If_Eye as detector

import numpy as np
import math


# The moving area
delta_x = 960
delta_y = 540
m = 0
n = 0


# Click the button then the window will move
def moveit():
    global delta_x, delta_y, m, n
    window2 = tk.Tk()
    # On the top
    window2.attributes("-topmost", True)
    window2.overrideredirect(True)
    # Fill with red
    Full_color = tk.Label(window2, bg='red', width=10, height=10)
    Full_color.pack()
    while(n <= delta_y and m <= delta_x and m >= 0 and n >= 0):
        window2.geometry("%dx%d+%d+%d" % (10, 10, 480 + m, 270 + n))
        # Update the window
        window2.update()

    time.sleep(5)
    window2.destroy()



def catchPhoto():
    global m, n
    # Output path for image capturing
    base_dir = "../Dataset/first_dataset/train"
    cap = cv2.VideoCapture(0)
    print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
    # set window size
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    # variable for img name
    i = 16
    while True:
        ret, frame = cap.read()
        # mirror
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (128, 80), (512, 460), (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        dir_name = ("%dx%d" % (480 + m, 270 + n))
        output_dir = base_dir + "/" + dir_name
        key = cv2.waitKey(1)

        if key == 13:
            """Press Enter to capture image"""
            # create a new dir if it not exist
            try:
                os.mkdir(output_dir)
                i = 16
            except:
                pass
            count = 40
            while (count >= 0):
                # Gray value image
                gray = GrayImg(frame)
                # calculate the mean value
                mean = np.mean(gray)
                # adaptive gamma
                gamma_val = math.log10(0.5) / math.log10(mean / 255)
                # gamma transfer
                image_gamma = gamma_trans(frame, gamma_val)
                # back to gray-level
                image_gamma_correct = GrayImg(image_gamma)
                # light move
                img_gamma_Remove = lightRemove(image_gamma_correct)
                crop = img_gamma_Remove[80:460, 128:512]
                crop = letter_box(crop, [224, 224])
                output_path = os.path.join(output_dir, "%04d.jpg" % i)
                flag = detector(frame)
                if flag == True:
                    cv2.imwrite(output_path, crop)
                    i += 1
                    count = count - 1
                    ret, frame = cap.read()
                    # mirror
                    frame = cv2.flip(frame, 1)
                else:
                    ret, frame = cap.read()
                    # mirror
                    frame = cv2.flip(frame, 1)

        if key == 82:
            """Press r to rest the index"""
            i = 1
        if key == 87:
            """Press up to rest the index"""
            n = n - 270
        if key == 83:
            """Press down to rest the index"""
            n = n + 270
        if key == 65:
            """Press left to rest the index"""
            m = m - 480
        if key == 68:
            """Press right to rest the index"""
            m = m + 480
        if key == 27:
            """Press ESC to exit"""
            break



    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':

    point_thread = threading.Thread(target = moveit)
    point_thread.daemon = True
    camera_thread = threading.Thread(target = catchPhoto)
    camera_thread.daemon = True
    # start multi-processing
    point_thread.start()
    camera_thread.start()
    # wait a certain time the kill the main process
    time.sleep(1200)
    print("Process lasted too long and ended automatically !")



