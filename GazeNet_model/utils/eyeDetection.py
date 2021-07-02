import numpy as np
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist


# Eye opening rate: each eye is represented by 6 (x, y) coordinates, starting from the left corner of the eye, 
# and then displaying clockwise around the rest of the area
# ||P2 -P6|| + ||P3-P5|| / 2 * ||P4-P1||
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# the default resolution of the camera
WINDOWS_WIDTH = 640
WINDOWS_HEIGHT = 480
# when the EAR is below a certain threshold, the eyes are closed
EYE_AR_THRESH = 0.2
# when set to 3 frames or more, the eyes are considered to be closed, 
# which can detect blinking or effectively distinguish between blinking and closed eyes
EYE_AR_CONSEC_FRAMES = 2
# frame counter
COUNTER = 0
# number of closed eyes or blinks
TOTAL = 0
FPS = 0

# Right eye [37, 42], left eye [43, 48], 
# but since the subscripts start from zero, take [38, 42] and [42, 48] (left closed and right open)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]   # (42, 48)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # (36, 42)

def Eye_Detec(img):
    # set up the detector
    predictor_path = '../../shape_predictor_68_face_landmarks.dat'
    # face detection
    detector = dlib.get_frontal_face_detector()
    # predictor is responsible for finding the feature points of the face
    predictor = dlib.shape_predictor(predictor_path)
    # to prevent memory leaks, make a deep copy first
    img_show = img.copy()
    # determine whether it is a grayscale image
    if(len(img.shape) == 3 and img.shape[2] != 1):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    # use a list to store the coordinate points of the face
    pos_list = []
    # face detection
    faces = detector(img_gray, 0)
    # first determine whether a face is detected
    if (len(faces) != 0):
        for index, face in enumerate(faces):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, face).parts()])
            for idx, point in enumerate(landmarks):
                # 68 coordinates
                pos = (point[0, 0], point[0, 1])
                # save points
                pos_list.append(pos)
            # facial feature point detection
            shape = predictor(img, face)
            # convert the shape object into a two-dimensional array, 
            # so that it is convenient to find the coordinates of the left and right eyes
            shape = face_utils.shape_to_np(shape)
            # coordinates of the left and right eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            # corner of left eye
            left_left = leftEye[0, 0]
            left_right = leftEye[3, 0]
            left_top = max(leftEye[1, 1], leftEye[2, 1])
            left_down = max(leftEye[4, 1], leftEye[5, 1])
            left_upper_left = (left_left - 5, left_top - 5)
            left_down_right = (left_right + 5, left_down + 5)
            left = [left_upper_left, left_down_right]
            # left eye corner frame
            cv2.rectangle(img_show, left_upper_left, left_down_right, (0, 0, 255), 2)
            # corner of right eye
            right_left = rightEye[0, 0]
            right_right = rightEye[3, 0]
            right_top = max(rightEye[1, 1], rightEye[2, 1])
            right_down = max(rightEye[4, 1], rightEye[5, 1])
            right_upper_left = (right_left - 5, right_top - 5)
            right_down_right = (right_right + 5, right_down + 5)
            right = [right_upper_left, right_down_right]
            # right eye corner frame
            cv2.rectangle(img_show, right_upper_left, right_down_right, (0, 0, 255), 2)
            # how many faces are detected
            cv2.putText(img_show, 'There are {} faces in the picture.'.format(len(faces)),
                        (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # separately extract the human eye, and the cropping coordinates are [y0:y1, x0:x1]
            left_eye = img_show[left[0][1]:left[1][1], left[0][0]:left[1][0]]
            right_eye = img_show[right[0][1]:right[1][1], right[0][0]:right[1][0]]

    return left[0][0], left[0][1], left[1][0], left[1][1], right[0][0], right[0][1], right[1][0], right[1][1]

def If_Eye(img):
    # set up the detector
    predictor_path = '../../shape_predictor_68_face_landmarks.dat'
    # face detection
    detector = dlib.get_frontal_face_detector()
    # predictor is responsible for finding the feature points of the face
    predictor = dlib.shape_predictor(predictor_path)
    # determine whether it is a grayscale image
    if(len(img.shape) == 3 and img.shape[2] != 1):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    # face detection
    faces = detector(img_gray, 0)
    # first determine whether a face is detected
    if (len(faces) != 0):
        for index, face in enumerate(faces):
            # facial feature point detection
            shape = predictor(img, face)
            # convert the shape object into a two-dimensional array, 
            # so that it is convenient to find the coordinates of the left and right eyes
            shape = face_utils.shape_to_np(shape)
            # coordinates of the left and right eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            if(len(leftEye) != 0 and len(rightEye) != 0):
                return True
            else:
                return False
    else:
        return False



