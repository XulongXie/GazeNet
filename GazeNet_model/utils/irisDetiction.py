import cv2
import dlib
from imutils import face_utils


# Right eye [37, 42], left eye [43, 48], but since the subscripts start from zero, 
# take [36, 42] and [42, 48] (left closed and right open)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]   # (42, 48)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # (36, 42)

def Iris_Detec(img):
    # set up the detector first
    predictor_path = '../../shape_predictor_68_face_landmarks.dat'
    # the detector is only responsible for detecting faces
    detector = dlib.get_frontal_face_detector()
    # the predictor is responsible for finding the feature points of the face
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
            # the edge of the left eye
            leftEye = shape[lStart:lEnd]
            # find the center of mass of the left eye
            moments_left = cv2.moments(leftEye)
            left_x = int(moments_left['m10'] / moments_left['m00'])
            left_y = int(moments_left['m01'] / moments_left['m00'])
            # find the center of mass of the right eye
            rightEye = shape[rStart:rEnd]
            moments_right = cv2.moments(rightEye)
            right_x = int(moments_right['m10'] / moments_right['m00'])
            right_y = int(moments_right['m01'] / moments_right['m00'])
            # center distance of left and right center of mass
            x = (left_x + right_x) // 2
            y = (left_y + right_y) // 2

    return left_x, left_y, right_x, right_y, x, y






