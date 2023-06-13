import copy
from time import sleep

import cv2
import numpy as np
from iris_landmark import IrisLandmark

class EyeTracking(object):
  def __init__(self, cap_device):
    self.cap_device = cap_device
    self.cap_width = 960
    self.cap_height = 540
    # カメラ準備
    cap = cv2.VideoCapture(self.cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_height)
    # モデルをロードする
    self.irislandmark = IrisLandmark()
    # 入力テンソルの形状
    self.input_shape = self.irislandmark.get_input_shape()
  
  
  def tensor_to_list(self, eye_contour_, iris_, image_width, image_height, input_shape):

    iris_list = []
    eye_list = []

    for index in range(5):
      iris_x = (iris_[index * 3]) * image_width / input_shape[0]
      iris_y = (iris_[index * 3 + 1]) * image_height / input_shape[1]
      iris_x = int(iris_x)
      iris_y = int(iris_y)
      iris_list.append((iris_x, iris_y))

    for index_ in range(15):
      eye_x = (eye_contour_[index_ * 3]) * image_width / input_shape[0]
      eye_y = (eye_contour_[index_ * 3 + 1]) * image_height / input_shape[1]
      eye_x = int(eye_x)
      eye_y = int(eye_y)
      eye_list.append((eye_x, eye_y))
    
    return iris_list, eye_list
  
  
  def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv2.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    return center, radius
  
  def get_iris_local(iris_center, eye_list):
  
    center = iris_center[0] - eye_list[0][0]
    eye_width = eye_list[8][0] - eye_list[0][0]
    eye_point = (center - eye_width/2 ) / eye_width
    return eye_point

  def get_eye_level(eye_list):
    eye_level = eye_list[3][1] - eye_list[12][1]
    return eye_level
  
  def eyetrack(self, image):
    assert image is not None, "imageがNone"
    image = cv2.flip(image, 1) # 第２引数が１で上下左右反転
    debug_image = copy.deepcopy(image) # imageの複製
    eye_contour, iris = self.irislandmark(image) # 虹彩と目の輪郭を検出
    # imageの形状
    image_width, image_height = image.shape[1], image.shape[0]
    
    iris_list, eye_list = self.tensor_to_list(eye_contour, iris, image_width, image_height, self.input_shape)