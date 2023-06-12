##
# @file tflite_EyeTracking2023_06_11.py
# @version 1
# @author Aiba Gento
# @date 2023_06_12
# @brief EyeTrackingをTensorFlow Liteのmediapipe-irisモデルを用いて実現するプログラム
# @note 現段階では片目のみ対応しています。出力画像とマークの位置がずれています（tensor_to_list()関数の調整が必要そう）

import copy
from time import sleep

import cv2
import numpy as np
from iris_landmark import IrisLandmark

cap_device = "C:/Users/gn10g/Documents/GitHub/EyeTracking2023/iris-detection-using-py-mediapipe-main/iris_landmark/eye.mp4"  # 入力ディバイスをwebcamに指定
cap_width = 960 # 解像度
cap_height = 540

# カメラ準備
cap = cv2.VideoCapture(cap_device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

# モデルロード
iris_detector = IrisLandmark()
# 入力テンソルの形状
input_shape = iris_detector.get_input_shape()



def tensor_to_list(eye_contour_, iris_, image_width, image_height, input_shape):

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

while True:
  # カメラキャプチャ
  ret, image = cap.read()
  if not ret:             # 読み込まれなかったらwhileを抜ける
    break
  image = cv2.flip(image, 1) # 第２引数が１で上下左右反転
  debug_image = copy.deepcopy(image) # imageの複製
  eye_contour, iris = iris_detector(image) # 虹彩と目の輪郭を検出
  # imageの形状
  image_width, image_height = image.shape[1], image.shape[0]
  
  iris_list, eye_list = tensor_to_list(eye_contour, iris, image_width, image_height, input_shape)
  
  center, radius = calc_min_enc_losingCircle(iris_list)

  debug_image = cv2.circle(debug_image, center, radius, (0, 255, 0), 2)
  
  for i in range(15):
    debug_image = cv2.drawMarker(debug_image, eye_list[i], (0, 255, 0))
    debug_image = cv2.putText(debug_image, str(i), eye_list[i], cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
  
  iris_point = get_iris_local(center, eye_list)
  eye_level = get_eye_level(eye_list)
  print("iris:" + str(iris_point) + ", eye:" + str(eye_level))
  cv2.imshow("debug_image", debug_image)
  
  key = cv2.waitKey(5)
  if key == 27:
    break

cap.release()
cv2.destroyAllWindows()