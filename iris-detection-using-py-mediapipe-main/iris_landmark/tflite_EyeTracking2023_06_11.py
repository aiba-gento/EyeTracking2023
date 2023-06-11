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

while True:
  # カメラキャプチャ
  ret, image = cap.read()
  if not ret:             # 読み込まれなかったらwhileを抜ける
    break
  image = cv2.flip(image, 1) # 第２引数が１で上下左右反転
  debug_image = copy.deepcopy(image) # imageの複製
  eye_contour, iris = iris_detector(image) # 虹彩と目の輪郭を検出
  print(eye_contour)
  sleep(0.01)