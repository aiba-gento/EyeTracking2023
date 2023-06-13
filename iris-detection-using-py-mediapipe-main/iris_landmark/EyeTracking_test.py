import cv2
from eyetracking_tflite import EyeTracking
import copy


cap_device = "C:/Users/gn10g/Documents/GitHub/EyeTracking2023/iris-detection-using-py-mediapipe-main/iris_landmark/eye.mp4"
cap_width = 960
cap_height = 540
# カメラ準備
cap = cv2.VideoCapture(cap_device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
eyetracking = EyeTracking(cap_device)


while True:
  ret, image = cap.read()
  if not ret:
    break
  debug_image = copy.deepcopy(image) # imageの複製
  eye_x, eye_y, eye_lid, eye_list = eyetracking.eyetrack(image)
  print("EyeX " + str(eye_x) + ", EyeY " + str(eye_y) + ", lid " + str(eye_lid))
  
  for i in range(15):
    debug_image = cv2.drawMarker(debug_image, eye_list[i], (0, 255, 0))
    debug_image = cv2.putText(debug_image, str(i), eye_list[i], cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
  cv2.imshow("image", debug_image)
  cv2.moveWindow("image", x=500, y=0)
  key = cv2.waitKey(10)
  if key == 27:
    break
cap.release()
cv2.destroyAllWindows()