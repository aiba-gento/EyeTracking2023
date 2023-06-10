import numpy as np
import cv2
from time import sleep

#cap = cv2.VideoCapture(0)
x = 0
y = 0
w = 0
h = 0

while True:
  #ret, frame = cap.read()
  ret = 1
  frame = cv2.imread("eye.png", cv2.IMREAD_GRAYSCALE)
  assert frame is not None, "file could not be read, check with os.path.exists()" 
  gray = frame
  cv2.imshow("frame",frame)
  sleep(5)
  if ret is False:
    break
  height, width = frame.shape
  #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (7,7), 0)
  _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
  
  cv2.imshow("threhold",threshold)
  
  _, contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  print(contours)
  assert contours is not None, "file could not be read, check with os.path.exists()" 
  for cnt in contours:
    (x, y, w, h) = cv2.boundingRect(cnt)
    
    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
    
    cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), height), (0, 0, 255), 2)
    cv2.line(frame, (0, y + int(h/2)), (width, y + int(h/2)), (0, 0, 255), 2)
    break
  
  print(x + "  " + y + "  " + "  " + w + "  " + h)
  cv2.imshow("line", frame)
  key = cv2.waitKey(30)
  if key == 27:
    break

cv2.destroyAllWindows()