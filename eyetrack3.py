import numpy as np
import cv2

thresh = 65  # 閾値
maxval = 255 # 閾値以上の値を持つ値に対して割り当てる値
th_type = cv2.THRESH_BINARY_INV # THRESH_BINARY_INVの場合しきい値よりも大きな値であれば0、それ以外はmaxvalの値にする。

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 分類器の読み込み
eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

cap = cv2.VideoCapture(0)

eyes = np.array(0)
face = np.array(0)
eyes_th = np.array(0)
contours = np.array(0)

while True:
    ret, frame = cap.read()                         # 画像の読み込み
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # グレイスケールに変換
    
    face_pram = face_cascade.detectMultiScale(gray)  # 顔の検出
    for (fx, fy, fw, fh) in face_pram:
        face = gray[fy: fy + fh, fx: fx + fw]        # 顔の検出 
        cv2.imshow("face", face)
        if ret > 0:
            eyes_pram = eye_cascade.detectMultiScale(face)  # 目の検出
            for (ex, ey, ew, eh) in eyes_pram:
                eyes = face[ey: ey + eh, ex: ex + ew]       # 目の切り抜き
    
        print(eyes.dtype)  
    
        eyes = eyes.astype(np.uint8)
    
        print(eyes.dtype)
        
        eyes = cv2.GaussianBlur(eyes,  (7, 7), 0)
        ret, eyes_th = cv2.threshold(eyes, thresh, maxval, cv2.THRESH_BINARY)   # 画像の二値化
        
        cv2.imshow("threshold", eyes_th)
    
        _, contours = cv2.findContours(eyes_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
        #contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
        """findContoursの引数
    
        第一引数：画像（二値化されている方が精度が上がるらしい）
    
        第二引数：抽出モード
                cv2.RETR_LIST     :単純に輪郭を検出
                cv2.RETR_EXTRNAL  :最も外側の輪郭を検出
                cv2.RETR_CCOMP    :階層を考慮し2レベルの輪郭を検出
                cv2.RETR_TREE     :すべての輪郭を検出し階層構造を保持
             
        第三引数：近似手法
                cv2.CHAIN_APPROX_NONE   :輪郭上のすべての点を保持する
                cv2.CHAIN_APPROX_SIMPLE :冗長な点情報を削除して返す
             
        """
    
    if contours is not None:
    # 最大の面積を持つ輪郭を探します
        cnt = contours[0]
        for c in contours:
            if len(cnt) < len(c):
                print(c)
                cnt = c
        print(cnt)
    
    # 最大の面積を持つ輪郭の重心を求めます
        m = cv2.moments(cnt, True)
    
        cx = int(m['m10']/m['m00'])
        cy = int(m['m01']/m['m00'])
    
        cv2.circle(eyes, (cx, cy), 20, (255, 0, 0), thickness=-1)
        
        cv2.imshow("eye", eyes)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()  