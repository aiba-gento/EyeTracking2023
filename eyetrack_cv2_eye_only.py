##
# @file eyetrack_cv2_only.py
# @version 1
# @author Aiba Gento
# @date 2023_06_10
# @brief Opencvのみを使ってEyetrackingを行うプログラム
# @details 目以外の映っていない画像に対応しています。

import numpy as np # Numpy（数値計算）
import cv2 # OpenCV（画像処理）

cap = cv2.VideoCapture("eye.mp4")  # ファイルの場所、0を引数にとればwebcamからのリアルタイム画像を扱える

while True:
  # フレームの読み込み
  ret, frame = cap.read()  # 1フレーム読み込む
  if ret is False:         # 読み込めなかった場合（動画が終わったら）breakする
    break
  assert frame is not None, "frameがNoneです" # デバックをしやすくする
  
  # 画像の加工
  height, width, _ = frame.shape   # 画像の大きさを得る。第三引数は色の数
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 画像をモノクロにする
  gray = cv2.GaussianBlur(gray, (7,7), 0)  
  """画像の平滑化（ぼかす）
  第2引数(カーネル)を大きくするとぼかしが強くなる
  第3引数は標準偏差で0にしておけば自動で最適化される。"""
  
  _, threshold = cv2.threshold(gray, 10, 255,cv2.THRESH_BINARY_INV)
  """画像の二値化
  第2引数：小さいほうの閾値（大きくするとまつ毛や少し暗いところなど、敏感に反応する）
  第3引数：大きいほうの閾値（今回は変えない）
  """
  
  
  # 輪郭の検出
  contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  """二値化された画像から輪郭を検出する
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
  assert contours is not None, "contoursがNoneです"
  for cnt in contours:
    (x, y, w, h) = cv2.boundingRect(cnt)  # 点で囲われた矩形領域を取得
    
    # 検出された輪郭の表示
    cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 10)
    
    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)  # 最も大きな面積を持つ輪郭を抽出
    """sorted（並び替え）
    第1引数：複数の要素を持ち並び替えしたいオブジェクト
    第2引数：何を基準に並び変えるか(lambdaは何もしない)
      cv2.contourAreaで面積を基準にする
    reverse引数：Trueで降順,Falseで昇順
    """
    
    # 矩形からラインを描写
    cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), height), (0, 0, 255), 2)
    cv2.line(frame, (0, y + int(h/2)), (width, y + int(h/2)), (0, 0, 255), 2)
    break
  
  cv2.imshow("line", frame)  # ラインが描写されたフレームを表示
  
  key = cv2.waitKey(4)  # キーが押されたら、値を代入（引数は何ミリ秒待つか）
  if key == 27:
    break

cap.release()
cv2.destroyAllWindows() # 全てのウィンドウを閉じる