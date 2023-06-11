#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import tensorflow as tf


class IrisLandmark(object):
    # 1回目の呼び出し（コンストラクタ）
    def __init__(
        self,
        model_path="C:/Users/gn10g/Documents/GitHub/EyeTracking2023/iris-detection-using-py-mediapipe-main/iris_landmark/iris_landmark.tflite",
        num_threads=2
    ):
        self._interpreter = tf.lite.Interpreter(model_path=model_path,
                                                num_threads=num_threads)  #モデルのロード
        self._interpreter.allocate_tensors()                              #テンソルの確保
        self._input_details = self._interpreter.get_input_details()       #モデルの入力テンソルの詳細を取得する
        self._output_details = self._interpreter.get_output_details()     #モデルの出力テンソルの詳細を取得する

    # 2回目の呼び出し（Pythonの特殊機能）
    def __call__(
        self,
        image,
    ):
        input_shape = self._input_details[0]['shape']  #モデルの入力テンソルの形状を取得

        # 正規化・リサイズ
        img = cv.cvtColor(image, cv.COLOR_BGR2RGB)  #グレイスケール化
        img = img / 255.0                           # 最大値1.0最小値0.0のfloat型に変換
        # https://qiita.com/Hironsan/items/d2a6364221c588867a60#:~:text=tf.image.resize_images%20%28images%2C%20new_height%2C%20new_width%2C%20method%3D0%2C%20align_corners%3DFalse%29%20resize_images%E3%81%AF%E7%94%BB%E5%83%8F%E3%82%92%E6%8C%87%E5%AE%9A%E3%81%97%E3%81%9Fmethod%E3%81%A7new_height%20x,%5Bbatch%2C%20height%2C%20width%2C%20channels%5D%E3%81%8B3D%E3%81%AE%E3%83%86%E3%83%B3%E3%82%BD%E3%83%AB%20%5Bheight%2C%20width%2C%20channels%5D%E3%82%92%E4%B8%8E%E3%81%88%E3%82%8B%E3%81%93%E3%81%A8%E3%81%8C%E3%81%A7%E3%81%8D%E3%82%8B%E3%80%82%204D%E3%81%A7%E4%B8%8E%E3%81%88%E3%82%8C%E3%81%B0%E7%94%BB%E5%83%8F%E3%81%AE%E4%B8%80%E6%8B%AC%E5%A4%89%E6%8F%9B%E3%81%8C%E5%8F%AF%E8%83%BD%E3%80%82
        img_resized = tf.image.resize(img, [input_shape[1], input_shape[2]],
                                      method='bicubic',
                                      preserve_aspect_ratio=False)
        img_input = img_resized.numpy() # numpyにする
        img_input = (img_input - 0.5) / 0.5  # 最大1.0最小-1.0になる

        reshape_img = img_input.reshape(1, input_shape[1], input_shape[2],
                                        input_shape[3])                    # 入力テンソルに合わせる
        tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)       # テンソルに戻す

        # 推論実行
        input_details_tensor_index = self._input_details[0]['index']       # インタプリタ内のテンソルインデックス
        self._interpreter.set_tensor(input_details_tensor_index, tensor)   # 入力テンソルにデータをセット第一引数：インデックス、第二引数：データ
        self._interpreter.invoke()                                         # モデルを実行

        # 推論結果取得
        output_details_tensor_index0 = self._output_details[0]['index']    # インデックスを取得
        output_details_tensor_index1 = self._output_details[1]['index']
        eye_contour = self._interpreter.get_tensor(output_details_tensor_index0) # 引数インデックスの出力テンソルを取得
        iris = self._interpreter.get_tensor(output_details_tensor_index1)

        return np.squeeze(eye_contour), np.squeeze(iris)  # サイズが1の次元を削除

    def get_input_shape(self):
        input_shape = self._input_details[0]['shape']  # 入力テンソルの形状を取得
        return [input_shape[1], input_shape[2]]