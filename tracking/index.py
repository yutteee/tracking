#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import copy
import time
import argparse
import os

import cv2 as cv

WIDTH = 1031.6 # [µm]
HEIGHT = 547.08 # [µm]

# 動画の読み込み
def get_args(movie):
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default=movie)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    args = parser.parse_args()

    return args


def isint(s):
    p = '[-+]?\d+'
    return True if re.fullmatch(p, s) else False

def initialize_tracker(window_name, image):
    params = cv.TrackerDaSiamRPN_Params()
    params.model = "model/DaSiamRPN/dasiamrpn_model.onnx"
    params.kernel_r1 = "model/DaSiamRPN/dasiamrpn_kernel_r1.onnx"
    params.kernel_cls1 = "model/DaSiamRPN/dasiamrpn_kernel_cls1.onnx"
    tracker = cv.TrackerDaSiamRPN_create(params)

    # 追跡対象指定
    while True:
        bbox = cv.selectROI(window_name, image)

        try:
            tracker.init(image, bbox)
        except Exception as e:
            print(e)
            continue

        return tracker


def main():
    color_list = [
        [255, 0, 0],  # blue
    ]

    INPUT_DIR = '../data_input/'
    video_files = []
    for file in os.listdir(INPUT_DIR):
        if file.endswith('.avi') or file.endswith('.mp4'):
            video_files.append(file)
    
    for video_file in video_files:
        # 引数解析 #################################################################

        args = get_args(INPUT_DIR + video_file)

        cap_device = args.device
        cap_width = args.width
        cap_height = args.height


        # カメラ準備 ###############################################################
        if isint(cap_device):
            cap_device = int(cap_device)
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        # Tracker初期化 ############################################################
        cv.namedWindow(video_file)

        ret, image = cap.read()
        if not ret:
            sys.exit("Can't read first frame")
        tracker = initialize_tracker(video_file, image)

        x_list = []
        y_list = []
        time_list = []

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            debug_image = copy.deepcopy(image)

            # 追跡アップデート
            ok, bbox = tracker.update(image)
            if ok:
                # 追跡後のバウンディングボックス描画
                cv.rectangle(debug_image, bbox, color_list[0], thickness=2)
            
            # 中心座標描画
            x = int(bbox[0] + bbox[2] / 2) 
            y = int(bbox[1] + bbox[3] / 2)
            center = (x, y)

            x_list.append(x)
            y_list.append(y)
            time_list.append(cap.get(cv.CAP_PROP_POS_MSEC)) # [ms]
            
            cv.circle(debug_image, center, 3, color_list[0], thickness=-1)
            cv.putText(
                debug_image,
                'center' + " : " + str(center),
                (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, color_list[0], 2,
                cv.LINE_AA)

            cv.imshow(video_file, debug_image)

            k = cv.waitKey(1)
            if k == 32:  # SPACE
                # 追跡対象再指定
                tracker = initialize_tracker(video_file, image)
            if k == 27:  # ESC
                break

        min_x = min(x_list)
        min_y = min(y_list)
        filtered_x_list = [WIDTH * (x - min_x) / cap_width for x in x_list]
        filtered_y_list = [HEIGHT * (y - min_y) / cap_height for y in y_list]

        # csvに変位の保存
        with open(f'../data_output/{video_file}.csv', 'a', encoding="" ) as f:
            f.write('t(ms),x(µm),y(µm)\n')
            for t, x, y in zip(time_list, filtered_x_list, filtered_y_list):
                f.write(f'{t},{x},{y}\n')


if __name__ == '__main__':
    main()