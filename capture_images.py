#!/usr/bin/env python3
"""
数据采集：从摄像头抓图存到 dataset/raw/<类别名>/。

用法：

python3 capture_images.py --class mouse

空格拍一张

自动连拍：

python3 capture_images.py \
--class mouse \
--auto \
--interval 0.5


按键：

空格  拍一张
a     切换自动连拍
u     删除上一张
q/ESC 退出
"""


import argparse
import os
import time
from datetime import datetime

import cv2



def build_capture(args):

    if args.csi:

        pipeline = (
            f'nvarguscamerasrc sensor-id={args.device} ! '
            f'video/x-raw(memory:NVMM), '
            f'width=(int){args.width}, '
            f'height=(int){args.height}, '
            f'framerate=(fraction){args.fps}/1 ! '
            f'nvvidconv flip-method={args.flip} ! '
            f'video/x-raw, format=(string)BGRx ! '
            f'videoconvert ! '
            f'video/x-raw, format=(string)BGR ! '
            f'appsink drop=true max-buffers=1'
        )

        return cv2.VideoCapture(
            pipeline,
            cv2.CAP_GSTREAMER
        )


    cap = cv2.VideoCapture(
        args.device
    )


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        args.width
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        args.height
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        args.fps
    )


    return cap




def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--class',
        dest='cls',
        required=True,
        help='类别名，例如 mouse keyboard'
    )


    parser.add_argument(
        '--out',
        default='../dataset/raw',
        help='输出根目录'
    )


    parser.add_argument(
        '--device',
        type=int,
        default=0
    )


    parser.add_argument(
        '--csi',
        action='store_true',
        help='使用 Jetson CSI 摄像头'
    )


    parser.add_argument(
        '--flip',
        type=int,
        default=0
    )


    parser.add_argument(
        '--width',
        type=int,
        default=1280
    )


    parser.add_argument(
        '--height',
        type=int,
        default=720
    )


    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='摄像头FPS'
    )


    parser.add_argument(
        '--auto',
        action='store_true',
        help='启动自动连拍'
    )


    parser.add_argument(
        '--interval',
        type=float,
        default=0.5,
        help='自动连拍间隔'
    )


    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='采集数量，0表示不限'
    )


    args = parser.parse_args()



    out_dir = os.path.join(
        args.out,
        args.cls
    )


    os.makedirs(
        out_dir,
        exist_ok=True
    )



    cap = build_capture(args)


    if not cap.isOpened():

        raise SystemExit(
            '无法打开摄像头'
        )



    auto = args.auto

    last_save = 0.0

    saved = []



    print(
        f'输出目录: {os.path.abspath(out_dir)}'
    )


    print(
        '空格=拍照  a=自动连拍开关  u=撤销  q=退出'
    )



    while True:


        ok, frame = cap.read()


        if not ok:

            print('读取失败')

            break



        now = time.time()


        save_flag = False



        if auto and now-last_save >= args.interval:

            save_flag = True



        preview = frame.copy()



        current = len(saved)


        target = (
            args.limit
            if args.limit
            else "-"
        )


        hud = (
            f'{args.cls} | '
            f'{current}/{target} | '
            f'auto {"ON" if auto else "OFF"}'
        )


        cv2.rectangle(
            preview,
            (0,0),
            (preview.shape[1],30),
            (0,0,0),
            -1
        )


        cv2.putText(
            preview,
            hud,
            (8,22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )


        cv2.imshow(
            'capture',
            preview
        )



        key = cv2.waitKey(1)&0xff



        if key in (
            ord('q'),
            27
        ):

            break



        if key == ord(' '):

            save_flag=True



        if key == ord('a'):

            auto = not auto



        if key == ord('u') and saved:

            os.remove(
                saved.pop()
            )

            print(
                f'已撤销，剩余 {len(saved)} 张'
            )



        if save_flag:


            stamp = datetime.now().strftime(
                '%Y%m%d_%H%M%S_%f'
            )[:-3]


            path = os.path.join(
                out_dir,
                f'{args.cls}_{stamp}.jpg'
            )



            success = cv2.imwrite(
                path,
                frame
            )



            if success:

                saved.append(path)

                last_save = now

                print(
                    f'[{len(saved)}] {path}'
                )


            else:

                print(
                    '图片保存失败'
                )



            if (
                args.limit
                and len(saved)>=args.limit
            ):

                print(
                    '已达到目标张数'
                )

                break



    cap.release()

    cv2.destroyAllWindows()



    print(
        f'本次共采集 {len(saved)} 张'
    )


    print(
        f'保存位置: {os.path.abspath(out_dir)}'
    )



if __name__ == '__main__':

    main()