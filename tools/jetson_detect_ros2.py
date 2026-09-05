#!/usr/bin/env python3
"""
Jetson real-time YOLO detection with ROS2 publishing and result saving.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import cv2


script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)

from ultralytics import YOLO


def build_capture(args):

    if args.csi:
        pipeline = (
            f"nvarguscamerasrc sensor-id={args.device} ! "
            f"video/x-raw(memory:NVMM), "
            f"width=(int){args.width}, "
            f"height=(int){args.height}, "
            f"framerate=(fraction){args.fps}/1 ! "
            f"nvvidconv flip-method={args.flip} ! "
            f"video/x-raw, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! "
            f"appsink drop=true max-buffers=1"
        )

        return cv2.VideoCapture(
            pipeline,
            cv2.CAP_GSTREAMER
        )


    cap = cv2.VideoCapture(args.device)

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



class RosPublisher:

    def __init__(self, enabled, topic):

        self.enabled = enabled

        if not enabled:
            return

        try:
            import rclpy
            from std_msgs.msg import String

        except ImportError as exc:
            raise SystemExit(
                "ROS2 Python package is unavailable"
            ) from exc


        rclpy.init(args=None)

        self.rclpy = rclpy
        self.msg_type = String

        self.node = rclpy.create_node(
            "desk_object_detector"
        )

        self.publisher = self.node.create_publisher(
            String,
            topic,
            10
        )


    def publish(self, payload):

        if not self.enabled:
            return

        msg = self.msg_type()

        msg.data = json.dumps(
            payload,
            ensure_ascii=False
        )

        self.publisher.publish(msg)

        self.rclpy.spin_once(
            self.node,
            timeout_sec=0
        )


    def close(self):

        if not self.enabled:
            return

        self.node.destroy_node()

        self.rclpy.shutdown()



def draw_detections(frame, detections, fps):

    for det in detections:

        x1,y1,x2,y2 = det["xyxy"]

        label = (
            f'{det["class_name"]} '
            f'{det["confidence"]:.2f}'
        )


        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )


        cv2.putText(
            frame,
            label,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )


    hud = (
        f"FPS:{fps:.1f} "
        "| q quit | e save error"
    )


    cv2.putText(
        frame,
        hud,
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )



def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model",
        default="best_mouse_keyboard.pt"
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0
    )

    parser.add_argument(
        "--yolo-device",
        default="0"
    )

    parser.add_argument(
        "--csi",
        action="store_true"
    )

    parser.add_argument(
        "--flip",
        type=int,
        default=0
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--topic",
        default="/desk_object_detections"
    )

    parser.add_argument(
        "--no-ros",
        action="store_true"
    )

    parser.add_argument(
        "--out",
        default="results"
    )


    args = parser.parse_args()



    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    out_dir = os.path.join(
        args.out,
        run_id
    )

    error_dir = os.path.join(
        out_dir,
        "errors"
    )


    os.makedirs(
        error_dir,
        exist_ok=True
    )



    print(
        "Loading model:",
        os.path.abspath(args.model)
    )


    model = YOLO(args.model)


    cap = build_capture(args)


    if not cap.isOpened():

        raise SystemExit(
            "Cannot open camera"
        )



    ros = RosPublisher(
        not args.no_ros,
        args.topic
    )


    video_path = os.path.join(
        out_dir,
        "result.mp4"
    )


    csv_path = os.path.join(
        out_dir,
        "detections.csv"
    )



    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width,args.height)
    )



    frame_id = 0

    fps = 0

    fps_sum = 0

    fps_count = 0

    last_time = time.time()



    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:


        csv_writer = csv.writer(f)


        csv_writer.writerow(
            [
                "frame",
                "class",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "fps"
            ]
        )



        while True:


            ok,frame = cap.read()


            if not ok:
                break



            frame_id += 1



            now = time.time()


            fps = (
                0.9*fps
                +
                0.1*(1/max(now-last_time,1e-6))
            )


            last_time = now


            fps_sum += fps

            fps_count += 1



            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.yolo_device,
                verbose=False
            )[0]



            detections=[]



            for box in result.boxes:


                cls_id=int(
                    box.cls[0].item()
                )

                conf=float(
                    box.conf[0].item()
                )


                xyxy=[
                    int(v)
                    for v in box.xyxy[0].tolist()
                ]


                name=result.names[cls_id]


                detections.append(
                    {
                        "class_name":name,
                        "confidence":conf,
                        "xyxy":xyxy
                    }
                )


                csv_writer.writerow(
                    [
                        frame_id,
                        name,
                        conf,
                        *xyxy,
                        fps
                    ]
                )



            ros.publish(
                {
                    "frame":frame_id,
                    "fps":fps,
                    "detections":detections
                }
            )


            draw_detections(
                frame,
                detections,
                fps
            )


            writer.write(frame)


            cv2.imshow(
                "desk object detection",
                frame
            )



            key=cv2.waitKey(1)&0xff


            if key in [ord("q"),27]:
                break


            if key==ord("e"):

                cv2.imwrite(
                    os.path.join(
                        error_dir,
                        f"error_{frame_id}.jpg"
                    ),
                    frame
                )



    average_fps = (
        fps_sum/max(fps_count,1)
    )


    summary={

        "frames":frame_id,

        "average_fps":average_fps,

        "model":args.model

    }



    with open(
        os.path.join(out_dir,"summary.json"),
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )



    ros.close()

    cap.release()

    writer.release()

    cv2.destroyAllWindows()



    print(
        "Average FPS:",
        average_fps
    )


    print(
        "Results saved:",
        out_dir
    )



if __name__=="__main__":

    main()