#!/usr/bin/env python3
"""Jetson real-time YOLO detection with ROS2 publishing and result saving."""

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
            f"video/x-raw(memory:NVMM), width=(int){args.width}, "
            f"height=(int){args.height}, framerate=(fraction)30/1 ! "
            f"nvvidconv flip-method={args.flip} ! video/x-raw, format=(string)BGRx ! "
            f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


class RosPublisher:
    def __init__(self, enabled, topic):
        self.enabled = enabled
        self.node = None
        self.publisher = None
        self.msg_type = None
        if not enabled:
            return

        try:
            import rclpy
            from std_msgs.msg import String
        except ImportError as exc:
            raise SystemExit("ROS2 Python package rclpy/std_msgs is not available") from exc

        rclpy.init(args=None)
        self.rclpy = rclpy
        self.msg_type = String
        self.node = rclpy.create_node("desk_object_detector")
        self.publisher = self.node.create_publisher(String, topic, 10)

    def publish(self, payload):
        if not self.enabled:
            return
        msg = self.msg_type()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.publisher.publish(msg)
        self.rclpy.spin_once(self.node, timeout_sec=0)

    def close(self):
        if not self.enabled:
            return
        self.node.destroy_node()
        self.rclpy.shutdown()


def draw_detections(frame, detections, fps):
    palette = [(56, 56, 255), (49, 210, 207), (134, 219, 61), (255, 194, 0)]
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        cls_id = det["class_id"]
        color = palette[cls_id % len(palette)]
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, y0), (x1 + tw + 8, y0 + th + 8), color, -1)
        cv2.putText(frame, label, (x1 + 4, y0 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    hud = f"FPS {fps:.1f} | q quit | e save error"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(frame, hud, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--csi", action="store_true")
    parser.add_argument("--flip", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--topic", default="/desk_object_detections")
    parser.add_argument("--no-ros", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(os.path.join(args.out, run_id))
    err_dir = os.path.join(out_dir, "errors")
    os.makedirs(err_dir, exist_ok=True)

    model = YOLO(args.model)
    cap = build_capture(args)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")

    video_path = os.path.join(out_dir, "result.mp4")
    csv_path = os.path.join(out_dir, "detections.csv")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, args.fps, (args.width, args.height))
    ros = RosPublisher(not args.no_ros, args.topic)

    frame_id = 0
    fps = 0.0
    last = time.time()
    print(f"Model: {os.path.abspath(args.model)}")
    print(f"ROS2 topic: {args.topic if not args.no_ros else 'disabled'}")
    print(f"Saving results to: {out_dir}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        log.writerow(["frame", "time", "class_id", "class_name", "confidence",
                      "x1", "y1", "x2", "y2", "fps"])

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed")
                    break

                frame_id += 1
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 0.0
                last = now

                result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                       device=0, verbose=False)[0]
                detections = []
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    name = result.names.get(cls_id, str(cls_id))
                    det = {
                        "class_id": cls_id,
                        "class_name": name,
                        "confidence": conf,
                        "xyxy": xyxy,
                    }
                    detections.append(det)
                    log.writerow([frame_id, f"{now:.3f}", cls_id, name, f"{conf:.4f}",
                                  *xyxy, f"{fps:.2f}"])

                payload = {
                    "frame": frame_id,
                    "stamp": now,
                    "fps": fps,
                    "detections": detections,
                }
                ros.publish(payload)

                draw_detections(frame, detections, fps)
                writer.write(frame)
                cv2.imshow("desk object detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("e"):
                    err_path = os.path.join(err_dir, f"error_{frame_id:06d}.jpg")
                    cv2.imwrite(err_path, frame)
                    print(f"Saved error case: {err_path}")
        finally:
            ros.close()
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

    print(f"Video: {video_path}")
    print(f"CSV: {csv_path}")
    print(f"Errors: {err_dir}")


if __name__ == "__main__":
    main()
