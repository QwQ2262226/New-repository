# 桌面物体检测系统（YOLOv8 + Jetson + ROS2）

基于 YOLOv8 的桌面物体目标检测项目。

本项目完成： 数据采集 → 自动预标注 → 人工复核 → 数据集划分 → YOLOv8训练
→ Jetson实时部署。

最终检测类别： - mouse（鼠标） - keyboard（键盘）

## 项目流程

    数据采集
    ↓
    capture_images.py
    ↓
    auto_label.py 自动预标注
    ↓
    labelImg人工修正
    ↓
    check_labels.py 数据检查
    ↓
    split_dataset.py 数据划分
    ↓
    YOLOv8训练
    ↓
    best_mouse_keyboard.pt
    ↓
    Jetson实时检测
    ↓
    ROS2发布结果

## 工具说明

### capture_images.py

-   摄像头采集图片
-   自动连拍
-   手动拍摄
-   撤销错误图片

### auto_label.py

-   YOLO预训练模型自动标注
-   输出YOLO格式标签
-   配合labelImg人工修正

### check_labels.py

-   检查标签格式
-   检查类别编号
-   检查坐标范围
-   检查异常框

### split_dataset.py

-   划分train/val/test
-   自动生成data.yaml

## 环境

PC: - Python \>= 3.9 - YOLOv8 - PyTorch - OpenCV

Jetson: - Ubuntu 22.04 - JetPack 6.x - ROS2

安装：

``` bash
pip install -r requirements.txt
```

## 数据集制作

采集：

``` bash
python tools/capture_images.py --class mouse --limit 100 --out raw
python tools/capture_images.py --class keyboard --limit 100 --out raw
```

自动标注：

``` bash
python tools/auto_label.py --images raw --out labeled --classes mouse keyboard
```

划分数据集：

``` bash
python tools/split_dataset.py --input labeled --output dataset --classes mouse keyboard --ratio 0.7 0.2 0.1
```

## 模型训练

``` bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

训练模型：

    best_mouse_keyboard.pt

## Jetson部署

运行：

``` bash
python tools/jetson_detect_ros2.py --model best_mouse_keyboard.pt --csi
```

支持： - 实时摄像头检测 - 检测框显示 - 类别和置信度显示 - ROS2消息发布 -
视频和CSV结果保存

ROS2 Topic:

    /desk_object_detections

## 项目结构

    .
    ├── README.md
    ├── requirements.txt
    ├── best_mouse_keyboard.pt
    ├── tools
    │   ├── capture_images.py
    │   ├── auto_label.py
    │   ├── check_labels.py
    │   ├── split_dataset.py
    │   ├── common.py
    │   └── jetson_detect_ros2.py
    ├── results
    └── report

## 类别编号

    0 -> mouse
    1 -> keyboard

自动标注结果需要人工检查并修正。
