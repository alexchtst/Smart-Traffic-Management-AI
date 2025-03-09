## Usage

### Image Detection

The `image_detect.py` script analyzes a single image and identifies objects using a YOLO model.

```bash
python image_detect.py --image [PATH_TO_IMAGE] --model [PATH_TO_YOLO_MODEL] --output [PATH_TO_SAVE_RESULT]
```

Parameters:
- `--image`: Path to the input image
- `--model`: Path to the YOLO model weights (e.g., yolov8n.pt)
- `--output`: Path where the annotated image will be saved

Example:
```bash
python image_detect.py --image data/traffic.jpg --model models/yolov8n.pt --output results/traffic_detected.jpg
```

### Video Detection

The `video_detect.py` script processes video files and performs object detection on each frame.

```bash
python video_detect.py --video [PATH_TO_VIDEO] --model [PATH_TO_YOLO_MODEL] --output [PATH_TO_SAVE_RESULT]
```

Parameters:
- `--video`: Path to the input video
- `--model`: Path to the YOLO model weights (e.g., yolov8n.pt)
- `--output`: Path where the annotated video will be saved

Example:
```bash
python video_detect.py --video data/traffic.mp4 --model models/yolov8n.pt --output results/traffic_detected.mp4
```