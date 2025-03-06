import cv2
from ultralytics import YOLO
import argparse

def detect_objects_in_video(video_path, model_path, output_path):
    # Load the YOLOv11n model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform object detection
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Write the frame to output video
        out.write(annotated_frame)
        
        # Display the annotated frame (optional)
        cv2.imshow('YOLOv11n Detection', annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects in video using YOLOv11n')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model weights')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output video')
    
    args = parser.parse_args()
    
    detect_objects_in_video(args.video, args.model, args.output)
