import cv2
import torch
from ultralytics import YOLO

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

# Example usage
video_path = 'videoplayback.mp4'
model_path = 'ambulance5mar.pt'  # Replace with actual path to YOLOv11n weights
output_path = 'output_video.mp4'

detect_objects_in_video(video_path, model_path, output_path)
