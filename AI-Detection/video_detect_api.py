import cv2
import argparse
from inference_sdk import InferenceHTTPClient

def detect_objects_in_video(video_path, output_path, confidence=0.5):
    # Use your existing API key and model ID
    API_KEY = "pZl3oXtPnrV76f4g5qtS"
    MODEL_ID = "ph-ambulances/1"
    
    # Initialize the Roboflow API client
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=API_KEY
    )
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}")
        
        # Save the frame temporarily to disk (Roboflow API requires a file)
        temp_frame_path = "temp_frame.jpg"
        cv2.imwrite(temp_frame_path, frame)
        
        # Perform object detection using Roboflow API
        try:
            # Call infer without the confidence parameter
            results = client.infer(temp_frame_path, model_id=MODEL_ID)
            
            # Filter predictions based on confidence after receiving results
            predictions = results.get("predictions", [])
            filtered_predictions = [pred for pred in predictions if pred.get("confidence", 0) >= confidence]
            
            # Draw bounding boxes on the frame
            for prediction in filtered_predictions:
                x = prediction["x"]
                y = prediction["y"]
                width_box = prediction["width"]
                height_box = prediction["height"]
                confidence_score = prediction["confidence"]
                class_name = prediction["class"]
                
                # Calculate box coordinates
                x1 = int(x - width_box / 2)
                y1 = int(y - height_box / 2)
                x2 = int(x + width_box / 2)
                y2 = int(y + height_box / 2)
                
                # Draw the box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with confidence
                label = f"{class_name}: {confidence_score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        # Write the frame to output video
        out.write(frame)
        
        # Display the annotated frame (optional)
        cv2.imshow('Roboflow Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects in video using Roboflow API')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output video')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for filtering results (default: 0.5)')
    
    args = parser.parse_args()
    
    detect_objects_in_video(
        args.video,
        args.output,
        args.confidence
    )