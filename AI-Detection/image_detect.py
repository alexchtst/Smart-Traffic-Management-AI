import cv2
from ultralytics import YOLO
import argparse
import os

def detect_objects_in_image(image_path, model_path, output_path):
    # Load the YOLOv11n model
    model = YOLO(model_path)
    
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Perform object detection
    results = model(image)
    
    # Visualize the results on the image
    annotated_image = results[0].plot()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the annotated image
    cv2.imwrite(output_path, annotated_image)
    
    print(f"Detection completed. Annotated image saved to {output_path}")
    
    # Display the annotated image (optional)
    cv2.imshow('YOLOv11n Detection', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects in image using YOLOv11n')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model weights')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output image')
    args = parser.parse_args()
    
    detect_objects_in_image(args.image, args.model, args.output)