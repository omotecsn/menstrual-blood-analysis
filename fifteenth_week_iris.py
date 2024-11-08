import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2

# Load the YOLO model
model_path = r"C:\Users\OMOPC58\Downloads\dviya\weights\best.pt"
model = YOLO(model_path)

# Function to process the image with YOLO and return the result
def process_image(image):
    # Run inference on the input image
    results = model(image)

    # Create a copy of the image to draw bounding boxes on
    img_copy = np.array(image)

    # Process each result (since results is a list, iterate over it)
    for result in results:
        # Draw bounding boxes and labels
        for box in result.boxes:
            # Extract bounding box coordinates (ensure correct tensor handling)
            xyxy = box.xyxy.cpu().numpy()  # Convert to numpy array
            x1, y1, x2, y2 = xyxy[0]  # Extract individual coordinates
            
            class_id = int(box.cls)
            confidence = round(float(box.conf), 2)

            # Draw the bounding box
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Put the label with class and confidence
            label = f"Class: {class_id}, Conf: {confidence}"
            cv2.putText(img_copy, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the image back to RGB for displaying with Gradio
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    
    # Return the processed image
    return img_rgb

# Create a Gradio interface
interface = gr.Interface(
    fn=process_image,  # Function to run when an image is uploaded
    inputs=gr.Image(type="pil"),  # Input type: PIL image
    outputs=gr.Image(type="numpy"),  # Output type: processed image
    live=True  # Enable live feedback (optional)
)

# Launch the Gradio app
interface.launch()
