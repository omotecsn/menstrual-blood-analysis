import os
import cv2
import numpy as np
import gradio as gr
import pandas as pd
import joblib
from ultralytics import YOLO

# Load the trained models
volume_model = joblib.load('volume_model.joblib')
viscosity_model = joblib.load('viscosity_model.joblib')
# Function to convert RGB values to HEX color
def rgb_to_hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])

# Function to resize image while maintaining aspect ratio
def resize_image(image, width=400):
    aspect_ratio = image.shape[1] / image.shape[0]
    new_height = int(width / aspect_ratio)
    resized_image = cv2.resize(image, (width, new_height))
    return resized_image

# Function to detect specific blood shades and apply correct mask colors
def find_blood_shades_on_pad(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image[y_min:y_max, x_min:x_max]
    blurred = cv2.GaussianBlur(cropped_image, (11, 11), 0)
    image_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    overall_mask = np.zeros(image_hsv.shape[:2], dtype=np.uint8)

    shades = {
        "bright red//medium red": ([0, 54, 112], [21, 140, 240]),
        "Blood Red/Dark Red": ([0, 169, 98], [179, 255, 164]), 
        "Red-Black-Brown": ([0, 70, 55], [27, 94, 189]), 
        "Black": ([0, 20, 49], [22, 88, 162]),
        "light red": ([0, 17, 115], [15, 110, 255])
    }

    shade_images = []  # Store images of each shade detected
    all_contours_image = cropped_image.copy()  # Copy for contours
    detected_contours = False
    detected_areas = {}  # Store detected areas for each shade
    total_area = 0  # Store total detected area

    for shade_name, (lower_bound, upper_bound) in shades.items():
        lower_bound = np.array(lower_bound, dtype=np.uint8)
        upper_bound = np.array(upper_bound, dtype=np.uint8)
        shade_mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
        shade_mask = cv2.bitwise_and(shade_mask, cv2.bitwise_not(overall_mask))

        kernel = np.ones((3, 3), np.uint8)
        shade_mask = cv2.morphologyEx(shade_mask, cv2.MORPH_CLOSE, kernel)
        shade_mask = cv2.morphologyEx(shade_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(shade_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            valid_contours = False
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area > 200:  # Only consider larger contours
                    valid_contours = True
                    cv2.drawContours(cropped_image, [contour], -1, (0, 255, 0), 2)

                    # Store detected area for this shade
                    detected_areas[shade_name] = detected_areas.get(shade_name, 0) + contour_area
                    total_area += contour_area  # Add to the total area

            if valid_contours:
                detected_contours = True

                # Create a mask where detected areas are set to black
                shade_image = cv2.bitwise_and(cropped_image, cropped_image, mask=shade_mask)
                blacked_out_shade = cropped_image.copy()
                blacked_out_shade[shade_mask > 0] = (0, 0, 0)  # Black where the mask is applied
                shade_images.append(blacked_out_shade)
            else:
                shade_images.append(np.zeros_like(cropped_image))  # Return blank if no valid contours
        overall_mask = cv2.bitwise_or(overall_mask, shade_mask)

    # Prepare the output string for detected shades and their areas
    detected_info = "Detected blood shades:\n : "
    for shade, area in detected_areas.items():
        color_hex = rgb_to_hex((0, 0, 0))  # Placeholder for color; adjust as needed for real detection
        detected_info += f"{shade}: Area = {area:.1f} px, Colors = [{color_hex}]\n"

    # Add total area to the detected info
    detected_info += f"Total detected blood area: {total_area:.1f} px\n"

    # Final image with all contours if any were detected
    if detected_contours:
        cv2.drawContours(all_contours_image, contours, -1, (0, 255, 0), 2)

    shade_images = shade_images + [np.zeros_like(cropped_image)] * (5 - len(shade_images))

    return cropped_image, shade_images, all_contours_image, detected_info, total_area  # Return detected info and total area

# Function to classify the total detected area
def classify_area(total_area):
    if total_area < 15000:
        return "Low"
    elif total_area > 25000:
        return "High"
    else:
        return "Medium"

# Function to process the uploaded image
def process_image(uploaded_image):
    image = cv2.imread(uploaded_image)  # Use the file path directly
    image = resize_image(image)

    model_path = r"C:\Users\OMOPC58\Downloads\dviya\weights\best.pt"  # Path to YOLO model weights
    model = YOLO(model_path)
    threshold = 0.5  # Detection threshold

    results = model(image)[0]
    images_to_display = []  # Collect images to return to Gradio
    detected_info = ""  # Initialize detected info
    total_detected_area = 0  # Initialize total detected area

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            annotated_image, shade_images, all_contours_image, detected_info_temp, total_area = find_blood_shades_on_pad(image, bbox)

            # Append images for each detected shade
            images_to_display.extend(shade_images)
            detected_info += detected_info_temp  # Accumulate detected info
            total_detected_area += total_area  # Accumulate total area
# Classify total area
    area_classification = classify_area(total_detected_area)
    detected_info += f"Area Classification: {area_classification}\n"

# M# Sample max standard deviation for volume confidence normalization
# You may want to set this based on observed max deviations in the data
    max_std = 0.1  # Adjust this based on your data

# Make predictions for volume and viscosity
    new_data = pd.DataFrame({'area': [total_detected_area]})  # Use total detected area as input
    predicted_volume = volume_model.predict(new_data)

# Compute volume confidence by normalizing standard deviation
    volume_std = np.std([tree.predict(new_data) for tree in volume_model.estimators_], axis=0)
# Ensure volume confidence is capped between 1% and 100%
    volume_confidence = np.clip((1 - volume_std / max_std) * 100, 1, 100)

    predicted_viscosity = viscosity_model.predict(new_data)
    viscosity_probabilities = viscosity_model.predict_proba(new_data)

# Ensure viscosity confidence is between 1% and 100%
    viscosity_confidence = np.clip(np.max(viscosity_probabilities, axis=1) * 100, 1, 100)

# Add predictions to detected info with confidence displayed as percentage from 1 to 100
    detected_info = f"Predicted Volume: {predicted_volume[0]:.2f} mL, Volume Confidence: {volume_confidence[0]:.2f}%\n"
    detected_info += f"Predicted Viscosity: {predicted_viscosity[0]}, Viscosity Confidence: {viscosity_confidence[0]:.2f}%\n"

    print(detected_info)

    # Add the final contour image to the list of images to return
    images_to_display.append(all_contours_image)

    # Ensure we have exactly six images to return (five shades + one contours)
    return (
        annotated_image.astype(np.uint8),
        *images_to_display[:5],  
        all_contours_image.astype(np.uint8),  
        detected_info,
        total_detected_area  
    )
def create_interface_page_2():
    folder_path = r"C:\Users\OMOPC58\Downloads\dviya\photos for ui"
    images = load_images_from_folder(folder_path)

    # Function to display images with indices and return description based on selected number
    def display_images_and_description(selected_number):
        # Display images
        indexed_images = []
        for index, image_path in enumerate(images):
            indexed_images.append((image_path, f"Image {index + 1}"))  # Tuple of (image_path, title)
        
        # Description based on selected number
        descriptions = {
            1: "you haved selected image 13.",
            2: "you haved selected image 2",
            3: "you haved selected image 3",
            4: "you haved selected image 4",
            5: "you haved selected image 5",
            6: "you haved selected image 6",
            7: "you haved selected image 7",
            8: "you haved selected image 8",
            9: "you haved selected image 9",
            10: "you haved selected image 10",
            11: "you haved selected image 11",
            12: "you haved selected image 12",
            13: "you haved selected image 13"
        }
        
        description = descriptions.get(selected_number, "Invalid input. Please enter a number from 1 to 13.")
        return indexed_images, description  # Return both images and description

    # Create Gradio interface
    return gr.Interface(
        fn=display_images_and_description,  # Single function that returns both images and description
        inputs=gr.Number(label="Select a Number from 1 to 13", minimum=1, maximum=13),  # Number input
        outputs=[
            gr.Gallery(label="Images from Folder"),  # Display images in a gallery
            gr.Textbox(label="Area Description")  # Textbox for displaying the description
        ],
        live=True,  # Enable live updates for inputs
        title="Disease detection using pre-clicked photos",
        description="View images and select a number to view the corresponding disease description."
    )


def main():
    # Create two Gradio pages and launch the app
    
    folder_image_interface = create_interface_page_2()
    
    # Create Gradio tabs to toggle between both functionalities
    demo = gr.TabbedInterface([ folder_image_interface], ["Pre-clicked photos"])
    demo.launch()

if __name__ == "__main__":
    main()
