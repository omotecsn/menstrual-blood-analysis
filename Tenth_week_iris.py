import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Function to convert RGB values to HEX color
def rgb_to_hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])

# Function to resize image while maintaining aspect ratio
def resize_image(image, width=400):  # Set width to 400 pixels
    aspect_ratio = image.shape[1] / image.shape[0]
    new_height = int(width / aspect_ratio)
    resized_image = cv2.resize(image, (width, new_height))
    return resized_image

# Function to convert normalized bounding box to pixel coordinates
def get_bounding_box(image, bbox):
    h, w, _ = image.shape
    x_center, y_center, box_width, box_height = bbox
    x_min = int((x_center - box_width / 2) * w)
    y_min = int((y_center - box_height / 2) * h)
    x_max = int((x_center + box_width / 2) * w)
    y_max = int((y_center + box_height / 2) * h)
    return x_min, y_min, x_max, y_max



# Function to detect specific blood shades on a pad in an image
def find_blood_shades_on_pad(image, bbox):
    # Get the bounding box region
    x_min, y_min, x_max, y_max = get_bounding_box(image, bbox)
    # Crop the image to the bounding box region
    cropped_image = image[y_min:y_max, x_min:x_max]

    # blur the image
    blurred = cv2.GaussianBlur(cropped_image, (11, 11), 0)
    
    # Convert the cropped image to HSV color space
    image_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # Create a blank mask to keep track of detected areas
    overall_mask = np.zeros(image_hsv.shape[:2], dtype=np.uint8)  # Initially empty
    
    # Define HSV ranges for specific shades
    shades = {
        "bright red//medium red": ([0, 54, 112], [21, 140, 240]),
        "Blood Red/Dark Red": ([0, 169, 98], [179, 255, 164]), 
        "Red-Black-Brown": ([0, 70, 55], [27, 94, 189]), 
        "Black": ([ 0 ,20, 49], [ 22 , 88 ,162]), # Issue with Range
        "lightred":([ 0 ,17, 115], [ 15 , 110 ,255])
    }

    detected_shades = {}
    total_area = 0  # Total area for all detected blood shades
    for shade_name, (lower_bound, upper_bound) in shades.items():
        lower_bound = np.array(lower_bound, dtype=np.uint8)
        upper_bound = np.array(upper_bound, dtype=np.uint8)


        
        # Create a mask for the specific shade in HSV
        shade_mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
        # Exclude already detected areas from this mask
        shade_mask = cv2.bitwise_and(shade_mask, cv2.bitwise_not(overall_mask))
        # Morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        shade_mask = cv2.morphologyEx(shade_mask, cv2.MORPH_CLOSE, kernel)  # Close small holes in the foreground
        shade_mask = cv2.morphologyEx(shade_mask, cv2.MORPH_OPEN, kernel)   # Remove small objects from the foreground

        cv2.imshow('shade_mask', shade_mask)
        cv2.waitKey(0)
        
        # Find contours of the detected regions
        contours, _ = cv2.findContours(shade_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Check if the shade is present in the image
        if contours:
            detected_shades[shade_name] = {
                "hex_colors": [],
                "area": 0  # Initialize area for this shade
            }
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area > 200:  # Adjust contour area threshold
                    # Draw contours on the cropped image
                    cv2.drawContours(cropped_image, [contour], -1, (0, 255, 0), 2)
                    # Mask the detected region to find the average color
                    mask_region = np.zeros_like(shade_mask)
                    cv2.drawContours(mask_region, [contour], -1, 255, thickness=cv2.FILLED)
                    # Find the average color in the masked region
                    mask_area = mask_region > 0
                    avg_color_per_row = np.average(cropped_image[mask_area], axis=0)
                    avg_color = avg_color_per_row.astype(int)
                    hex_color = rgb_to_hex(avg_color)
                    # Store the hex color and update the area for the detected shade
                    detected_shades[shade_name]["hex_colors"].append(hex_color)
                    detected_shades[shade_name]["area"] += contour_area
                    # Get the coordinates of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:  # Check if the area is not zero
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # Place the text next to the boundary (slightly to the right and above)
                        text_position = (cX + 10, cY - 10)  # Adjust these values for placement
                        cv2.putText(cropped_image, shade_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # Add to the overall mask to ensure these areas are not double-counted
            overall_mask = cv2.bitwise_or(overall_mask, shade_mask)
            # Add to the total area
            total_area += detected_shades[shade_name]["area"]
    return cropped_image, detected_shades, total_area
    # return shade_mask, detected_shades, total_area

# Function to display the image using matplotlib
def display_image(image_rgb):
    # Convert the image from BGR (OpenCV default) to RGB (matplotlib format)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("Detected Blood Shades on Pad (Cropped Region)")
    plt.axis('off')  # Hide axes
    plt.show()

# Load the image
image_path = r"C:\Users\OMOPC58\Downloads\dviya\photos for ui\medium red_20ml_71.jpg" # Replace with your image path
image = cv2.imread(image_path)
# Resize the image before processing
image = resize_image(image)
# Load the YOLO model
model_path = os.path.join(r"C:\Users\OMOPC58\Downloads\dviya\weights\best.pt")  # Path to your YOLO model weights
model = YOLO(model_path)
# Detection threshold
threshold = 0.5
# Get image dimensions for normalization
image_height, image_width = image.shape[:2]
# Perform inference
results = model(image)[0]

# Print detected bounding boxes and details
print("Detected objects:")
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    if score > threshold:
        # Draw bounding box on the original image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        # Get the class label from the results.names dictionary
        class_label = results.names[int(class_id)].upper()
        # Prepare the text to display
        text = f"{class_label}: {score:.2f}"
        # Draw the class label and confidence score on the image
        cv2.putText(image, text, (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        # Calculate normalized bounding box coordinates
        normalized_x_center = (x1 + x2) / 2 / image_width
        normalized_y_center = (y1 + y2) / 2 / image_height
        normalized_width = (x2 - x1) / image_width
        normalized_height = (y2 - y1) / image_height
        # Print normalized bounding box coordinates
        print(f"Class: {class_label}, Score: {score:.2f}, Normalized Bounding Box: ({normalized_x_center:.6f}, {normalized_y_center:.6f}, {normalized_width:.6f}, {normalized_height:.6f})")
        # Detect blood shades only inside the bounding box
        bbox = [normalized_x_center, normalized_y_center, normalized_width, normalized_height]
        annotated_image, blood_shades, total_area = find_blood_shades_on_pad(image, bbox)
        # Display the cropped image with detected shades using matplotlib
        display_image(annotated_image)
        # Print the detected shades and their areas
        print("Detected blood shades:")
        for shade, info in blood_shades.items():
            print(f"{shade}: Area = {info['area']} px, Colors = {info['hex_colors']}")
        print(f"Total detected blood area: {total_area} px")

# Display the image with bounding boxes and labels
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()