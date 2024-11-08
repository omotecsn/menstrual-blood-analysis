

import cv2
import numpy as np

# Function to convert RGB values to HEX color
def rgb_to_hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])

# Function to detect specific blood shades on a pad in an image
def find_blood_shades_on_pad(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.resize(image, (700, 700))
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV

    # Define HSV ranges for specific shades, merging Bright Red and Dark Red/Maroon into one "Red" shade
    shades = {
        "Red": ([0, 50, 50], [10, 255, 255]),        
        "lightred" :([1,151,172],[21,83,207]) ,   # Combined Red range
        "dark_red":([170,42,50],[180,255,255]),
        "Pink": ([160, 50, 100], [180, 255, 255]),        # Pink tones
        "Brown": ([10, 50, 20], [20, 150, 100]),          # Brownish tones
        "Black": ([0, 0, 0], [180, 255, 50]),             # Black shades with very low brightness
    }

    detected_shades = {}

    for shade_name, (lower_bound, upper_bound) in shades.items():
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)

        # Create a mask for the specific shade in HSV
        mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
        
        # Morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes in the foreground
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small objects from the foreground

        # Find contours of the detected regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if the shade is present in the image
        if contours:
            detected_shades[shade_name] = []

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Adjusted area filter
                    # Fill the detected regions with the original color from the mask
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)

                    # Find the average color in the masked region
                    mask_area = mask > 0
                    avg_color_per_row = np.average(image[mask_area], axis=0)
                    avg_color = avg_color_per_row.astype(int)  # Convert to integer
                    hex_color = rgb_to_hex(avg_color)

                    # Store the hex color for each detected shade
                    detected_shades[shade_name].append(hex_color)

    return image, detected_shades

# Function to display the image
def display_image(image_rgb):
    # Display the final image with detected shades
    cv2.imshow("Detected Blood Shades on Pad", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = r'C:\Users\OMOPC58\Documents\Angha+Divya\darkred.jpg'  # Replace with your image path
annotated_image, blood_shades = find_blood_shades_on_pad(image_path)

# Display the image with detected shades
if blood_shades:
    display_image(annotated_image)
    
    # Print the detected shades and their hex codes
    print("Detected Blood Shades and their Hex Codes:")
    for shade_name, hex_colors in blood_shades.items():
        for hex_color in hex_colors:
            print(f"{shade_name}: {hex_color}")
else:
    print("No blood shades detected in the image.")
