import cv2
import numpy as np

# Read the image and convert it to grayscale
image = cv2.imread(r"C:\Users\OMOPC58\Documents\Angha+Divya\augmented_images\moderate2_augmented_0.png")
# image = cv2.resize(image, None, fx=0.9, fy=0.9)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to binary image
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Detect the contours
contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

# Set a threshold to filter small contours (adjust this value as needed)
area_threshold = 100  # Change this based on your needs

# Filter contours by area
filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > area_threshold]

# Print number of remaining contours and their areas
print(f"Number of contours detected: {len(contours)}")
print(f"Number of contours after filtering: {len(filtered_contours)}")

# Draw filtered contours on the original image
image_copy = image.copy()

for i, contour in enumerate(filtered_contours):
    area = cv2.contourArea(contour)
    print(f"Contour {i} has an area of: {area}")

    # Calculate the centroid of the contour to place the text
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Draw the contour number at the centroid position
    cv2.putText(image_copy, f"{i}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Draw the contours on the image
cv2.drawContours(image_copy, filtered_contours, -1, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

# Visualize the results
cv2.imshow('Grayscale Image', gray)
cv2.imshow('Filtered Contours with Numbers', image_copy)
cv2.imshow('Binary Image', binary)

cv2.waitKey(0)
cv2.destroyAllWindows()


