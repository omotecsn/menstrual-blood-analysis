import cv2
import numpy as np

def callback(x):
    pass

# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
cv2.createTrackbar('Low H', 'image', 0, 179, callback)
cv2.createTrackbar('High H', 'image', 179, 179, callback)
cv2.createTrackbar('Low S', 'image', 0, 255, callback)
cv2.createTrackbar('High S', 'image', 255, 255, callback)
cv2.createTrackbar('Low V', 'image', 0, 255, callback)
cv2.createTrackbar('High V', 'image', 255, 255, callback)

# Load the image
image = cv2.imread(r"C:\Users\OMOPC58\Downloads\dviya\downloaded_images\light red_16ml_63.jpg")
image = cv2.resize(image, (400, 400))
# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

while True:
    # Get current positions of all trackbars
    low_h = cv2.getTrackbarPos('Low H', 'image')
    high_h = cv2.getTrackbarPos('High H', 'image')
    low_s = cv2.getTrackbarPos('Low S', 'image')
    high_s = cv2.getTrackbarPos('High S', 'image')
    low_v = cv2.getTrackbarPos('Low V', 'image')
    high_v = cv2.getTrackbarPos('High V', 'image')

    # Set the lower and upper HSV range according to the values of the trackbars
    lower_bound = np.array([low_h, low_s, low_v])
    upper_bound = np.array([high_h, high_s, high_v])

    # Threshold the HSV image to get only selected colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    # Display result
    cv2.imshow('image', res)

    # Break the loop when user hits 'esc'
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()