import numpy as np
import cv2

# Replace with your conversion factor (e.g., 0.01 if 100 pixels = 1 inch)
pixel_to_inch = 0.01

# Placeholder function for biomass calculation (to be replaced with actual calculation)
def calculate_biomass(height, diameter):
    # This is a dummy calculation, replace with your actual formula
    return height * diameter * 0.1  # Example calculation

# Define height thresholds for plant growth weeks
height_to_week = {
    (0, 4): "1st Week",
    (4, 8): "2nd Week",
    (8, 12): "3rd Week",
    (12, float('inf')): "4th Week or More"
}

def determine_growth_week(height):
    for (min_height, max_height), week in height_to_week.items():
        if min_height <= height < max_height:
            return week
    return "Unknown Week"

# Load the image
source = cv2.imread("C:/Users/suchi/OneDrive/Desktop/Plant-Height-Detection-master/D14.png")

# Check if the image was successfully loaded
if source is None:
    print("Error: Image not found or unable to load.")
    exit()

# Get image dimensions
height, width = source.shape[:2]
print("Image resolution:", height, "x", width)

# Resize the image for better visibility
source = cv2.resize(source, (int(width*1.5), int(height*1.5)))

# Convert the image to HSV color space
hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)

# Define the color range for masking
lower_color = np.array([28, 70, 133])
upper_color = np.array([70, 255, 255])

# Create the mask
mask = cv2.inRange(hsv, lower_color, upper_color)

# Perform bitwise AND operation to extract the plant
res = cv2.bitwise_and(source, source, mask=mask)

# Find contours
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming it's the entire plant)
max_area = 0
best_cnt = None

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > max_area:
        max_area = area
        best_cnt = cnt

if best_cnt is not None:
    # Draw the yellow contours of the plant
    cv2.drawContours(source, [best_cnt], -1, (0, 255, 255), 2)  # Yellow color

    # Draw a single rectangle from top to bottom
    x, y, w, h = cv2.boundingRect(best_cnt)
    cv2.rectangle(source, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Convert height from pixels to inches
    height_in_pixels = h
    height_in_inches = height_in_pixels * pixel_to_inch

    # Measure stem diameter using the width of the bounding box
    diameter_in_pixels = w
    diameter_in_inches = diameter_in_pixels * pixel_to_inch

    # Calculate biomass
    biomass = calculate_biomass(height_in_inches, diameter_in_inches)

    # Determine the plant growth week
    growth_week = determine_growth_week(height_in_inches)

    # Set initial position for the text at the top left
    x_offset = 10
    text_y_offset = 40

    # Display the results
    font_scale = 2
    font_thickness = 3
    line_gap = 50  # Adjust this value to increase or decrease the gap between lines
    cv2.putText(source, "Height: {:.2f} inches".format(height_in_inches), (x_offset, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(source, "Diameter: {:.2f} inches".format(diameter_in_inches), (x_offset, text_y_offset + line_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(source, "Biomass: {:.2f} grams".format(biomass), (x_offset, text_y_offset + 2 * line_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(source, "Growth Week: {}".format(growth_week), (x_offset, text_y_offset + 3 * line_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

# Display the results
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 800, 600)  # Set the window size to 800x600
cv2.imshow('frame', source)

cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('mask', 400, 300)  # Set the window size to 400x300
cv2.imshow('mask', mask)

# Wait for a key press or 5 seconds
cv2.waitKey(0)  # or cv2.waitKey(5000) for 5 seconds
cv2.destroyAllWindows()
