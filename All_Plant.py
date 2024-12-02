import numpy as np
import cv2

# Replace with your conversion factor (e.g., 0.01 if 100 pixels = 1 inch)
pixel_to_inch = 0.01

# Placeholder function for biomass calculation (to be replaced with actual calculation)
def calculate_biomass(height, diameter):
    # This is a dummy calculation, replace with your actual formula
    return height * diameter * 0.1

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
source = cv2.imread("C:/Users/suchi/OneDrive/Desktop/Plant-Height-Detection-master/D11.png")

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
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through all contours
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)

    # Filter out small contours (noise)
    if area > 100:
        # Calculate the total height of the plant
        total_height_in_pixels = h
        total_height_in_inches = total_height_in_pixels * pixel_to_inch

        # Measure stem diameter using the width of the bounding box
        diameter_in_pixels = w
        diameter_in_inches = diameter_in_pixels * pixel_to_inch

        # Calculate biomass
        biomass = calculate_biomass(total_height_in_inches, diameter_in_inches)

        # Determine the plant growth week
        growth_week = determine_growth_week(total_height_in_inches)

        # Draw a single rectangle from top to bottom
        cv2.rectangle(source, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the results
        cv2.putText(source, "Plant {}".format(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(source, "Height: {:.2f} inches".format(total_height_in_inches), (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(source, "Diameter: {:.2f} inches".format(diameter_in_inches), (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(source, "Biomass: {:.2f}".format(biomass), (x, y-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(source, "Growth Week: {}".format(growth_week), (x, y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
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