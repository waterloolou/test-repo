import cv2
import numpy as np
import time
import math

# Define the HSV range for the ball's color (e.g., green ball)
# These values will need adjustment based on your specific ball and lighting conditions

lower_green = np.array([40, 100, 100])
upper_green = np.array([100, 255, 255])

lower_purple = np.array([130, 50, 50])
upper_purple = np.array([180, 255, 255])

colors = [("green", lower_green, upper_green), ("purple", lower_purple, upper_purple)]

# Load the image or start video capture
# frame = cv2.imread('frames/limelight_10.png') # Or use cv2.VideoCapture(0) for webcam

print("processing")
vs = cv2.VideoCapture("videos/limelight_39.mp4")

# allow the camera or video file to warm up
#time.sleep()

idx=0
while True:
  frame = vs.read()[1]
  if frame is None:
    break  
  print(f"frame {idx}")
  idx+=1
  time.sleep(0.2)
  # Blur the image
  blurred = cv2.GaussianBlur(frame, (21, 21), sigmaX=10, sigmaY=10)

  cv2.imshow("blurred", blurred)

  # Convert to HSV color space
  hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

  for color, lower, upper in colors:
    # Create a mask for the specified color range
    mask = cv2.inRange(hsv, lower, upper)

    # Perform morphological operations to clean up the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cv2.imshow(color + " mask", mask)
    
    # Find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Only proceed if at least one contour was found
    if len(cnts) > 0:
        # Find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if color == "green":
          green_center = center
          green_centerY = int(M["m01"] / M["m00"])
          green_centerX = int(M["m10"] / M["m00"])
          #print(f"green center: {green_center}")
        if color == "purple":
          purple_center = center
          purple_centerY = int(M["m01"] / M["m00"])
          purple_centerX = int(M["m10"] / M["m00"])
          #print(f"purple center: {purple_center}")
          try:
            slope = (purple_centerY-green_centerY)/(purple_centerX-green_centerX)
          except ZeroDivisionError:
            slope = math.inf
        
          
          print(f"slope: {slope}")
          angle_radians = math.atan(slope)
          angle_degrees = math.degrees(angle_radians)
          print(f"angle: {angle_degrees}")

        # Only proceed if the radius meets a minimum size
        if radius > 10: # Adjust minimum radius as needed
            # Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

  if green_center and purple_center:          
    cv2.line(frame, green_center, purple_center, (255, 0, 0), 2)


  cv2.imshow("Frame", frame)
  # Exit on 'q' key press
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

vs.release()  
cv2.waitKey(0)
cv2.destroyAllWindows()


