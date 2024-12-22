import cv2
import numpy as np

def detect_boxes(image_path):
    """
    Detects and displays the boxes in an image using OpenCV.

    Args:
        image_path: Path to the image file.

    Returns:
        The number of detected boxes.
    """

    # Load the image
    img = cv2.imread(image_path)
    original_img = img.copy() # Keep a copy for drawing
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return 0

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 200)

    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on shape characteristics (e.g., approximate rectangle)
    boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # Check for quadrilaterals
            boxes.append(approx) # Append the approximated contour

    # Draw the detected boxes on the original image
    for box in boxes:
        cv2.drawContours(original_img, [box], -1, (0, 255, 0), 3) # Green boxes

    # Count the number of detected boxes
    num_boxes = len(boxes)

    # Display the image with detected boxes
    cv2.imshow("Detected Boxes", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return num_boxes

if __name__ == "__main__":
    image_path = r"Technovanza\Box_detection\Images\box3.jpeg"  # Replace with the actual image path
    num_boxes = detect_boxes(image_path)
    print("Number of boxes detected:", num_boxes)