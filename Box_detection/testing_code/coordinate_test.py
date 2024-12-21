import cv2
import numpy as np

def detect_edges(image_path):
    """
    Performs edge detection on an image using Canny.

    Args:
        image_path: Path to the image file.

    Returns:
        The edge-detected image (or None if there's an error).
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 30, 200)  # Adjust thresholds as needed

    return edges

def display_image(window_name, image):
    """Displays an image in a named window."""
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = r"C:\Users\shard\Music\TechNoVanza\Technovanza\Box_detection\Images\box1.jpeg"  # Replace with your image path
    edges = detect_edges(image_path)

    if edges is not None:
        display_image("Edge Detection", edges)

        # Optionally save the edge image
        # cv2.imwrite("edges.jpg", edges)