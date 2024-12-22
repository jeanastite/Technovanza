import cv2
import numpy as np

 
image_path = r"Technovanza\task_1\given_task_soln\images\box3.jpg"
image = cv2.imread(image_path)

 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

 
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

 
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

 
cv2.imshow("Thresholded Image", thresh)
cv2.waitKey(0)
 
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 
output_image = image.copy()

 
box_count = 0

for contour in contours:
    
    if cv2.contourArea(contour) > 290:   
         
        rect = cv2.minAreaRect(contour)
        box_count += 1

        # Get box points and draw
        box_points = cv2.boxPoints(rect)
        box_points = np.int32(box_points)
        cv2.drawContours(output_image, [box_points], 0, (0, 255, 0), 2)

       
        angle = rect[2]
        center = tuple(np.int32(rect[0]))
        cv2.putText(output_image, f"Angle: {angle:.2f}", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

 
cv2.putText(output_image, f"Boxes: {box_count}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

 
cv2.imshow("Detected Boxes", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

 
output_path = "output/detected_boxes_improved.jpg"
cv2.imwrite(output_path, output_image)
print(f"Processed image saved to: {output_path}")