import cv2
import numpy as np
import cv2.aruco as aruco

def detect_boxes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image. Exiting.")
        return

    # Define the ArUco dictionary and detector parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, detector_params)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        # Draw and process only markers with ID 1
        for i, corner in enumerate(corners):
            if ids[i][0] == 10:  # Check if the marker ID is 1
                aruco.drawDetectedMarkers(image, [corner], np.array([[10]]))  # Ensure IDs are passed as a NumPy array

                # Pose estimation for the marker
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeffs)
                
                if rvec is not None and tvec is not None:
                    # Extract position
                    x, y, z = tvec[0][0]

                    # Compute angular deviation (degree of rotation)
                    angle_rad = np.linalg.norm(rvec[0])
                    angle_deg = np.degrees(angle_rad)

                    # Draw axes for visualization
                    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.1)

                    # Display position and angular deviation
                    cv2.putText(image, f"ID: {ids[i][0]}",
                                (int(corner[0][0][0]), int(corner[0][0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(image, f"X: {x:.2f} m",
                                (int(corner[0][0][0]), int(corner[0][0][1]) + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(image, f"Y: {y:.2f} m",
                                (int(corner[0][0][0]), int(corner[0][0][1]) + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(image, f"Z: {z:.2f} m",
                                (int(corner[0][0][0]), int(corner[0][0][1]) + 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(image, f"Angle: {angle_deg:.2f} deg",
                                (int(corner[0][0][0]), int(corner[0][0][1]) + 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Print position and angular deviation in the console
                    print(f"Marker ID: {ids[i][0]} | Position - X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}")
                    print(f"Angular Deviation: {angle_deg:.2f} deg")
    else:
        print("No markers detected in the image.")

    # Display the resulting image
    cv2.imshow("Image Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Camera calibration parameters (example values)
camera_matrix = np.array([[800.0, 0.0, 320.0], 
                          [0.0, 800.0, 240.0], 
                          [0.0, 0.0, 1.0]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

if __name__ == "__main__":
    image_path = r"C:\Users\shard\Music\TechNoVanza\Technovanza\Test_Images\ref_checking.png"  # Replace with the path to your image
    detect_boxes(image_path)