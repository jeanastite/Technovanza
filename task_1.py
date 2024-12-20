import cv2
import numpy as np
import cv2.aruco as aruco

def detect_boxes():
    # Load camera feed
    cap = cv2.VideoCapture(0)

    # Define the ArUco dictionary and detector parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, detector_params)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            # Draw detected markers
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i, corner in enumerate(corners):
                c = corner[0]
                center_x = int((c[0][0] + c[2][0]) / 2)
                center_y = int((c[0][1] + c[2][1]) / 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

                # Pose estimation
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeffs)
                if rvec is not None and tvec is not None:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.1)

        # Display the resulting frame
        cv2.imshow("Box Detection", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Camera calibration parameters (to be updated with actual values)
camera_matrix = np.array([[800.0, 0.0, 320.0], 
                          [0.0, 800.0, 240.0], 
                          [0.0, 0.0, 1.0]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

if __name__ == "__main__":
    detect_boxes()
