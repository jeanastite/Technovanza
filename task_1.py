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

    calibrated = False
    initial_tvec = None
    initial_rvec = None

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
                # Pose estimation for each detected marker
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeffs)
                
                if rvec is not None and tvec is not None:
                    if calibrated:
                        # Compute relative position
                        relative_tvec = tvec[0][0] - initial_tvec
                        x, y, z = relative_tvec

                        # Compute relative orientation
                        R_current, _ = cv2.Rodrigues(rvec[0])
                        R_initial, _ = cv2.Rodrigues(initial_rvec)
                        R_relative = np.dot(R_current, R_initial.T)

                        sy = np.sqrt(R_relative[0, 0] ** 2 + R_relative[1, 0] ** 2)
                        singular = sy < 1e-6

                        if not singular:
                            roll = np.arctan2(R_relative[2, 1], R_relative[2, 2])
                            pitch = np.arctan2(-R_relative[2, 0], sy)
                            yaw = np.arctan2(R_relative[1, 0], R_relative[0, 0])
                        else:
                            roll = np.arctan2(-R_relative[1, 2], R_relative[1, 1])
                            pitch = np.arctan2(-R_relative[2, 0], sy)
                            yaw = 0

                        # Convert radians to degrees
                        roll_deg = np.degrees(roll)
                        pitch_deg = np.degrees(pitch)
                        yaw_deg = np.degrees(yaw)

                        # Draw axes for visualization
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.1)

                        # Display relative position and orientation
                        cv2.putText(frame, f"ID: {ids[i][0]}",
                                    (int(corner[0][0][0]), int(corner[0][0][1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"X: {x:.2f} m",
                                    (int(corner[0][0][0]), int(corner[0][0][1]) + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame, f"Y: {y:.2f} m",
                                    (int(corner[0][0][0]), int(corner[0][0][1]) + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"Z: {z:.2f} m",
                                    (int(corner[0][0][0]), int(corner[0][0][1]) + 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        cv2.putText(frame, f"Roll: {roll_deg:.2f}°",
                                    (int(corner[0][0][0]), int(corner[0][0][1]) + 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame, f"Pitch: {pitch_deg:.2f}°",
                                    (int(corner[0][0][0]), int(corner[0][0][1]) + 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"Yaw: {yaw_deg:.2f} deg",
                                    (int(corner[0][0][0]), int(corner[0][0][1]) + 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # Print relative position and orientation in the console
                        print(f"Marker ID: {ids[i][0]} | Relative Position - X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}")
                        print(f"Relative Orientation - Roll: {roll_deg:.2f} deg, Pitch: {pitch_deg:.2f}°, Yaw: {yaw_deg:.2f}°")
                    else:
                        # Display message to calibrate
                        cv2.putText(frame, "Press 'c' to set initial position",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow("Box Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c') and ids is not None:  # Calibrate
            initial_tvec = tvec[0][0]
            initial_rvec = rvec[0]
            calibrated = True
            print("Calibration complete. Initial position and orientation recorded.")

    cap.release()
    cv2.destroyAllWindows()


# Camera calibration parameters (example values)
camera_matrix = np.array([[800.0, 0.0, 320.0], 
                          [0.0, 800.0, 240.0], 
                          [0.0, 0.0, 1.0]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

if __name__ == "__main__":
    detect_boxes()
