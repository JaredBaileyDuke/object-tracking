"""
ARUCO MARKER DETECTION WITH OPENCV

- This script detects ArUco markers.
- It uses OpenCV’s aruco module to identify and draw markers in real time.
"""

import cv2
import cv2.aruco as aruco
import numpy as np

def initialize_camera():
    """Initializes the webcam and checks for availability."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")
    return cap

def setup_aruco_detector():
    """Sets up the ArUco marker detector and dictionary."""
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    return detector

def get_camera_calibration():
    """Provides dummy camera calibration parameters."""
    camera_matrix = np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0, 0, 1]], dtype=float)
    dist_coeffs = np.zeros((5, 1))  # Assuming no distortion
    return camera_matrix, dist_coeffs

def draw_pyramid(frame, rvec, tvec, camera_matrix, dist_coeffs):
    """Draws a regular pyramid on the detected marker."""
    size = 0.03  # Base size in meters
    h = size * 1.5  # Height of the pyramid

    # Define 3D points for pyramid base and tip
    base = np.float32([[-size, -size, 0],
                       [ size, -size, 0],
                       [ size,  size, 0],
                       [-size,  size, 0]])
    tip = np.float32([[0, 0, h]])
    pts3d = np.vstack((base, tip))

    # Project 3D points to image plane
    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, camera_matrix, dist_coeffs)
    pts2d = np.int32(pts2d).reshape(-1, 2)

    # Draw pyramid edges
    for i in range(4):
        cv2.line(frame, pts2d[i], pts2d[(i+1)%4], (0, 255, 255), 2)
        cv2.line(frame, pts2d[i], pts2d[4], (0, 255, 255), 2)

def draw_cube(frame, rvec, tvec, camera_matrix, dist_coeffs):
    """Draws a cube on the detected marker."""
    size = 0.1  # Cube size larger than marker
    half = size / 2.0

    # Define 3D points of the cube
    pts3d = np.float32([
        [-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0],
        [-half, -half, size], [half, -half, size], [half, half, size], [-half, half, size]
    ])

    # Project 3D points to image plane
    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, camera_matrix, dist_coeffs)
    pts2d = np.int32(pts2d).reshape(-1, 2)

    # Draw cube edges
    for i in range(4):
        cv2.line(frame, pts2d[i], pts2d[(i+1)%4], (255, 0, 255), 2)       # Bottom face
        cv2.line(frame, pts2d[i+4], pts2d[((i+1)%4)+4], (255, 0, 255), 2) # Top face
        cv2.line(frame, pts2d[i], pts2d[i+4], (255, 0, 255), 2)           # Vertical edges

def detect_and_draw_markers(frame, detector, camera_matrix, dist_coeffs):
    """Detects ArUco markers and draws corresponding 3D objects."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i]
            tvec = tvecs[i]
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            if marker_id == 8:
                draw_pyramid(frame, rvec, tvec, camera_matrix, dist_coeffs)
            elif marker_id == 9:
                draw_cube(frame, rvec, tvec, camera_matrix, dist_coeffs)

def main():
    """Main loop to run ArUco marker detection and object rendering."""
    print("Press 'q' to quit")
    cap = initialize_camera()
    detector = setup_aruco_detector()
    camera_matrix, dist_coeffs = get_camera_calibration()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_and_draw_markers(frame, detector, camera_matrix, dist_coeffs)
        cv2.imshow("3D Object on ArUco Marker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()