import cv2
import numpy as np

drawing = False
current_blob = []
all_blobs = []

def draw_blob(event, x, y, flags, param):
    global drawing, current_blob
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_blob = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_blob.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_blob.append((x, y))

def get_user_drawn_features(frame):
    """
    Allow the user to draw multiple blobs and extract feature points from those regions.
    """
    global current_blob, all_blobs
    clone = frame.copy()
    cv2.namedWindow("Draw Blobs - ESC=Done, Enter=Finish Blob")
    cv2.setMouseCallback("Draw Blobs - ESC=Done, Enter=Finish Blob", draw_blob)

    while True:
        temp = clone.copy()
        # Draw all previously completed blobs
        for blob in all_blobs:
            if len(blob) >= 3:
                cv2.polylines(temp, [np.array(blob)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw current blob in progress
        if len(current_blob) > 1:
            cv2.polylines(temp, [np.array(current_blob)], isClosed=False, color=(0, 255, 255), thickness=2)

        cv2.imshow("Draw Blobs - ESC=Done, Enter=Finish Blob", temp)
        key = cv2.waitKey(1)

        if key == 13:  # Enter to finish current blob
            if len(current_blob) >= 3:
                all_blobs.append(current_blob.copy())
            current_blob = []
        elif key == 27:  # ESC to finish drawing
            break

    cv2.destroyWindow("Draw Blobs - ESC=Done, Enter=Finish Blob")

    # Create mask and extract features
    mask = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    for blob in all_blobs:
        if len(blob) >= 3:
            cv2.fillPoly(mask, [np.array(blob)], 255)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=300, qualityLevel=0.3, minDistance=7)
    return p0

def compute_optical_flow_webcam(cap, p0, gray_old):
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_old, gray_new, p0, None, **lk_params)

        if p1 is not None and st is not None:
            for new, old in zip(p1[st == 1], p0[st == 1]):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            p0 = p1.reshape(-1, 1, 2)
            gray_old = gray_new.copy()

        cv2.imshow('Webcam Optical Flow', frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        cap.release()
        return

    gray_old = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    p0 = get_user_drawn_features(first_frame)

    if p0 is None:
        print("No features detected. Exiting.")
        cap.release()
        return

    compute_optical_flow_webcam(cap, p0, gray_old)

if __name__ == "__main__":
    main()
