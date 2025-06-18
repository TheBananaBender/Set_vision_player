import cv2
import numpy as np

def order_box_points(pts):
    """
    Given 4 points of a rotated rectangle (unordered), return them in order:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    # Sum and diff help identify corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # top-left (smallest sum)
    rect[2] = pts[np.argmax(s)]       # bottom-right (largest sum)
    rect[1] = pts[np.argmin(diff)]    # top-right (smallest diff)
    rect[3] = pts[np.argmax(diff)]    # bottom-left (largest diff)

    return rect

def detect_cards_in_frame(frame, debug=False):
    """
    Finds all card‐like quadrilaterals in a frame and returns a list of
    4-point boxes (each as a 4×2 array of float32).
    """
    # 1) Resize for speed (width=600)
    height, width = frame.shape[:2]
    scale = 600.0 / width
    newW = 600
    newH = int(height * scale)
    resized = cv2.resize(frame, (newW, newH))
    ratio = width / float(newW)  # to scale boxes back to original if needed

    # 2) Grayscale + blur
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) Edge detection
    #    Lower and upper thresholds may need some tuning (50,150 are common defaults)
    edges = cv2.Canny(blur, 25, 85)

    # 4) Morphological closing (dilate then erode) to close small gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 5) Find contours
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_boxes = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2000:
            # skip small contours (noise)
            continue

        # 6) Approximate contour to polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # We want exactly four points (quadrilateral) and convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Compute rotated bounding box via minAreaRect to get exact orientation
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="float32")

            # 7) Filter by aspect ratio and area
            (x, y), (w, h), angle = rect
            if w == 0 or h == 0:
                continue
            aspect_ratio = max(w, h) / min(w, h)
            # Typical SET card ratio is roughly 0.7–0.8 (portrait) or 1.25–1.4 (landscape).
            # We allow a range, e.g., between 1.1 and 2.0 to catch most playing‐card shapes
            if aspect_ratio < 1.1 or aspect_ratio > 2.0:
                continue
            if area < 2000:
                # skip if still too small after filtering; adjust this if cards appear small
                continue

            # Re‐order the 4 corners to TL, TR, BR, BL
            ordered = order_box_points(box)
            # Scale the box coordinates back to the original frame size (if drawing on original)
            ordered *= ratio
            card_boxes.append(ordered.astype(int))

    return card_boxes

def main():
    cap = cv2.VideoCapture(1)  # Open default camera (change index if necessary)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow("SET Card Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        boxes = detect_cards_in_frame(frame)

        # Draw tight bounding quadrilaterals
        for box in boxes:
            # box is 4×2 array of int coordinates (TL, TR, BR, BL)
            cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("SET Card Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # press ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
