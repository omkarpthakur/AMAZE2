import cv2
import numpy as np

def calculate_angle(pt1, pt2, pt3):
    """Calculate the angle between three points."""
    v1 = pt1 - pt2
    v2 = pt3 - pt2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def detect_red_triangle(frame):
    """Detect red-colored triangle using contours with improved accuracy."""
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for red color (two ranges to cover the red hue spectrum)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the best triangle
    best_triangle = None
    best_score = float('inf')

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 3 vertices (triangle)
        if len(approx) == 3:
            # Calculate the area and perimeter of the triangle
            area = cv2.contourArea(approx)
            perimeter = cv2.arcLength(approx, True)

            # Calculate the minimum area rectangle
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rect_area = cv2.contourArea(box)

            # Calculate how well the contour fits a triangle
            triangle_score = abs(1 - (area / rect_area))

            # Check if this triangle is better than the previous best
            if triangle_score < best_score and area > 1000:  # Add minimum area threshold
                best_score = triangle_score
                best_triangle = approx

    if best_triangle is not None:
        # Draw the triangle on the frame
        cv2.drawContours(frame, [best_triangle], 0, (0, 255, 0), 2)

        # Calculate and display side lengths and angles
        pts = best_triangle.reshape(3, 2)
        side_lengths = [np.linalg.norm(pts[i] - pts[(i+1)%3]) for i in range(3)]
        angles = [calculate_angle(pts[i], pts[(i+1)%3], pts[(i+2)%3]) for i in range(3)]

        for i, (length, angle) in enumerate(zip(side_lengths, angles)):
            cv2.putText(frame, f"L{i+1}: {length:.2f}", tuple(pts[i]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"A{i+1}: {angle:.2f}",
                        tuple(pts[i] + [0, 20]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame, (side_lengths, angles)

    return frame, None

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, triangle_data = detect_red_triangle(frame)

        cv2.imshow('Red Triangle Detection', processed_frame)

        if triangle_data:
            side_lengths, angles = triangle_data
            print(f"Red triangle detected - Side lengths: {side_lengths}, Angles: {angles}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()