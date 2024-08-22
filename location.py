import cv2
import numpy as np
from pyzbar.pyzbar import decode


def calculate_angle(pt1, pt2, pt3):
    """Calculate the angle between three points."""
    v1 = pt1 - pt2
    v2 = pt3 - pt2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def detect_red_triangle(frame):
    """Detect red-colored triangle using contours with improved accuracy."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_triangle = None
    best_score = float('inf')

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3:
            area = cv2.contourArea(approx)
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rect_area = cv2.contourArea(box)

            triangle_score = abs(1 - (area / rect_area))

            if triangle_score < best_score and area > 1000:
                best_score = triangle_score
                best_triangle = approx

    if best_triangle is not None:
        cv2.drawContours(frame, [best_triangle], 0, (0, 255, 0), 2)

        pts = best_triangle.reshape(3, 2)
        side_lengths = [np.linalg.norm(pts[i] - pts[(i + 1) % 3]) for i in range(3)]
        angles = [calculate_angle(pts[i], pts[(i + 1) % 3], pts[(i + 2) % 3]) for i in range(3)]

        for i, (length, angle) in enumerate(zip(side_lengths, angles)):
            cv2.putText(frame, f"L{i + 1}: {length:.2f}", tuple(pts[i]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"A{i + 1}: {angle:.2f}",
                        tuple(pts[i] + [0, 20]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame, (side_lengths, angles)

    return frame, None


def detect_qr_code(frame):
    """Detect QR codes in the frame."""
    decoded_objects = decode(frame)
    qr_data = []

    for obj in decoded_objects:
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(points)
            points = hull

        for i in range(len(points)):
            cv2.line(frame, points[i], points[(i + 1) % len(points)], (255, 0, 0), 3)

        qr_code_data = obj.data.decode("utf-8")
        qr_data.append(qr_code_data)
        cv2.putText(frame, qr_code_data, (points[0][0], points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame, qr_data


def detect_triangle_and_qr():
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    triangle_data = None
    qr_data = None

    while triangle_data is None or not qr_data:
        ret, frame = cap.read()
        if not ret:
            break

        frame, triangle_result = detect_red_triangle(frame)
        frame, qr_result = detect_qr_code(frame)

        cv2.imshow('Red Triangle and QR Code Detection', frame)

        if triangle_result and triangle_data is None:
            triangle_data = triangle_result

        if qr_result and not qr_data:
            qr_data = qr_result

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return qr_data


if __name__ == "__main__":
    qr_info = detect_triangle_and_qr()
    print("Function returned:", qr_info)