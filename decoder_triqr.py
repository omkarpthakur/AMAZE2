import cv2
import numpy as np


def detect_triangle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3 and cv2.contourArea(approx) > 1000:
            return approx

    return None


def order_points(pts):
    # Order points: [top, right, left]
    center = np.mean(pts, axis=0)
    return sorted(pts, key=lambda p: -p[1] if p[1] > center[1] else p[0])


def decode_triangular_qr(warped, size=50):
    h, w = warped.shape[:2]
    dot_spacing = w / (size - 1)

    binary = ""
    for i in range(size):
        y = int(i * np.sqrt(3) / 2 * dot_spacing)
        for j in range(size - i):
            x = int(i / 2 * dot_spacing + j * dot_spacing)

            if y < h and x < w:
                cell = warped[y - 1:y + 2, x - 1:x + 2]
                if np.mean(cell) < 127:
                    binary += "1"
                else:
                    binary += "0"

    # Convert binary to text
    message = ""
    for i in range(0, len(binary), 8):
        byte = binary[i:i + 8]
        if len(byte) == 8:
            message += chr(int(byte, 2))

    return message


def scan_triangular_qr():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        triangle = detect_triangle(frame)

        if triangle is not None:
            pts = order_points(triangle.reshape(3, 2))
            cv2.drawContours(frame, [triangle], 0, (0, 255, 0), 2)

            # Warp the detected triangle into a straight one
            width = int(np.linalg.norm(pts[1] - pts[2]))
            height = int(width * np.sqrt(3) / 2)
            dst = np.array([[width // 2, 0], [width, height], [0, height]], dtype=np.float32)
            matrix = cv2.getAffineTransform(np.float32(pts), dst)
            warped = cv2.warpAffine(frame, matrix, (width, height))

            # Convert warped image to binary
            gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, binary_warped = cv2.threshold(gray_warped, 127, 255, cv2.THRESH_BINARY)

            # Decode the QR code
            message = decode_triangular_qr(binary_warped)
            cv2.putText(frame, f"Decoded: {message}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Triangular QR Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    scan_triangular_qr()