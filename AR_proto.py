import numpy as np
import cv2
from typing import List, Tuple

Coordinate = Tuple[float, float]


def direction(current: Coordinate, path: List[Coordinate]) -> dict:
    if len(path) < 3:
        return {"error": "Path must contain at least three points after the current position"}

    next_point, point_after_next, point_after_after_next = path[:3]

    def displacement(a: Coordinate, b: Coordinate) -> np.ndarray:
        return np.array(b) - np.array(a)

    def cross_product_2d(v1: np.ndarray, v2: np.ndarray) -> float:
        return np.cross(np.append(v1, 0), np.append(v2, 0))[2]

    disp1 = displacement(current, next_point)
    disp2 = displacement(next_point, point_after_next)
    disp3 = displacement(point_after_next, point_after_after_next)

    cross_product1 = cross_product_2d(disp1, disp2)
    cross_product2 = cross_product_2d(disp2, disp3)

    next_turn_direction = "left" if cross_product2 > 0 else "right" if cross_product2 < 0 else "straight"

    distance_to_next = np.linalg.norm(disp1)
    distance_to_turn = distance_to_next + np.linalg.norm(disp2)

    immediate_turn = "left" if cross_product1 > 0 else "right" if cross_product1 < 0 else None

    warning = []
    if immediate_turn:
        warning.append(f"Immediate {immediate_turn} turn.")
    if next_turn_direction != "straight":
        warning.append(f"Prepare to turn {next_turn_direction} after {distance_to_next:.2f} units.")

    return {
        "warning": " ".join(warning),
        "immediate_turn": immediate_turn,
        "next_turn_direction": next_turn_direction,
        "distance_to_next": distance_to_next,
        "distance_to_turn": distance_to_turn,
    }


def draw_arrow(image: np.ndarray, center: Tuple[int, int], direction: str, scale: float) -> np.ndarray:
    arrow_points = np.array([[-1, 1], [0, 0], [-1, -1], [1, 0]], dtype=np.float32)
    angle = -90 if direction == "left" else 90 if direction == "right" else 0

    arrow_points *= scale
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    arrow_points = cv2.transform(np.array([arrow_points]), rotation_matrix)[0]
    arrow_points += np.array(center)

    cv2.polylines(image, [arrow_points.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)
    return image


def render_navigation_arrow(frame: np.ndarray, current_position: Coordinate, path: List[Coordinate]):
    direction_info = direction(current_position, path)

    if "error" in direction_info:
        cv2.putText(frame, direction_info["error"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    next_checkpoint = tuple(map(int, path[1]))
    scale = max(0.5, min(2.0, 10 / max(direction_info["distance_to_next"], 0.1)))  # Avoid divide by zero

    frame = draw_arrow(frame, next_checkpoint, direction_info["next_turn_direction"], scale)

    # Add text instructions
    cv2.putText(frame, direction_info["warning"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


def main():
    path = [(100, 100), (200, 150), (250, 250), (300, 300)]
    current_position = (50, 50)  # Start outside the path

    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = render_navigation_arrow(frame, current_position, path)
            cv2.imshow('AR Navigation', frame)

            # Move towards the next point
            if len(path) > 1:
                direction_to_next = np.array(path[0]) - np.array(current_position)
                distance_to_next = np.linalg.norm(direction_to_next)

                if distance_to_next > 1:  # If we're not very close to the next point
                    step = direction_to_next / distance_to_next  # Normalize the direction
                    current_position = tuple(np.array(current_position) + step)
                else:
                    current_position = path.pop(0)  # Reach the point and remove it from the path

            # If we've reached the end of the path, start over
            if len(path) < 3:
                path = [(100, 100), (200, 150), (250, 250), (300, 300)]
                current_position = (50, 50)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Script interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()