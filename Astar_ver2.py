import heapq
import pickle
import os
from typing import Dict, List, Tuple, Optional

Coordinate = Tuple[float, float]
Graph = Dict[Coordinate, Dict[Coordinate, float]]


def weighted_graph(coordinate_dict):
    coordinate_list = list(coordinate_dict.values())

    def create_weighted_graph(coord_list):
        weighted_graph = {}

        for coord_a in coord_list:
            weighted_graph[coord_a] = {}
            x_a, y_a = coord_a

            for coord_b in coord_list:
                x_b, y_b = coord_b

                if x_a == x_b and y_a == y_b:
                    continue  # Skip the same coordinate

                # Calculate 2D Euclidean distance
                distance = ((x_b - x_a) ** 2 + (y_b - y_a) ** 2) ** 0.5
                weighted_graph[coord_a][coord_b] = distance

        return weighted_graph

    return create_weighted_graph(coordinate_list)


def dynamic_path(path, current: Coordinate) -> Optional[List[Coordinate]]:
    if not path:
        return None
    goal = path[-1]
    while current != goal:
        if is_near(current, path[0]):
            path.pop(0)
        if not path:  # Check if path is empty after popping
            return None
        return [current, path[0]]
    return None


def is_near(coord1: Coordinate, coord2: Coordinate) -> bool:
    return (abs(coord1[0] - coord2[0]) <= 1 and
            abs(coord1[1] - coord2[1]) <= 1)


def astar(graph: Graph, start: Coordinate, goal: Coordinate) -> Optional[List[Coordinate]]:
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in graph.get(current, {}):
            tentative_g_score = g_score[current] + graph[current][neighbor]

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None


def heuristic(a: Coordinate, b: Coordinate) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def reconstruct_path(came_from: Dict[Coordinate, Coordinate], current: Coordinate) -> List[Coordinate]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


def load_dictionary(file_name: str) -> Optional[Dict]:
    try:
        with open(file_name, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None


def list_pkl_files() -> List[str]:
    return [f for f in os.listdir() if f.endswith('.pkl')]


def select_file() -> Optional[str]:
    pkl_files = list_pkl_files()
    if not pkl_files:
        print("No .pkl files found in the current directory.")
        return None

    print("Available .pkl files:")
    for i, file in enumerate(pkl_files, 1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = int(input("Enter the number of the file you want to open: "))
            if 1 <= choice <= len(pkl_files):
                return pkl_files[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def direction(current: Coordinate, path: List[Coordinate]) -> Dict[str, any]:
    if len(path) < 3:
        return {"error": "Path must contain at least three points after the current position"}

    def displacement(a: Coordinate, b: Coordinate) -> Tuple[float, float]:
        return (b[0] - a[0], b[1] - a[1])

    def cross_product_2d(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        return v1[0] * v2[1] - v1[1] * v2[0]

    next_point = path[0]
    point_after_next = path[1]
    point_after_after_next = path[2]

    disp1 = displacement(current, next_point)
    disp2 = displacement(next_point, point_after_next)
    disp3 = displacement(point_after_next, point_after_after_next)

    # Calculate the 2D cross products
    cross_product1 = cross_product_2d(disp1, disp2)
    cross_product2 = cross_product_2d(disp2, disp3)

    # Determine turn direction for the next turn
    if cross_product2 > 0:
        next_turn_direction = "left"
    elif cross_product2 < 0:
        next_turn_direction = "right"
    else:
        next_turn_direction = "straight"

    # Calculate distance to the next point and the turn
    distance_to_next = sum(x ** 2 for x in disp1) ** 0.5
    distance_to_turn = distance_to_next + sum(x ** 2 for x in disp2) ** 0.5

    # Check for immediate direction change
    if cross_product1 != 0:
        immediate_turn = "left" if cross_product1 > 0 else "right"
    else:
        immediate_turn = None

    # Prepare the warning message
    warning = ""
    if immediate_turn:
        warning += f"Immediate {immediate_turn} turn. "
    if next_turn_direction != "straight":
        warning += f"Prepare to turn {next_turn_direction} after {distance_to_next:.2f} units. "

    return {
        "warning": warning.strip(),
        "immediate_turn": immediate_turn,
        "next_turn_direction": next_turn_direction,
        "distance_to_next": distance_to_next,
        "distance_to_turn": distance_to_turn,
    }


def main():
    coordinate_file = select_file()
    if not coordinate_file:
        return
    coordinate_dict = load_dictionary(coordinate_file)
    if not coordinate_dict:
        return

    graph_file = select_file()
    if not graph_file:
        return
    graph = load_dictionary(graph_file)
    if not graph:
        return
    w_graph = weighted_graph(graph)

    starting_point = input("Starting point: ")
    goal_point = input("Ending point: ")

    starting_coordinate = coordinate_dict.get(starting_point)
    ending_coordinate = coordinate_dict.get(goal_point)

    if starting_coordinate and ending_coordinate:
        print(f"Starting coordinate: {starting_coordinate}")
        print(f"Ending coordinate: {ending_coordinate}")

        checkpoints = astar(w_graph, starting_coordinate, ending_coordinate)
        pathc = [(0, 0), (10.0, 12.0), (10.0, 17.0), (13.0, 17.0)]  # path of the characters
        i = 0
        while i < len(pathc):
            c = pathc[i]  # current location of user
            path_update = dynamic_path(checkpoints, c)
            print(f"Current position: {c}")
            print(f"Path update: {path_update}")

            # Ensure we have at least 3 points for direction calculation
            if path_update and len(checkpoints) >= 3:
                extended_path = [c] + checkpoints[:3]  # Include current position and next 3 points
                dir_info = direction(c, extended_path)
                print(f"Direction info: {dir_info}")
            elif path_update:
                print("Not enough points for detailed direction info.")

            i += 1

        if checkpoints:
            print("Shortest path:", checkpoints)
        else:
            print("No path found")
    else:
        if not starting_coordinate:
            print(f"Starting point '{starting_point}' not found in the dictionary.")
        if not ending_coordinate:
            print(f"Ending point '{goal_point}' not found in the dictionary.")


if __name__ == "__main__":
    "main()"