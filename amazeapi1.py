from flask import Flask, request, jsonify, render_template
from typing import Dict, List, Tuple, Optional
import heapq

app = Flask(__name__)

# Type aliases
Coordinate = Tuple[float, float]
Graph = Dict[Coordinate, Dict[Coordinate, float]]


# Your existing functions
def is_near(coord1: Coordinate, coord2: Coordinate) -> bool:
    return abs(coord1[0] - coord2[0]) <= 1 and abs(coord1[1] - coord2[1]) <= 1


def dynamic_path(path: List[Coordinate], current: Coordinate) -> Optional[List[Coordinate]]:
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
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from: Dict[Coordinate, Coordinate], current: Coordinate) -> List[Coordinate]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


# Sample data (replace with your actual data)
coordinate_dict = {
    'A': (100, 100),
    'B': (200, 200),
    'C': (300, 150),
    'D': (400, 300),
    'E': (500, 250)
}


def weighted_graph(cordinate_dict):
    coordinate_list = list(cordinate_dict.values())

    def create_weighted_graph(coord_list):
        weighted_graph = {}

        for coord_a in coord_list:
            weighted_graph[coord_a] = {}
            x_a, y_a = coord_a

            for coord_b in coord_list:
                x_b, y_b = coord_b

                if x_a == x_b and y_a == y_b:
                    continue  # Skip the same coordinate

                if x_a == x_b or y_a == y_b:
                    distance = abs(x_a - x_b) + abs(y_a - y_b)
                    weighted_graph[coord_a][coord_b] = distance

        return weighted_graph

    return create_weighted_graph(coordinate_list)


graph = weighted_graph(coordinate_dict)


@app.route('/')
def index():
    return render_template('index.html', points=coordinate_dict)


@app.route('/api/path', methods=['POST'])
def get_path():
    data = request.json
    start = tuple(data['start'])
    goal = tuple(data['goal'])

    initial_path = astar(graph, start, goal)
    if not initial_path:
        return jsonify({"error": "No path found"}), 404

    return jsonify({"path": initial_path})


@app.route('/api/dynamic_path', methods=['POST'])
def update_path():
    data = request.json
    current = tuple(data['current'])
    path = [tuple(p) for p in data['path']]

    result = dynamic_path(path, current)
    if result is None:
        return jsonify({"error": "Path completed or invalid"}), 404

    return jsonify({"next_point": result[1]})


if __name__ == '__main__':
    app.run(debug=True)