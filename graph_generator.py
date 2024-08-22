import pickle

with open('dyp2nd.pkl', 'rb') as file:
    cordinate_dict = pickle.load(file)

print(cordinate_dict)
"initializing a array coordinate to make my work go from hash map to array"
coordinate_list = []
for value in cordinate_dict.values():
    coordinate_list.append(value)
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

weighted_graph = create_weighted_graph(coordinate_list)
print(weighted_graph)
