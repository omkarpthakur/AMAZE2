import cv2
import numpy as np
import pickle
import os
print("\033[38;5;208m(please give name of the package) \033[0;0m")
name = str(input())


class UnknownColorError(Exception):
    pass
def jpeg_to_matrix(image_path, pixel_size):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # can add colors if needed
    colors = {
        (0, 0, 0): 0,      # Black (for the roads)
        (255, 255, 255): 1,  # White ( for the void)
        (17, 99, 33): 2,   # Blue (for the nodes)
        (255, 0, 0): 3,    # Red  ()
        (0, 255, 0): 4     # Green()
    }

    def map_color(value):
        for color, code in colors.items():
            if np.all(np.abs(value - np.array(color)) < pixel_size):
                return code
        raise UnknownColorError("Unknown color")

    try:
        colored_img = np.array([[map_color(value) for value in row] for row in img])
    except UnknownColorError as e:
        print(f"Color mapping error: {str(e)}")
        colored_img = None

    return colored_img

def get_coordinates(colored_matrix, color_code):
    coordinates = np.argwhere(colored_matrix == color_code)
    return coordinates.tolist()

image_path = r"D:\AMAZE\map1.png"   # i don't know how but change it to server adress letter
pixel_size = 20
colored_matrix = jpeg_to_matrix(image_path, pixel_size)

if colored_matrix is not None:
   # print(colored_matrix)

    # nodes
    blue_coordinates = get_coordinates(colored_matrix, 2)
    #print("Blue coordinates:", blue_coordinates)
with open("matrix.pkl",'wb') as f :
    pickle.dump(colored_matrix,f)

def create_directory_with_pickle_files(directory_name, num_files=3):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    for i in range(num_files):
        # Generate a unique filename for each pickle file
        pickle_filename = os.path.join(directory_name, f'pickle_file_{i}.pkl')

        # Create some sample data (you can replace this with your own data)
        data = {'example_key': f'example_data_{i}'}

        # Serialize and save the data as a pickle file
        with open(pickle_filename, 'wb') as file:
            pickle.dump(data, file)

    print(f"Created directory '{directory_name}' with {num_files} pickle files.")

# Usage example:
directory_name = 'example_directory'
create_directory_with_pickle_files(directory_name)