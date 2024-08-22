import pickle
import os

def add_to_dictionary_and_save(dictionary, key, value, file_name):
    dictionary[key] = value
    with open(file_name, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dictionary(file_name):
    try:
        with open(file_name, 'rb') as file:
            loaded_dictionary = pickle.load(file)
        return loaded_dictionary
    except FileNotFoundError:
        print(f"File '{file_name}' not found. Creating a new dictionary.")
        return {"origin": (0, 0)}

def get_file_name():
    while True:
        file_name = input("Enter the file name (without extension) or 'q' to quit: ")
        if file_name.lower() == 'q':
            return None
        return f"{file_name}.pkl"

def main():
    while True:
        file_name = get_file_name()
        if file_name is None:
            break

        coordinate_dict = load_dictionary(file_name)

        while True:
            new_key = input("Please give the new location (or 'q' to quit, 'c' to change file): ")

            if new_key.lower() == 'q':
                break
            elif new_key.lower() == 'c':
                break

            x_coordinate = float(input("Please give the x-coordinate of the location on the map: "))
            y_coordinate = float(input("Please give the y-coordinate of the location on the map: "))

            new_value = (x_coordinate, y_coordinate)
            add_to_dictionary_and_save(coordinate_dict, new_key, new_value, file_name)

            print(f"Added {new_key}: {new_value} to {file_name}")

        print(f"Final Dictionary in {file_name}:")
        print(coordinate_dict)

        if new_key.lower() != 'c':
            break

if __name__ == "__main__":
    main()