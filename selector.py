import pickle
import os

def load_dictionary(file_name):
    try:
        with open(file_name, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None

def get(dict, key):
    return dict.get(key, None)

def list_pkl_files():
    return [f for f in os.listdir() if f.endswith('.pkl')]

def select_file():
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
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    file_name = select_file()
    if not file_name:
        return

    coordinate_dict = load_dictionary(file_name)
    if not coordinate_dict:
        return

    starting_point = input("Starting point: ")
    goal_point = input("Ending point: ")

    starting_coordinate = get(coordinate_dict, starting_point)
    ending_coordinate = get(coordinate_dict, goal_point)

    if starting_coordinate and ending_coordinate:
        print(f"Starting coordinate: {starting_coordinate}")
        print(f"Ending coordinate: {ending_coordinate}")
    else:
        if not starting_coordinate:
            print(f"Starting point '{starting_point}' not found in the dictionary.")
        if not ending_coordinate:
            print(f"Ending point '{goal_point}' not found in the dictionary.")

if __name__ == "__main__":
    main()
