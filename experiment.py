"converting color matrix to numpy matrix to make process easy"
feedmatrix = np.array(colored_matrix)
np.save('matrix1.npy',feedmatrix)
grid = feedmatrix

"change it to set(input()) from user letter"
start = starting_coordinate
goal = ending_coordinate

# applying astar search function
rout = astar_search(grid, start, goal)
print(rout)
