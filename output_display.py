import pickle

from matplotlib import pyplot,colors
from Astar_ver2 import astar

import numpy
with open('matrix.pkl','rb') as f :
    colored_matrix = pickle.load(f)
grid = colored_matrix
data = grid
path =astar(graph, starting_coordinate, ending_coordinate)
numrows = len(grid)
numcols = len(grid[0])
#print(numrows,numcols)
colormap = colors.ListedColormap(["paleturquoise","darkcyan"])
colormap2=colors.ListedColormap(["Red"])
pyplot.figure(figsize=(numrows,numcols))

pyplot.imshow(grid, cmap=colormap, origin='upper', extent=(0, len(grid), 0, len(grid[0])))

# Plot the path
for i in range(1, len(path)):
    x1, y1 = path[i - 1]
    x2, y2 = path[i]
    pyplot.plot([x1, x2], [y1, y2], 'bo-')
fig, ax = pyplot.subplots()
pyplot.show()
