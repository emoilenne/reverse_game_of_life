import numpy as np

class Window:

    def __init__(self, grid, startPoint, size):
        """
            startPoint, height and width define the window from initial grid.
            The window will be size * size cells.
            If dimensions exceed the grid, the exception will be raised.
        """
        gridHeight, gridWidth = grid.shape
        startPointHeight, startPointWidth = startPoint
        endPointHeight = startPointHeight + size
        endPointWidth = startPointWidth + size

        self.size = size
        self.data = grid[startPointHeight:endPointHeight, startPointWidth:endPointWidth]

    def toHash(self):
        """
            Calculate the hash of the window.
        """

        hashedWindow = 2 ** self.data

        return sum([sum(hashedWindow[row]) for row in range(self.size)])
