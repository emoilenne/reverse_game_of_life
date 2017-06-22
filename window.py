import numpy as np

class Window:

    @staticmethod
    def overlap(first, second, position):
        """
            For overlapping windows determine the area that is overlapping.
        """
        firstStart = (max(0, position[0]), max(0, position[1]))
        firstEnd = (min(first.shape[0], second.shape[0] + position[0]), min(first.shape[1], second.shape[1] + position[1]))
        secondStart = (max(0, -position[0]), max(0, -position[1]))
        secondEnd = (min(second.shape[0], first.shape[0] - position[0]), min(second.shape[1], first.shape[1] - position[1]))
        return firstStart, firstEnd, secondStart, secondEnd

    def __init__(self, grid, start, size):
        """
            Start and size of the window define the window from initial grid.
            The window will be size * size cells.
        """
        self.size = size
        self.data = np.zeros((size, size), dtype=np.int8)
        gStart, gEnd, wStart, wEnd = Window.overlap(grid, self.data, start)
        self.data[wStart[0]:wEnd[0], wStart[1]:wEnd[1]] = grid[gStart[0]:gEnd[0], gStart[1]:gEnd[1]]


    def toHash(self):
        """
            Calculate the hash of the window.
        """

        hashedWindow = [2 ** (self.size * h + w) * self.data[h,w] for h in range(self.size) for w in range(self.size)]
        return sum(hashedWindow)
