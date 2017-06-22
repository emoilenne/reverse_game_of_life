import numpy as np
import random

class Map:
    """
        Represents map of Conway's Game of Life.
    """
    cellfate = staticmethod(lambda cell, neighbors: [0, 0, cell, 1, 0, 0, 0, 0, 0][neighbors])

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.total = height * width

    def generate(self):
        """
            Generate new map and do warmup.
        """
        self.data = np.zeros((self.height, self.width), dtype=np.int8)

        aliveCells = random.randrange(int(0.01 * self.total + 0.49), int(0.99 * self.total - 0.5))
        cells = [1] * aliveCells + [0] * (self.total - aliveCells)
        for h in range(self.height):
            for w in range(self.width):
                cell = random.choice(cells)
                cells.remove(cell)
                self.data[h,w] = cell

        self.step(5)
        self.steps = random.randint(1, 5)

    def getSteps(self):
        """
            Return number of steps for the map.
        """
        return self.steps

    def step(self, count=1):
        """
            Make 1 step on the map.
        """
        self.data = np.pad(self.data, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        newMap = np.zeros((self.height, self.width), dtype=np.int8)
        neighbors = sum(np.roll(np.roll(self.data, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1)
            if (i != 0 or j != 0))
        for h in range(self.height):
            for w in range(self.width):
                newMap[h,w] = Map.cellfate(self.data[h,w], neighbors[h+1,w+1])
        self.data = newMap

    def getValues(self):
        """
            Get map values in a list of size height * width.
        """
        return [str(self.data[h,w]) for h in range(self.height) for w in range(self.width)]

    def aliveCells(self):
        """
            Get number of alive steps.
        """
        return sum(sum(self.data[row]) for row in range(self.height))
