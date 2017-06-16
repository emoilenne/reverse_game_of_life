import numpy as np
from functools import reduce
from operator import add

class CellPosition:
    # States of cell
    CORNER = 'corner'
    EDGE = 'edge'
    MIDDLE = 'middle'

    all = [CORNER, EDGE, MIDDLE]

    @staticmethod
    def getCellPosition(position, grid):
        """
            Return position of the cell in the grid (corner, edge, middle).
        """
        # Get positions
        cellHeight, cellWidth = position
        gridHeight, gridWidth = grid.shape

        # Corners
        if (cellWidth == 0 or cellWidth + 1 == gridWidth) and (cellHeight == 0 or cellHeight + 1 == gridHeight):
            return CellPosition.CORNER

        # Edges
        if cellWidth == 0 or cellWidth + 1 == gridWidth or cellHeight == 0 or cellHeight + 1 == gridHeight:
            return CellPosition.EDGE

        # Middle
        return CellPosition.MIDDLE

    @staticmethod
    def getNeighbors(position, grid):
        """
            Get number of neighbors of the cell.
        """
        cellHeight, cellWidth = position
        gridHeight, gridWidth = grid.shape
        neighbors = 0

        # Define neighborhood of the cell
        nStartHeight = max(cellHeight - 1, 0)
        nStartWidth = max(cellWidth - 1, 0)
        nEndHeight = min(cellHeight + 1, gridHeight - 1)
        nEndWidth = min(cellWidth + 1, gridWidth - 1)

        # Count neighbors
        for h in range(nStartHeight, nEndHeight + 1):
            for w in range(nStartWidth, nEndWidth + 1):
                if (h, w) != position:
                    neighbors += grid[h,w]

        return neighbors


class TrainingModel:

    fields = ['alive', 'dead', 'same', 'opposite', 'occurrences']
    prediction = {
        'alive': lambda x: 1,
        'dead': lambda x: 0,
        'same': lambda x: x,
        'opposite': lambda x : 1 - x
    }

    def __init__(self, steps=0, neighbors=0, fields=None, row=None):
        """
            Initiate training model, which stores predictions of the start grid
            and number of occurrences of this model in the testing.
        """
        self.data = {position: [0] * len(TrainingModel.fields) for position in CellPosition.all}
        if fields and row:
            self.parseRow(fields, row)
        else:
            self.steps = steps
            self.neighbors = neighbors

    def train(self, wasAlive, isAlive, cellPosition):
        """
            Add start grid value of this model to predictions.
        """
        self.data[cellPosition] = map(add, self.data[cellPosition], [wasAlive, 1 - wasAlive, 1 - abs(isAlive - wasAlive), abs(isAlive - wasAlive), 1])

    def predict(self, cellPosition, stopGridValue):
        """
            Predict is cell alive or dead in the position.
        """
        # Return 0 or 1 based on training data
        if self.data[cellPosition][-1] == 0:
            return 0

        # Get predictions on every state
        predictions = [val / float(self.data[cellPosition][-1]) for val in self.data[cellPosition][:-1]]

        #TODO can save this state when predictions is called, and when training is called, unset it
        # Get most likely state
        state = TrainingModel.fields[predictions.index(max(predictions))]

        return int(TrainingModel.prediction[state](stopGridValue))

    def parseRow(self, fields, row):
        """
            Retrieve data from row.
        """
        try:
            data = {fields[index]: row[index] for index in range(len(fields))}
            self.steps = int(data['steps'])
            self.neighbors = int(data['neighbors'])
            position = data['position']
            if position not in CellPosition.all:
                raise Exception("There is no such cell position.")
            self.data[position][0] = int(data['alive'])
            self.data[position][1] = int(data['dead'])
            self.data[position][2] = int(data['same'])
            self.data[position][3] = int(data['opposite'])
            self.data[position][-1] = int(data['occurrences'])
        except Exception as e:
            raise Exception("CSV file is not valid: " + str(e))

    def createRow(self):
        """
            Create row of CSV that will store data from the model.
        """
        # There will be a row for each position of the cell
        rows = {pos: "%d,%d,%s,%s\n" % (self.steps, self.neighbors, pos, ','.join(str(val) for val in self.data[pos])) for pos in CellPosition.all}

        return reduce(lambda x, y: x + y, rows.values())
