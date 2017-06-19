import numpy as np
from trainingModel import CellPosition

class TestCase:
    def __init__(self, fields, row, height, width, isTraining):
        """
            Initiate testcase with a row containing id, steps and grids
        """
        # Store data from row
        try:
            #TODO can just read values as id,delta,start.1,start.2,... without reading field names
            self.data = {fields[index]: row[index] for index in range(len(fields))}
            self.id = int(self.data['id'])
            del(self.data['id'])
            self.steps = int(self.data['delta'])
            del(self.data['delta'])
        except:
            raise Exception("CSV file is not valid.")

        # Check if grids are valid
        self.height = height
        self.width = width
        totalValues = height * width * 2 if isTraining else height * width
        if len(self.data) < totalValues:
            raise Exception("Dimentions of the grid don't match values in the CSV file.")

        # Create grids
        self.stopGrid = self.createGrid('stop')
        if isTraining:
            self.startGrid = self.createGrid('start')

    def getId(self):
        return self.id

    def createGrid(self, name):
        """
            Create a grid with specific name from data stored in this test case.
        """
        grid = []
        try:
            # Read values from the file
            for index in range(self.height * self.width):
                grid.append(int(self.data[name + '.' + str(index + 1)]))

            # Creade NumPy array from obtained values
            grid = np.array(grid).reshape(self.height, self.width)
        except Exception as e:
            raise Exception("Unable to create %s grid" % name)
        return grid

    def trainCell(self, position, models):
        """
            Determine cell's model and add predictions to training models.
        """
        # Get number of neighbors
        neighbors = CellPosition.getNeighbors(position, self.stopGrid, self.steps)

        # Get cell position (corner, edge, middle)
        cellPosition = CellPosition.getCellPosition(position, self.stopGrid)

        # Obtain stored model for this window
        model = models[self.steps][neighbors]

        # Add predictions to stored model
        model.train(self.startGrid[position], self.stopGrid[position], cellPosition)

        # Save model (needed if this was the first time the model has found)
        models[self.steps][neighbors] = model

    def train(self, models):
        """
            Run training on this test case, which will store predictions in the models.
        """
        # Run training on each window in the grid
        for height in range(self.height):
            for width in range(self.width):
                self.trainCell((height, width), models)

    def predict(self, models):
        """
            Predict start grid based on stop grid of this test case using models.
        """
        # Create empty start grid
        startGrid = np.zeros((self.height, self.width), dtype=np.int8)

        # Collect predictions from each cell
        for height in range(self.height):
            for width in range(self.width):
                # Get number of neighbors
                neighbors = CellPosition.getNeighbors((height, width), self.stopGrid, self.steps)

                # Get cell position (corner, edge, middle)
                cellPosition = CellPosition.getCellPosition((height, width), self.stopGrid)

                # Obtain stored model for this cell
                model = models[self.steps][neighbors]

                # Add predicitons of this model to start grid
                startGrid[height][width] = model.predict(cellPosition, self.stopGrid[height][width])

        return startGrid
