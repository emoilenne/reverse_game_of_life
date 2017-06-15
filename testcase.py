import numpy as np
from window import Window, Transformation

class TestCase:
    def __init__(self, fields, row, height, width, isTraining, windowSize=4):
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
        self.windowSize = windowSize
        totalValues = height * width * 2 if isTraining else height * width
        if len(self.data) != totalValues:
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

    def trainWindow(self, position, models):
        """
            Determine window's model and add predictions to training models.
        """
        # Create a window from stop grid
        stopWindow = Window(self.stopGrid, position, self.windowSize)

        # Create same window from start grid
        startWindow = Window(self.startGrid, position, self.windowSize)

        # Determine model of stop window (get hash value and transformation needed for hash value)
        transformation, stopWindowHash = stopWindow.toHash()

        # Obtain stored model for this window
        model = models[self.steps][stopWindowHash]

        # Transform start window the same as stop window was transformed in hash
        startWindow.data = Transformation.do[transformation](startWindow.data)

        # Add predictions to stored model
        model.addPrediction(startWindow)

        # Save model (needed if this was the first time the model has found)
        models[self.steps][stopWindowHash] = model

    def train(self, models):
        """
            Run training on this test case, which will write predictions in the models.
        """
        # Run training on each window in the grid
        for height in range(self.height - self.windowSize + 1):
            for width in range(self.width - self.windowSize + 1):
                self.trainWindow((height, width), models)

    def predict(self, models):
        """
            Predict start grid based on stop grid of this test case using models.
        """
        # Create predicitons grid
        predictionGrid = np.zeros((self.height, self.width, 2))

        # Collect predictions grid from each window
        for height in range(self.height - self.windowSize + 1):
            for width in range(self.width - self.windowSize + 1):
                # Create a window of stop grid
                stopWindow = Window(self.stopGrid, (height, width), self.windowSize)

                # Determine model of stop window (get hash value and transformation needed for hash value)
                transformation, stopWindowHash = stopWindow.toHash()

                # Obtain stored model for this window
                model = models[self.steps][stopWindowHash]

                # Add predicitons grid of this model
                predictionGrid += model.predict(transformation, (height, width), self.height, self.width)

        #TODO can create start grid directly from prediciton grid
        # Create probability grid, where each element will represent probability of this cell alive (from 0. to 1.)
        probability = lambda count, occurrences: 0. if occurrences == 0 else count / float(occurrences)
        probabilityGrid = np.array([probability(predictionGrid[h,w,0], predictionGrid[h,w,1]) for h in range(self.height) for w in range(self.width)]).reshape(self.height, self.width)

        # Create start grid based on probability grid
        startGrid = np.array([round(probabilityGrid[h,w]) for h in range(self.height) for w in range(self.width)], dtype=np.int8).reshape(self.height, self.width)

        return startGrid
