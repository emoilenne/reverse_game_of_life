import numpy as np
from window import Window, Transformation

class TestCase:
    def __init__(self, fields, row, height, width, windowSize, isTraining):
        """
            Initiate testcase with a row containing id, steps and grids
        """
        try:
            self.data = {fields[index]: row[index] for index in range(len(fields))}
            self.id = int(self.data['id'])
            del(self.data['id'])
            self.steps = int(self.data['delta'])
            del(self.data['delta'])
        except:
            raise Exception("CSV file is not valid.")

        self.height = height
        self.width = width
        self.windowSize = windowSize
        totalValues = height * width * 2 if isTraining else height * width
        if len(self.data) != totalValues:
            raise Exception("Dimentions of the grid don't match values in the CSV file.")

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
            for index in range(self.height * self.width):
                grid.append(int(self.data[name + '.' + str(index + 1)]))
            grid = np.array(grid).reshape(self.height, self.width)
        except Exception as e:
            raise Exception("Unable to create %s grid" % name)
        return grid

    def trainWindow(self, position, models):
        """
            Determine window's model and add predictions to training models.
        """
        stopWindow = Window(self.stopGrid, position, self.windowSize)
        startWindow = Window(self.startGrid, position, self.windowSize)
        transformation, stopWindowHash = stopWindow.toHash()
        model = models[self.steps][stopWindowHash]
        startWindow.data = Transformation.do[transformation](startWindow.data)
        model.addPrediction(startWindow)
        models[self.steps][stopWindowHash] = model

    def train(self, models):
        """
            Run training on this test case, which will write predictions in the models.
        """
        for height in range(self.height - self.windowSize + 1):
            for width in range(self.width - self.windowSize + 1):
                self.trainWindow((height, width), models)

    def predict(self, models):
        """
            Predict start grid based on stop grid of this test case using models.
        """
        predictionGrid = np.zeros((self.height, self.width, 2))
        for height in range(self.height - self.windowSize + 1):
            for width in range(self.width - self.windowSize + 1):
                stopWindow = Window(self.stopGrid, (height, width), self.windowSize)
                transformation, stopWindowHash = stopWindow.toHash()
                model = models[self.steps][stopWindowHash]
                predictionGrid += model.predict(transformation, (height, width), self.height, self.width)

        #TODO can create start grid directly from prediciton grid
        probability = lambda count, occurrences: 0. if occurrences == 0 else count / float(occurrences)
        probabilityGrid = np.array([probability(predictionGrid[h,w,0], predictionGrid[h,w,1]) for h in range(self.height) for w in range(self.width)]).reshape(self.height, self.width)
        startGrid = np.array([round(probabilityGrid[h,w]) for h in range(self.height) for w in range(self.width)], dtype=np.int8).reshape(self.height, self.width)

        return startGrid
