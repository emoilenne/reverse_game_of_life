import numpy as np
from window import Window

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
        self.expansion = windowSize - 1
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
            Create a grid with specific name from data stored in this test case,
            expanded in size - 1 cells in each dimention.
        """
        grid = []
        try:
            for index in range(self.height * self.width):
                grid.append(int(self.data[name + '.' + str(index + 1)]))
            grid = np.array(grid).reshape(self.height, self.width)
        except Exception as e:
            raise Exception("Unable to create %s grid" % name)
        return np.pad(grid, ((self.expansion, self.expansion), (self.expansion, self.expansion)), mode='constant', constant_values=0)

    def trainWindow(self, position, models):
        """
            Determine window's model and add predictions to training models.
        """
        stopWindow = Window(self.stopGrid, position, self.windowSize)
        startWindow = Window(self.startGrid, position, self.windowSize)
        stopWindowHash = stopWindow.toHash()
        model = models[self.steps][stopWindowHash]
        model.addPrediction(startWindow)
        models[self.steps][stopWindowHash] = model

    def train(self, models):
        """
            Run training on this test case, which will write predictions in the models.
        """
        for height in range(self.height + self.expansion):
            for width in range(self.width + self.expansion):
                self.trainWindow((height, width), models)

    def predict(self, models):
        """
            Predict start grid based on stop grid of this test case using models.
        """
        predictionGrid = np.zeros((self.height + 2 * self.expansion, self.width + 2 * self.expansion))
        for height in range(self.height + self.expansion):
            for width in range(self.width + self.expansion):
                stopWindow = Window(self.stopGrid, (height, width), self.windowSize)
                stopWindowHash = stopWindow.toHash()
                model = models[self.steps][stopWindowHash]
                predictionGrid[height:height+self.windowSize, width:width+self.windowSize] += model.predict()

        predictionGrid = predictionGrid[self.expansion: -self.expansion, self.expansion: -self.expansion]
        startGrid = np.array([int(round(predictionGrid[h,w] / (self.windowSize ** 2))) for h in range(self.height) for w in range(self.width)]).reshape((self.height, self.width))
        return startGrid
