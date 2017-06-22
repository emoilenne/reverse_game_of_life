import numpy as np
from window import Window

class TestCase:
    def __init__(self, fields, row, height, width, isTraining):
        """
            Initiate testcase with a row containing id, steps and grids
        """
        try:
            self.data = {fields[index]: row[index] for index in range(len(fields))}
            self.id = int(self.data['id'])
            del(self.data['id'])
            self.steps = int(self.data['delta'])
            del(self.data['delta'])
        except Exception as e:
            raise Exception("CSV file is not valid: " + str(e))

        self.height = height
        self.width = width
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
        return grid

    def trainWindow(self, position, models, size):
        """
            Determine window's model and add predictions to training models.
        """
        stopWindow = Window(self.stopGrid, position, size)
        startWindow = Window(self.startGrid, position, size)
        stopWindowHash = stopWindow.toHash()
        model = models[self.steps][stopWindowHash]
        model.addPrediction(startWindow)
        models[self.steps][stopWindowHash] = model

    def train(self, allModels):
        """
            Run training on this test case, which will write predictions in the models.
        """
        for size, models in allModels.items():
            for height in range(1 - size, self.height):
                for width in range(1 - size, self.width):
                    self.trainWindow((height, width), models, size)

    def predict(self, allModels):
        """
            Predict start grid based on stop grid of this test case using models.
        """
        predictionGrid = np.zeros((self.height, self.width))
        for size, models in reversed(allModels.items()):
            for height in range(1 - size, self.height):
                for width in range(1 - size, self.width):
                    stopWindow = Window(self.stopGrid, (height, width), size)
                    stopWindowHash = stopWindow.toHash()
                    model = models[self.steps][stopWindowHash]
                    prediction = model.predict()
                    gStart, gEnd, wStart, wEnd = Window.overlap(predictionGrid, prediction, (height, width))
                    predictionGrid[gStart[0]:gEnd[0], gStart[1]:gEnd[1]] += prediction[wStart[0]:wEnd[0], wStart[1]:wEnd[1]] / (size ** 2)

        # print predictionGrid
        startGrid = np.array([1 if predictionGrid[h,w] >= 0.2 else 0 for h in range(self.height) for w in range(self.width)]).reshape((self.height, self.width))
        return startGrid
