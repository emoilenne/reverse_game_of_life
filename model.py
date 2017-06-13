import numpy as np
import csv
from window import Window, Transformation

class ModelStorage(dict):
    def __getitem__(self, idx):
        self.setdefault(idx, TrainingModel())
        return dict.__getitem__(self, idx)

class TestCase:
    def __init__(self, fields, row, height, width, isTraining, windowSize=4):
        """
            Initiate testcase with a row containing id, steps and grids
        """
        # Store data from row
        try:
            self.data = {fields[index]: row[index] for index in range(len(fields))}
            self.id = self.data['id']
            del(self.data['id'])
            self.steps = self.data['delta']
            del(self.data['delta'])
        except:
            raise Exception("CSV file is not valid.")

        # Check if grids are valid
        self.height = height
        self.width = width
        self.windowSize = windowSize
        self.cells = height * width * 2 if isTraining else height * width
        if len(self.data) != self.cells:
            raise Exception("Dimentions of the grid don't match values in the CSV file.")
        # Create grids
        self.stopGrid = self.createGrid('stop')
        #TODO create empty start grid for test or don't create at all ?
        if isTraining:
            self.startGrid = self.createGrid('start')

    def createGrid(self, name):
        """
            Create a grid with specific name from data stored in this test case.
        """
        grid = []
        try:
            # Read values from the file
            for index in range(self.cells):
                grid.append(self.data[name + '.' + str(index + 1)])

            # Creade NumPy array from obtained values
            grid = np.array(grid).reshape(self.height, self.width)
        except Exception as e:
            raise Exception("Unable to create %s grid" % name)
        return grid

    def trainWindow(startPoint, startGrid, stopGrid, steps, models):
        """
            Determine window's model and add predictions to training models.
        """
        # Create a window from stop grid
        stopWindow = Window(stopGrid, startPoint, self.windowSize)

        # Create same window from start grid
        startWindow = Window(startGrid, startPoint, self.windowSize)

        # Determine model of stop window (get hash value and transformation needed for hash value)
        transformation, stopWindowHash = stopWindow.toHash()

        # Obtain stored model for this window
        model = models[steps][stopWindowHash]

        # Transform start window the same as stop window was transformed in hash
        startWindow = Transformation.do[transformation](startWindow)

        # Add predictions to stored model
        model.addPrediction(startWindow)

    def train(self, models):
        """
            Run training on this test case, which will write predictions in the models.
        """
        pass

    def predict(self, models):
        """
            Predict start grid based on stop grid of this test case using models.
        """
        pass

class TrainingModel:
    def __init__(self):
        """
            Initiate training model, which stores predictions of the start grid
            and number of occurances of this model in the testing.
        """
        self.occurances = 0
        self.data = None
        self.size = 0

    def addPrediction(self, window): #TODO if size is not needed, can change window to data
        """
            Add start grid values of this model to predictions.
        """
        self.occurances += 1
        if self.data:
            self.data += window.data
        else:
            self.data = window.data
            self.size = window.size

    def predict(self):
        """
            Return predictions array, that was formed on the training data.
        """
        return self.data #TODO maybe divide by occurances

class TrainingAgent:
    def __init__(self, height, width, minSteps, maxSteps, windowSize = 4):
        """
            Initiate training agent by grid dimentions, size of the window that
            will scan training data and create storage for training models.
        """
        # Check if the window size exceeds grid dimentions
        if windowSize > height or windowSize > width:
            raise Exception("Window size exceeds the dimentions of the grid")

        self.height = height
        self.width = width
        self.windowSize = windowSize

        # Create models storage for each humber of game steps (1...5)
        self.models = {index: ModelStorage() for index in range(minSteps, maxSteps + 1)}

    def train(self, filename):
        """
            Train agent to predict start grid based on training data from the csv file.
        """

        # Check that file is SCV
        if not filename.endswith(".csv"):
            raise Exception("File should be in the CSV format.")

        # Read file. If file doesn't exist, Exception will be raised
        with open(filename, 'r') as csvfile:
            # Create CSV reader
            csvreader = csv.reader(csvfile)

            # Store the names of the fields
            fields = csvreader.next()

            # Read one row at a time and perform training
            for row in csvreader:
                # Create a testcase
                testcase = TestCase(fields, row, self.height, self.width, isTraining=True)

                # Train this test case
                testcase.train()

    def predict(self, filename):
        """
            Predict start grid from stop grid from the csv file.
        """
        pass
