import numpy as np
import csv
from window import Window, Transformation

class ModelStorage(dict):
    def __getitem__(self, idx):
        self.setdefault(idx, TrainingModel())
        return dict.__getitem__(self, idx)

class TrainingModel:
    def __init__(self):
        """
            Initiate training model, which stores predictions of the start grid
            and number of occurances of this model in the testing.
        """
        self.occurances = 0
        self.data = None

    def addPrediction(self, window): #TODO if size is not needed, can change window to data
        """
            Add start grid values of this model to predictions.
        """
        self.occurances += 1
        self.data += window.data

    def predict(self, transformation, position, height, width):
        """
            Return predictions grid, that was formed on the training data.
            Grid has shape (height, width, 2), where every [height, width] cell
            represents set [countCellAlive, occurancesOfCell]: countCellAlive indicates
            how many times cell was alive in total of occurancesOfCell cases
            during training.
        """
        # Create empty prediction grid with shape (height, width, 2)
        predictionGrid = np.zeros(height * width * 2).reshape(height, width, 2)

        # Transform model according to transformation needed
        model = Transformation.do[transformation](self.data)

        # Create model3d that stores [countCellAlive, occurancesOfCell] for each element of model,
        # where occurancesOfCell == self.occurances
        model3d = np.array([model[h,w] if position == 0 else self.occurances for h in range(self.size) for w in range(self.size) for position in range(2)]).reshape(self.size, self.size, 2)

        # Add model on the prediction grid at the position
        predictionGrid[position[0]:position[0] + self.size, position[1]: position[1] + self.size] = model3d

        return predictionGrid


class TestCase:
    def __init__(self, fields, row, height, width, isTraining, windowSize=4):
        """
            Initiate testcase with a row containing id, steps and grids
        """
        # Store data from row
        try:
            #TODO can just read values as id,delta,start.1,start.2,... without reading field names
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
        totalValues = height * width * 2 if isTraining else height * width
        if len(self.data) != totalValues:
            raise Exception("Dimentions of the grid don't match values in the CSV file.")

        # Create grids
        self.stopGrid = self.createGrid('stop')
        if isTraining:
            self.startGrid = self.createGrid('start')

    def createGrid(self, name):
        """
            Create a grid with specific name from data stored in this test case.
        """
        grid = []
        try:
            # Read values from the file
            for index in range(self.height * self.width):
                grid.append(self.data[name + '.' + str(index + 1)])

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
        startWindow = Transformation.do[transformation](startWindow)

        # Add predictions to stored model
        model.addPrediction(startWindow)

        # Save model (needed if this was the first time the model has found)
        models[steps][stopWindowHash] = model

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
        predictionGrid = 0

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
        probabilityGrid = np.array([predictionGrid[h,w,0] / float(predictionGrid[h,w,1]) for h in range(self.height) for w in range(self.width)]).reshape(height, width)

        # Create start grid based on probability grid
        startGrid = np.array([round(probabilityGrid[h,w]) for h in range(self.height) for w in range(self.width)]).reshape(height, width)
        return startGrid

    def getId(self):
        return self.id

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

    def train(self, trainFilename):
        """
            Train agent to predict start grid based on training data from the csv file.
        """
        # Check that file is CSV
        if not trainFilename.endswith(".csv"):
            raise Exception("File should be in the CSV format.")

        # Read file. If file doesn't exist, Exception will be raised
        with open(trainFilename, 'r') as csvfile:
            # Create CSV reader
            csvreader = csv.reader(csvfile)

            # Store the names of the fields
            fields = csvreader.next()

            # Read one row at a time and perform training
            for row in csvreader:
                # Create a testcase
                testcase = TestCase(fields, row, self.height, self.width, isTraining=True)

                # Train this test case
                testcase.train(self.models)

    def predict(self, predictFilename, outputFilename):
        """
            Predict start grid from stop grid from the csv file.
        """
        # Check that file is CSV
        if not predictFilename.endswith(".csv"):
            raise Exception("File should be in the CSV format.")

        # Read file. If file doesn't exist, Exception will be raised
        with open(predictFilename, 'r') as predictCSV:
            # Create CSV reader for predition file
            predictCSVreader = csv.reader(predictCSV)

            # Store the names of the fields
            fields = predictCSVreader.next()

            # Create output file (delete previous if exists)
            with open(outputFilename, 'w+') as outputCSV:
                # Create CSV writer for output file
                outputCSVwriter = csv.writer(outputCSV)

                # Write names of the fields to output file
                outputCSVwriter.writerow(['id'] + ['start.' + str(i + 1) for i in range(self.height * self.width)])

                # Read one row at a time and predict start grid
                for row in predictCSVreader:
                    # Create a testcase
                    testcase = TestCase(fields, row, self.height, self.width, isTraining=False)

                    # Predict start grid for the test case
                    startGrid = testcase.predict(models)

                    # Write values to output file
                    outputCSVwriter.writerow([testcase.getId()] + list(startGrid.reshape(self.height * self.width)))
