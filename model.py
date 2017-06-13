import numpy as np
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
        self.size = 0

    def addPrediction(self, window):
        """
            Add start grid values of this model to predictions.
        """
        self.occurances += 1
        if self.data:
            self.data += window.data
        else
            self.data = window.data
            self.size = window.size

    def predict(self):
        """
            Return predictions array, that was formed on the training data.
        """
        return self.data #TODO maybe divide by occurances

class TrainingAgent:
    def __init__(self, height=20, width=20, windowSize=4, maxSteps=5):
        """
            Initiate training agent by grid dimentions, size of the window that
            will scan training data and create storage for training models.
        """
        self.height = height
        self.width = width
        self.windowSize = windowSize
        self.models = []
        for index in range(maxSteps):
            self.models = ModelStorage()

    def trainWindow(startPoint, startGrid, stopGrid, steps):
        """
            Determine window's model and add predictions to training models.
        """
        # Create a window from stop grid
        stopWindow = Window(stopGrid, startPoint, self.windowSize)

        # Create same window from start grid
        startWindow = Window(startGrid, startPoint, self.windowSize)

        # Determine model of stop window (get hash value and transformation needed for hash value)
        transformation, stopWindowHash = stopWindow.toHash()

        # Obrain stored model for this window
        model = self.models[steps][stopWindowHash]

        # Transform start window the same as stop window was transformed in hash
        startWindow = Transformation.do[transformation]

        # Add predictions to stored model
        model.addPrediction(startWindow)

    def train(self, rows):
        """
            Train agent to predict start grid based on training data.
        """
