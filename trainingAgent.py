import numpy as np
import csv
import time
import os
from testcase import TestCase
from modelStorage import ModelStorage

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

        # Delete previous log file
        try:
            os.remove('log.txt')
        except:
            pass

        # Create models storage for each humber of game steps (1...5)
        self.models = {index: ModelStorage(steps=index, size=windowSize) for index in range(minSteps, maxSteps + 1)}

    def loadModels(self, modelsFilename, log=True):
        """
            Load model values from the file for predictions
        """
        # Measure how much time takes to run
        timeStart = time.time()

        # Check that file is CSV
        if not modelsFilename.endswith(".csv"):
            modelsFilename += '.csv'

        # Read file. If file doesn't exist, Exception will be raised
        with open(modelsFilename, 'r') as modelsCSV:
            # Create CSV reader
            modelsCSVreader = csv.reader(modelsCSV)

            # Store the names of the fields
            fields = modelsCSVreader.next()

            if log:
                # Indicate training
                print("---- Loading models ----")

            # Read one row at a time and store the model
            for row in modelsCSVreader:
                # Track execution time of one case
                timeCaseStart = time.time()

                # Get number of steps to determine models storage
                steps = ModelStorage.getSteps(fields, row)

                # Add model to storage
                modelHash = self.models[steps].add(fields, row)

                timeCaseEnd = time.time()
                if log:
                    # Print hash of the model
                    print("Loading #%d for #%d steps took %.3f" % (modelHash, steps, (timeCaseEnd - timeCaseStart) * 1000.))

        timeEnd = time.time()
        if log:
            # Print time of execution
            print("Loading models took %.3f s" % (timeEnd - timeStart))
            with open('log.txt', 'a+') as log:
                log.write("Loading models took %.3f s\n" % (timeEnd - timeStart))


    def saveModels(self, modelsFilename, log=True):
        """
            Save trained models to the file for later use
        """
        # Measure how much time takes to run
        timeStart = time.time()

        # Check that file is CSV
        if not modelsFilename.endswith(".csv"):
            trainFilename += '.csv'

        # Create models file (delete previous if exists)
        with open(modelsFilename, 'w+') as modelsCSV:
            # Create CSV writer for models file
            modelsCSVwriter = csv.writer(modelsCSV)

            # Write names of the fields to models file
            modelsCSVwriter.writerow(['hash', 'steps', 'size', 'occurrences'] + ['model.' + str(i + 1) for i in range(self.windowSize ** 2)])

            if log:
                # Indicate saving
                print("---- Saving models ----")

            # Go through each model storage
            for steps in self.models.keys():
                # Get model storage
                modelStorage = self.models[steps]

                # Go through each model and write it in the file
                for modelHash, model in modelStorage.items():
                    # Track execution time of one case
                    timeCaseStart = time.time()

                    # Write model values to models file
                    modelsCSVwriter.writerow(model.createRow())

                    timeCaseEnd = time.time()
                    if log:
                        # Print hash of the model
                        print("Saving model #%d for #%d steps took %.3f ms" % (modelHash, steps, (timeCaseEnd - timeCaseStart) * 1000.))

        timeEnd = time.time()
        if log:
            # Print time of execution
            print("Saving models took %.3f s" % (timeEnd - timeStart))
            with open('log.txt', 'a+') as log:
                log.write("Saving models took %.3f s\n" % (timeEnd - timeStart))



    def train(self, trainFilename, log=True):
        """
            Train agent to predict start grid based on training data from the csv file.
        """
        # Measure how much time takes to run
        timeStart = time.time()

        # Check that file is CSV
        if not trainFilename.endswith(".csv"):
            trainFilename += '.csv'

        # Read file. If file doesn't exist, Exception will be raised
        with open(trainFilename, 'r') as trainCSV:
            # Create CSV reader
            trainCSVreader = csv.reader(trainCSV)

            # Store the names of the fields
            fields = trainCSVreader.next()

            if log:
                # Indicate training
                print("---- Training ----")

            # Read one row at a time and perform training
            for row in trainCSVreader:
                # Track execution time of one case
                timeCaseStart = time.time()

                # Create a testcase
                testcase = TestCase(fields, row, self.height, self.width, self.windowSize, isTraining=True)

                # Train this test case
                testcase.train(self.models)

                timeCaseEnd = time.time()
                if log:
                    # Print hash of the model
                    print("Training #%d took %.3f ms" % (testcase.getId(), (timeCaseEnd - timeCaseStart) * 1000.))

        timeEnd = time.time()
        if log:
            # Print time of execution
            print("Training took %.3f s or %.3f min" % (timeEnd - timeStart, (timeEnd - timeStart) / 60.))
            with open('log.txt', 'a+') as log:
                log.write(("Training took %.3f s or %.3f min\n" % (timeEnd - timeStart, (timeEnd - timeStart) / 60.)))


    def predict(self, predictFilename, outputFilename, log=True):
        """
            Predict start grid from stop grid from the csv file.
        """
        # Measure how much time takes to run
        timeStart = time.time()

        # Check that file is CSV
        if not predictFilename.endswith(".csv"):
                predictFilename += '.csv'

        # Read file. If file doesn't exist, Exception will be raised
        with open(predictFilename, 'r') as predictCSV:
            # Create CSV reader for predition file
            predictCSVreader = csv.reader(predictCSV)

            # Store the names of the fields
            fields = predictCSVreader.next()

            # Check that file is CSV
            if not outputFilename.endswith(".csv"):
                outputFilename += '.csv'

            # Create output file (delete previous if exists)
            with open(outputFilename, 'w+') as outputCSV:
                # Create CSV writer for output file
                outputCSVwriter = csv.writer(outputCSV)

                # Write names of the fields to output file
                outputCSVwriter.writerow(['id'] + ['start.' + str(i + 1) for i in range(self.height * self.width)])

                if log:
                    # Indicate testing
                    print("---- Testing ----")


                # Read one row at a time and predict start grid
                for row in predictCSVreader:
                    # Track execution time of one case
                    timeCaseStart = time.time()

                    # Create a testcase
                    testcase = TestCase(fields, row, self.height, self.width, self.windowSize, isTraining=False)

                    # Predict start grid for the test case
                    startGrid = testcase.predict(self.models)

                    # Write values to output file
                    outputCSVwriter.writerow([testcase.getId()] + list(startGrid.reshape(self.height * self.width)))

                    timeCaseEnd = time.time()
                    if log:
                        # Print id of testing episode
                        print("Testing #%d took %.3f ms" % (testcase.getId(), (timeCaseEnd - timeCaseStart) * 1000.))

        timeEnd = time.time()
        if log:
            # Print time of execution
            print("Testing took %.3f s or %.3f min" % (timeEnd - timeStart, (timeEnd - timeStart) / 60.))
            with open('log.txt', 'a+') as log:
                log.write("Testing took %.3f s or %.3f min\n" % (timeEnd - timeStart, (timeEnd - timeStart) / 60.))
