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
        if windowSize > height or windowSize > width:
            raise Exception("Window size exceeds the dimentions of the grid")

        self.height = height
        self.width = width
        self.windowSize = windowSize

        try:
            os.remove('log.txt')
        except:
            pass

        self.models = {index: ModelStorage(steps=index, size=windowSize) for index in range(minSteps, maxSteps + 1)}

    def loadModels(self, modelsFilename, log=True):
        """
            Load model values from the file for predictions
        """
        timeStart = time.time()

        if not modelsFilename.endswith(".csv"):
            modelsFilename += '.csv'

        with open(modelsFilename, 'r') as modelsCSV:
            modelsCSVreader = csv.reader(modelsCSV)
            fields = modelsCSVreader.next()

            if log:
                print("---- Loading models ----")

            for row in modelsCSVreader:
                timeCaseStart = time.time()
                steps = ModelStorage.getSteps(fields, row)
                modelHash = self.models[steps].add(fields, row)
                self.windowSize = self.models[steps][modelHash].size
                timeCaseEnd = time.time()
                if log:
                    print("Loading model #%d for #%d steps took %.3f ms" % (modelHash, steps, (timeCaseEnd - timeCaseStart) * 1000.))

        timeEnd = time.time()
        if log:
            print("Loading models took %.3f s" % (timeEnd - timeStart))
            with open('log.txt', 'a+') as log:
                log.write("Loading models took %.3f s\n" % (timeEnd - timeStart))


    def saveModels(self, modelsFilename, log=True):
        """
            Save trained models to the file for later use
        """
        timeStart = time.time()

        if not modelsFilename.endswith(".csv"):
            trainFilename += '.csv'

        with open(modelsFilename, 'w+') as modelsCSV:
            modelsCSVwriter = csv.writer(modelsCSV)
            modelsCSVwriter.writerow(['hash', 'steps', 'size', 'occurrences'] + ['model.' + str(i + 1) for i in range(self.windowSize ** 2)])

            if log:
                print("---- Saving models ----")

            for steps in self.models.keys():
                modelStorage = self.models[steps]

                for modelHash, model in modelStorage.items():
                    timeCaseStart = time.time()
                    modelsCSVwriter.writerow(model.createRow())
                    timeCaseEnd = time.time()
                    if log:
                        print("Saving model #%d for #%d steps took %.3f ms" % (modelHash, steps, (timeCaseEnd - timeCaseStart) * 1000.))

        timeEnd = time.time()
        if log:
            print("Saving models took %.3f s" % (timeEnd - timeStart))
            with open('log.txt', 'a+') as log:
                log.write("Saving models took %.3f s\n" % (timeEnd - timeStart))



    def train(self, trainFilename, log=True):
        """
            Train agent to predict start grid based on training data from the csv file.
        """
        timeStart = time.time()

        if not trainFilename.endswith(".csv"):
            trainFilename += '.csv'

        with open(trainFilename, 'r') as trainCSV:
            trainCSVreader = csv.reader(trainCSV)
            fields = trainCSVreader.next()

            if log:
                print("---- Training ----")

            for row in trainCSVreader:
                timeCaseStart = time.time()
                testcase = TestCase(fields, row, self.height, self.width, self.windowSize, isTraining=True)
                testcase.train(self.models)
                timeCaseEnd = time.time()
                if log:
                    print("Training #%d took %.3f ms" % (testcase.getId(), (timeCaseEnd - timeCaseStart) * 1000.))

        timeEnd = time.time()
        if log:
            print("Training took %.3f s or %.3f min" % (timeEnd - timeStart, (timeEnd - timeStart) / 60.))
            with open('log.txt', 'a+') as log:
                log.write(("Training took %.3f s or %.3f min\n" % (timeEnd - timeStart, (timeEnd - timeStart) / 60.)))


    def predict(self, predictFilename, outputFilename, log=True):
        """
            Predict start grid from stop grid from the csv file.
        """
        timeStart = time.time()

        if not predictFilename.endswith(".csv"):
                predictFilename += '.csv'

        with open(predictFilename, 'r') as predictCSV:
            predictCSVreader = csv.reader(predictCSV)
            fields = predictCSVreader.next()
            if not outputFilename.endswith(".csv"):
                outputFilename += '.csv'

            with open(outputFilename, 'w+') as outputCSV:
                outputCSVwriter = csv.writer(outputCSV)
                outputCSVwriter.writerow(['id'] + ['start.' + str(i + 1) for i in range(self.height * self.width)])

                if log:
                    print("---- Testing ----")

                for row in predictCSVreader:
                    timeCaseStart = time.time()
                    testcase = TestCase(fields, row, self.height, self.width, self.windowSize, isTraining=False)
                    startGrid = testcase.predict(self.models)
                    outputCSVwriter.writerow([testcase.getId()] + list(startGrid.reshape(self.height * self.width)))
                    timeCaseEnd = time.time()
                    if log:
                        print("Testing #%d took %.3f ms" % (testcase.getId(), (timeCaseEnd - timeCaseStart) * 1000.))

        timeEnd = time.time()
        if log:
            print("Testing took %.3f s or %.3f min" % (timeEnd - timeStart, (timeEnd - timeStart) / 60.))
            with open('log.txt', 'a+') as log:
                log.write("Testing took %.3f s or %.3f min\n" % (timeEnd - timeStart, (timeEnd - timeStart) / 60.))
