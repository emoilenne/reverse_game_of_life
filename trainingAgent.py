import numpy as np
import csv
import time
import os
from testcase import TestCase
from modelStorage import ModelStorage
from map import Map

class TrainingAgent:
    def __init__(self, height, width, minSteps, maxSteps):
        """
            Initiate training agent by grid dimentions, size of the window that
            will scan training data and create storage for training models.
        """
        self.height = height
        self.width = width
        self.total = height * width
        self.modelsFilename = 'csv/models'

        try:
            os.remove('log.txt')
        except:
            pass

        models4 = {index: ModelStorage(steps=index, size=4) for index in range(minSteps, maxSteps + 1)}
        models5 = {index: ModelStorage(steps=index, size=5) for index in range(minSteps, maxSteps + 1)}
        models6 = {index: ModelStorage(steps=index, size=6) for index in range(minSteps, maxSteps + 1)}
        self.models = {4: models4, 5: models5, 6: models6}


    def loadModels(self):
        """
            Load model values from the files for predictions
        """

        for size in range(4,7):
            timeStart = time.time()

            with open(self.modelsFilename + str(size) + '.csv', 'r') as modelsCSV:
                modelsCSVreader = csv.reader(modelsCSV)
                fields = modelsCSVreader.next()

                print("---- Loading models %d ----" % size)

                for row in modelsCSVreader:
                    timeCaseStart = time.time()
                    steps = ModelStorage.getSteps(fields, row)
                    modelHash = self.models[size][steps].add(fields, row)
                    timeCaseEnd = time.time()
                    print("Loading model #%d for #%d steps with window %d x %d took %.3f ms" % (modelHash, steps, size, size, (timeCaseEnd - timeCaseStart) * 1000.))

            timeEnd = time.time()
            print("Loading models %d took %.3f s" % (size, timeEnd - timeStart))
            with open('log.txt', 'a+') as log:
                log.write("Loading models %d took %.3f s\n" % (size, timeEnd - timeStart))


    def saveModels(self):
        """
            Save trained models to the file for later use
        """
        for size in range(4,7):
            timeStart = time.time()

            with open(self.modelsFilename + str(size) + '.csv', 'w+') as modelsCSV:
                modelsCSVwriter = csv.writer(modelsCSV)
                modelsCSVwriter.writerow(['hash', 'steps', 'occurrences'] + ['model.' + str(i + 1) for i in range(size ** 2)])

                print("---- Saving models %d ----" % size)

                for steps in self.models[size].keys():
                    modelStorage = self.models[size][steps]

                    for modelHash, model in modelStorage.items():
                        timeCaseStart = time.time()
                        modelsCSVwriter.writerow(model.createRow())
                        timeCaseEnd = time.time()
                        print("Saving model #%d for #%d steps with window %d x %d took %.3f ms" % (modelHash, steps, size, size, (timeCaseEnd - timeCaseStart) * 1000.))

            timeEnd = time.time()
            print("Saving models %d took %.3f s" % (size, timeEnd - timeStart))
            with open('log.txt', 'a+') as log:
                log.write("Saving models %d took %.3f s\n" % (size, timeEnd - timeStart))



    def train(self, trainFilename):
        """
            Train agent to predict start grid based on training data from the csv file.
        """
        self.tryLoadModels()
        timeStart = time.time()

        if not trainFilename.endswith(".csv"):
            trainFilename += '.csv'

        with open(trainFilename, 'r') as trainCSV:
            trainCSVreader = csv.reader(trainCSV)
            fields = trainCSVreader.next()

            print("---- Training ----")

            for row in trainCSVreader:
                timeCaseStart = time.time()
                testcase = TestCase(fields, row, self.height, self.width, isTraining=True)
                testcase.train(self.models)
                timeCaseEnd = time.time()
                print("Training #%d took %.3f ms" % (testcase.getId(), (timeCaseEnd - timeCaseStart) * 1000.))

        timeEnd = time.time()
        print("Training took %.3f s or %.3f min" % (timeEnd - timeStart, (timeEnd - timeStart) / 60.))
        with open('log.txt', 'a+') as log:
            log.write(("Training took %.3f s or %.3f min\n" % (timeEnd - timeStart, (timeEnd - timeStart) / 60.)))


    def predict(self, predictFilename, outputFilename):
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

                print("---- Testing ----")

                for row in predictCSVreader:
                    timeCaseStart = time.time()
                    testcase = TestCase(fields, row, self.height, self.width, isTraining=False)
                    startGrid = testcase.predict(self.models)
                    outputCSVwriter.writerow([testcase.getId()] + list(startGrid.reshape(self.height * self.width)))
                    timeCaseEnd = time.time()
                    print("Testing #%d took %.3f ms" % (testcase.getId(), (timeCaseEnd - timeCaseStart) * 1000.))

        timeEnd = time.time()
        print("Testing took %.3f s or %.3f min" % (timeEnd - timeStart, (timeEnd - timeStart) / 60.))
        with open('log.txt', 'a+') as log:
            log.write("Testing took %.3f s or %.3f min\n" % (timeEnd - timeStart, (timeEnd - timeStart) / 60.))

    def tryLoadModels(self):
        models_exist = sum([not os.path.isfile(self.modelsFilename + str(windowSize) + '.csv') for windowSize in range(4, 7)]) == 0
        if models_exist:
            load = raw_input("There are existing models. Do you want to load them in addition to training? (y/n): ")
            if load == 'y':
                try:
                    self.loadModels()
                    print "Successfully loaded models, running training on top of them..."
                except:
                    print "Couldn't laod previous models, running training without them..."
            elif load == 'n':
                pass
            else:
                print "Please select y or n."
                exit(1)

    def generateAndTrain(self, count):
        self.tryLoadModels()
        fields = ['id', 'delta'] + ['start.' + str(i + 1) for i in range(self.total)] + \
                                    ['stop.' + str(i + 1) for i in range(self.total)]
        totalTimeStart = time.time()
        mapgen = Map(self.height, self.width)
        for id in range(count):
            timeStart = time.time()
            while True:
                row = [str(id + 1)]
                mapgen.generate()
                steps = mapgen.getSteps()
                row.append(str(steps))
                row.extend(mapgen.getValues())
                mapgen.step(steps)
                row.extend(mapgen.getValues())
                if mapgen.aliveCells() != 0:
                    break
            timeEnd = time.time()
            print("Generating map #%d took %.3f ms" % (id, (timeEnd - timeStart) * 1000.))

            timeStart = time.time()
            testcase = TestCase(fields, row, self.height, self.width, isTraining=True)
            testcase.train(self.models)
            timeEnd = time.time()
            print("Training map #%d took %.3f ms" % (id, (timeEnd - timeStart) * 1000.))

        totalTimeEnd = time.time()
        print("Generating and training maps took %.3f s or %.3f min" % (totalTimeEnd - totalTimeStart, (totalTimeEnd - totalTimeStart) / 60.))
        with open('log.txt', 'a+') as log:
            log.write("Generating and training maps took %.3f s or %.3f min\n" % (totalTimeEnd - totalTimeStart, (totalTimeEnd - totalTimeStart) / 60.))
